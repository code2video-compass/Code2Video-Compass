"""Gemini generateContent adapter for local MP4 inputs (inlineData).

Sends the judged video as base64 inlineData to the official Gemini API
(`POST {base}/v1beta/models/{model}:generateContent`), authenticated with an
API key in the `x-goog-api-key` header. Configure with the environment
variables `GEMINI_API_KEY` (required, from Google AI Studio) and optionally
`GEMINI_API_BASE` (default https://generativelanguage.googleapis.com, e.g. a
self-hosted mirror); an optional `.env` file next to this script is loaded
first and never overrides existing environment variables.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests

HERE = Path(__file__).resolve().parent
_ENV_LOCK = threading.Lock()


def load_env():
    """Read an optional .env next to this script once, without overriding."""
    with _ENV_LOCK:
        path = HERE / ".env"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def redact(value) -> str:
    """Strip API keys from anything printed or persisted."""
    text = str(value)
    secret = os.environ.get("GEMINI_API_KEY")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return re.sub(r"AIza[A-Za-z0-9_-]{30,}", "[REDACTED]", text)


def normalize_usage(usage: dict, native: dict | None = None) -> dict:
    """Normalize native Gemini usage, including thinking exactly once."""
    native = native or usage.get("native_usage_metadata") or {}
    if not native:
        # Chat-compatible completion_tokens already includes reasoning.
        return dict(usage)
    result = dict(usage)
    answer = native.get("candidatesTokenCount")
    thinking = native.get("thoughtsTokenCount", 0)
    valid = lambda value: isinstance(value, int) and not isinstance(value, bool) and value >= 0
    completion = answer + thinking if valid(answer) and valid(thinking) else None
    details = dict(usage.get("completion_tokens_details") or {})
    details.update(reasoning_tokens=thinking, answer_tokens=answer)
    result.update(
        prompt_tokens=native.get("promptTokenCount"),
        completion_tokens=completion,
        completion_tokens_details=details,
        completion_tokens_includes_reasoning=True,
        total_tokens=native.get("totalTokenCount"),
        cached_input_tokens=native.get("cachedContentTokenCount"),
        prompt_tokens_details={"cached_tokens": native.get("cachedContentTokenCount")},
        native_usage_metadata=native,
    )
    return result


class GeminiNativeVideoAdapter:
    """Send local MP4 bytes using Gemini's contents/parts/inlineData schema."""

    def __init__(self, config, output=None):
        del output  # Kept for interface compatibility.
        load_env()
        self.config = config
        self.provider = config["provider"]
        self.policy = config["request"]
        key_name = self.provider.get("api_key_env", "GEMINI_API_KEY")
        self.key = os.environ.get(key_name)
        if not self.key:
            raise RuntimeError(f"Configured API key is missing: {key_name}")

        base_env = self.provider.get("base_url_env", "GEMINI_API_BASE")
        base = os.environ.get(base_env, "https://generativelanguage.googleapis.com").rstrip("/")
        if base.endswith("/v1beta"):
            base = base[:-len("/v1beta")]
        self.model = config["model"]["name"]
        model_path = quote(self.model, safe=".-_")
        self.endpoint = f"{base}/v1beta/models/{model_path}:generateContent"
        if not self.endpoint.startswith("https://"):
            raise RuntimeError("Gemini endpoint must use HTTPS")
        self.session = requests.Session()
        self.session.trust_env = False
        self.headers = {
            "x-goog-api-key": self.key,
            "Content-Type": "application/json",
        }
        self.http_requests = 0
        self.attempts = []

    def close(self):
        self.session.close()

    def build_request(self, video_bytes: bytes, prompt: str, context=None) -> dict:
        max_bytes = int(self.policy.get("max_inline_video_bytes", 40_000_000))
        if len(video_bytes) > max_bytes:
            raise RuntimeError(f"Video is {len(video_bytes)} bytes; inline limit is {max_bytes}")
        fps = float(self.policy.get("fps", 2.0))
        if not 0 < fps <= 24:
            raise RuntimeError(f"Video FPS must be greater than 0 and at most 24; got {fps}")
        text = prompt
        if context is not None:
            text += "\n\nREQUEST CONTEXT\n" + json.dumps(context, ensure_ascii=False)
        return {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": text},
                    {
                        "inlineData": {
                            "mimeType": "video/mp4",
                            "data": base64.b64encode(video_bytes).decode("ascii"),
                        },
                        "videoMetadata": {"fps": fps},
                    },
                ],
            }],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": int(self.policy.get("max_completion_tokens", 10_000)),
            },
        }

    def _request(self, payload: dict) -> dict:
        retries = int(self.policy.get("max_retries", 2))
        for attempt in range(retries + 1):
            started = time.monotonic()
            self.http_requests += 1
            try:
                response = self.session.post(
                    self.endpoint, headers=self.headers, json=payload,
                    timeout=(30, 600), allow_redirects=False,
                )
                elapsed = round(time.monotonic() - started, 3)
                self.attempts.append({"attempt": attempt + 1,
                                      "http_status": response.status_code,
                                      "latency_s": elapsed})
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {redact(response.text[:2000])}"
                    )
                return response.json()
            except requests.RequestException as exc:
                self.attempts.append({"attempt": attempt + 1,
                                      "error": redact(str(exc)),
                                      "latency_s": round(time.monotonic() - started, 3)})
                if attempt < retries and isinstance(
                    exc, (requests.Timeout, requests.ConnectionError)
                ):
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(redact(str(exc))) from exc
        raise RuntimeError("Gemini request exhausted retries")

    @staticmethod
    def _response_text(body: dict) -> tuple[str, str]:
        candidates = body.get("candidates") or []
        if not candidates:
            feedback = body.get("promptFeedback") or {}
            raise RuntimeError(f"No response candidate; promptFeedback={feedback}")
        candidate = candidates[0]
        parts = ((candidate.get("content") or {}).get("parts") or [])
        text = "\n".join(part["text"] for part in parts
                         if isinstance(part, dict) and isinstance(part.get("text"), str))
        if not text.strip():
            raise RuntimeError("Gemini returned no text in candidate content")
        return text, str(candidate.get("finishReason", "unknown"))

    @staticmethod
    def _normalized_usage(body: dict) -> tuple[dict, int | None]:
        native = body.get("usageMetadata") or {}
        prompt_details = native.get("promptTokensDetails") or []
        video_tokens = None
        if isinstance(prompt_details, list):
            found = [int(d.get("tokenCount", 0) or 0) for d in prompt_details
                     if isinstance(d, dict) and str(d.get("modality", "")).upper() == "VIDEO"]
            if found:
                video_tokens = sum(found)
        usage = normalize_usage({}, native)
        return usage, video_tokens

    def score(self, video, video_hash, prompt, context, schema=None):
        del schema  # The scoring prompt requests JSON; the native payload shape is fixed.
        video = Path(video)
        data = video.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != video_hash:
            raise RuntimeError("Local MP4 is missing or changed before assessment")
        payload = self.build_request(data, prompt, context)
        body = self._request(payload)
        text, finish_status = self._response_text(body)
        usage, video_tokens = self._normalized_usage(body)
        return {
            "raw_response": body,
            "text": text,
            "usage": usage,
            "actual_model": body.get("modelVersion", self.model),
            "finish_status": finish_status,
            "request_settings": {
                "endpoint": self.endpoint,
                "model": self.model,
                "video_sha256": actual_hash,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "transport": "Official Gemini API generateContent, inlineData local MP4",
                "provider_video_tokens": video_tokens,
                "sampling_request": {
                    "videoMetadata": {"fps": float(self.policy.get("fps", 2.0))},
                    "effective_fps_confirmed": False,
                },
                "media_resolution_request": None,
                "media_resolution_effective_confirmed": False,
            },
        }
