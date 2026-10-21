"""Media helpers for the judge: hashing, probing, and upload-size derivatives.

`ffmpeg`/`ffprobe` are taken from PATH. Videos within the inline-size
threshold are uploaded unchanged. Larger videos get an upload-only derivative
that keeps full duration, resolution, and frame rate: first a bit-for-bit
remux with only audio re-encoded to AAC; if still too large, a same-shape
H.264 encode with a size-limited bitrate. The source file is never modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

VERSION = "inline_full_fps_full_resolution_adaptive_v3"
DEFAULTS = {
    "enabled": True,
    "threshold_bytes": 40_000_000,
    "target_bytes": 35_000_000,
    "quality_floor_psnr_db": 25.0,
    "audio_bitrate": "96k",
    "video_encoder": "libx264",  # paper runs used h264_nvenc on an NVIDIA GPU
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ffprobe_json(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration,size:stream=codec_type,codec_name,width,height,"
                        "avg_frame_rate,r_frame_rate,nb_frames,sample_rate,channels",
                        "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if p.returncode:
        raise RuntimeError(f"ffprobe failed: {p.stderr[-1000:]}")
    result = json.loads(p.stdout)
    streams = result.get("streams", [])
    fmt = result.get("format", {})
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        raise RuntimeError("No video stream in MP4")
    v = videos[0]
    return {
        "duration_s": float(fmt.get("duration", 0) or 0),
        "bytes": int(fmt.get("size", 0) or Path(path).stat().st_size),
        "video_codec": v.get("codec_name"),
        "width": int(v.get("width", 0) or 0),
        "height": int(v.get("height", 0) or 0),
        "fps": v.get("avg_frame_rate"),
        "r_fps": v.get("r_frame_rate"),
        "nb_frames": v.get("nb_frames"),
        "audio_streams": [s for s in streams if s.get("codec_type") == "audio"],
    }


def probe_video(path):
    """Lightweight probe used for the job record (plain ffprobe, 60 s timeout)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration,size,format_name:stream=index,codec_type,codec_name,"
                        "width,height,avg_frame_rate,r_frame_rate",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(p.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("No video stream")

    def parse_fps(value):
        try:
            result = float(Fraction(value))
            return result if math.isfinite(result) and result > 0 else None
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    return dict(duration=duration, filesize=int(fmt.get("size") or Path(path).stat().st_size),
                container=fmt.get("format_name"), codec=video.get("codec_name"),
                width=video.get("width"), height=video.get("height"),
                fps=parse_fps(video.get("avg_frame_rate")) or parse_fps(video.get("r_frame_rate")),
                has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])))


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _quality_probe(source, derivative, duration, width, height):
    """Compare five timestamp-aligned full-resolution frames; guardrail, not a proof."""
    expected_bytes = int(width) * int(height) * 3
    psnrs = []
    for ratio in (0.05, 0.25, 0.5, 0.75, 0.95):
        t = min(max(0.05, duration * ratio), max(0.05, duration - 0.05))
        frames = []
        for media in (source, derivative):
            p = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-hwaccel", "none",
                 "-ss", f"{t:.6f}", "-i", str(media), "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                capture_output=True, check=False,
            )
            if p.returncode or len(p.stdout) != expected_bytes:
                detail = p.stderr.decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(f"Could not inspect quality sample at {t:.2f}s: {detail}")
            frames.append(p.stdout)
        squared_error = sum((a - b) * (a - b) for a, b in zip(frames[0], frames[1]))
        mse = squared_error / expected_bytes
        psnrs.append(99.0 if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse))
    return {"sample_times_s": [round(duration * r, 3) for r in (0.05, 0.25, 0.5, 0.75, 0.95)],
            "psnr_db": [round(x, 3) for x in psnrs], "minimum_psnr_db": round(min(psnrs), 3),
            "mean_psnr_db": round(sum(psnrs) / len(psnrs), 3)}


def _encode_command(policy, source, temp, rate, maxrate):
    encoder = str(policy["video_encoder"])
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "none",
           "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0",
           "-c:v", encoder]
    if encoder == "h264_nvenc":
        cmd += ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-gpu", "0"]
    else:
        cmd += ["-preset", "medium"]
    cmd += ["-b:v", str(rate), "-maxrate", str(maxrate), "-bufsize", str(maxrate * 2),
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
            "-c:a", "aac", "-b:a", str(policy["audio_bitrate"]),
            "-movflags", "+faststart", str(temp)]
    return cmd


def compress_for_inline(source, source_sha, output_dir, policy=None, fps=None):
    """Return (path, metadata). Preserve source; keep every video frame and resolution."""
    del fps  # The derivative keeps the source frame rate; the API-level fps is set per request.
    policy = {**DEFAULTS, **(policy or {})}
    source = Path(source)
    source_bytes = source.stat().st_size
    if not policy.get("enabled", True) or source_bytes <= int(policy["threshold_bytes"]):
        return source, {"applied": False, "source_sha256": source_sha, "source_bytes": source_bytes}

    out_dir = Path(output_dir) / "compressed_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source_sha}.mp4"
    metadata_path = out_dir / f"{source_sha}.json"
    config = {"version": VERSION, "source_sha256": source_sha,
              "threshold_bytes": int(policy["threshold_bytes"]),
              "target_bytes": int(policy["target_bytes"]),
              "quality_floor_psnr_db": float(policy["quality_floor_psnr_db"]),
              "allow_below_quality_floor": bool(policy.get("allow_below_quality_floor", False)),
              "audio_bitrate": str(policy["audio_bitrate"]),
              "video_encoder": str(policy["video_encoder"]),
              "video_policy": "full_frame_rate_full_resolution"}
    if target.is_file() and metadata_path.is_file():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            old_config = dict(cached.get("config") or {})
            current_config = dict(config)
            old_config.setdefault("allow_below_quality_floor", False)
            current_config.setdefault("allow_below_quality_floor", False)
            if old_config == current_config and cached.get("output_sha256") == sha256(target):
                return target, cached
        except (OSError, json.JSONDecodeError):
            pass

    original = _ffprobe_json(source)
    if original["duration_s"] <= 0:
        raise RuntimeError("Invalid source duration")
    temp = target.with_suffix(".tmp.mp4")
    errors = []
    output_probe = None
    video_mode = "stream_copy"
    # Lossless video path: all compressed video packets pass through unchanged.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-hwaccel", "none",
           "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", str(policy["audio_bitrate"]),
           "-movflags", "+faststart", str(temp)]
    p = _run(cmd)
    if p.returncode:
        errors.append("remux: " + p.stderr[-800:])
        temp.unlink(missing_ok=True)
    elif temp.is_file():
        output_probe = _ffprobe_json(temp)

    quality_probe = None
    # Only re-encode pixels when remux is still too large. No fps/scale filter.
    if not temp.is_file() or not output_probe or output_probe["bytes"] > int(policy["target_bytes"]):
        temp.unlink(missing_ok=True)
        video_mode = "bitrate_limited_full_fps_full_resolution"
        audio_bps = int(str(policy["audio_bitrate"]).lower().replace("k", "000"))
        # Leave 1 MB mux/base64 headroom under the request-size guardrail.
        target_bps = max(300_000, int((int(policy["target_bytes"]) - 1_000_000) * 8 / original["duration_s"] - audio_bps))
        for attempt in range(4):
            rate = max(250_000, int(target_bps * (0.92 ** attempt)))
            maxrate = int(rate * 1.18)
            p = _run(_encode_command(policy, source, temp, rate, maxrate))
            if p.returncode:
                errors.append("reencode: " + p.stderr[-1000:])
                temp.unlink(missing_ok=True)
                raise RuntimeError("Could not prepare upload copy: " + " | ".join(errors))
            output_probe = _ffprobe_json(temp)
            if output_probe["bytes"] <= int(policy["target_bytes"]):
                break
            temp.unlink(missing_ok=True)
        if not temp.is_file() or output_probe["bytes"] > int(policy["target_bytes"]):
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"Full-frame-rate encode remains {output_probe['bytes']} bytes, above target {policy['target_bytes']}; original left untouched")
        with tempfile.TemporaryDirectory(prefix="video_quality_probe_") as tmpdir:
            quality_probe = _quality_probe(source, temp, original["duration_s"],
                                           original["width"], original["height"])
        # Keep the sampled PSNR as diagnostic metadata, but do not block scoring
        # on a fixed PSNR threshold. The source video remains untouched.
        quality_guardrail_passed = quality_probe["minimum_psnr_db"] >= float(policy["quality_floor_psnr_db"])
    else:
        quality_guardrail_passed = None

    if output_probe["bytes"] > int(policy["target_bytes"]):
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"High-quality full-frame-rate copy is {output_probe['bytes']} bytes, above safe target {policy['target_bytes']}; source left untouched")
    duration_delta = abs(output_probe["duration_s"] - original["duration_s"])
    if duration_delta > 0.55:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Derivative duration changed by {duration_delta:.3f}s")
    if (output_probe["width"], output_probe["height"]) != (original["width"], original["height"]):
        temp.unlink(missing_ok=True)
        raise RuntimeError("Derivative resolution changed; refusing upload copy")
    if output_probe["r_fps"] != original["r_fps"]:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Derivative nominal frame rate changed ({original['r_fps']} -> {output_probe['r_fps']}); refusing upload copy")
    frame_count_delta = None
    if original.get("nb_frames") and output_probe.get("nb_frames"):
        frame_count_delta = int(output_probe["nb_frames"]) - int(original["nb_frames"])
        if abs(frame_count_delta) > 1:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"Derivative frame count changed by {frame_count_delta} ({original['nb_frames']} -> {output_probe['nb_frames']}); refusing upload copy")
    elif original["fps"] != output_probe["fps"]:
        try:
            old_fps = float(Fraction(original["fps"]))
            new_fps = float(Fraction(output_probe["fps"]))
        except Exception:
            old_fps = new_fps = 0
        if not old_fps or abs(new_fps - old_fps) / old_fps > 0.01:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"Derivative average frame rate changed ({original['fps']} -> {output_probe['fps']}); refusing upload copy")
    if len(original["audio_streams"]) != len(output_probe["audio_streams"]):
        temp.unlink(missing_ok=True)
        raise RuntimeError("Derivative did not keep the source audio stream count")

    os.replace(temp, target)
    metadata = {"applied": True, "config": config, "source_path": str(source),
                "source_sha256": source_sha, "source": original, "output_path": str(target),
                "output_sha256": sha256(target), "output": _ffprobe_json(target),
                "video_mode": video_mode,
                "audio_mode": "AAC " + str(policy["audio_bitrate"]),
                "frame_count_delta": frame_count_delta,
                "sampled_quality_check": quality_probe,
                "quality_guardrail_passed": quality_guardrail_passed,
                "notes": "Upload-only derivative. Original file is unchanged. Full duration, source resolution, and source frame-rate metadata kept; no temporal subsampling or spatial scaling."}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target, metadata
