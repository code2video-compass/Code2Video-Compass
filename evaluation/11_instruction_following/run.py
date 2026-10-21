#!/usr/bin/env python3
"""Instruction Following (IF): binary QA checklist judged by Gemini.

An MLLM-as-a-judge protocol. Each generated video is submitted with its
generation prompt and the sample's QA checklist (`inputs/<sample_id>/qa.json`,
up to 14 shared questions plus the setting-specific section) to a native
generateContent request (inlineData MP4, temperature 0, 2 FPS sampling). The
judge returns one binary Yes/No answer per checklist question, and the video's
IF score is the fraction of Yes answers.

This call computes IF only; holistic quality (NC/VD/MQ/VE) is a separate
metric in `12_holistic_video_quality` with its own prompt and API call.

Finished jobs are cached by a content-addressed key (video bytes + checklist
+ prompts + request settings), so re-running never re-scores, and every raw
API response is kept on disk.

Requires `GEMINI_API_KEY` in the environment (or a `.env` file beside this
script; get a key at https://aistudio.google.com/apikey); `GEMINI_API_BASE`
optionally overrides the endpoint root. `ffmpeg`/`ffprobe` must be on PATH
for probing and (only above 40 MB) upload derivatives.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gemini_adapter import GeminiNativeVideoAdapter, load_env, redact  # noqa: E402
from video_prep import compress_for_inline, probe_video, sha256  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
SPECIFIC = {"t2v": "FullVideoSpecific", "i2v": "KeyframeSpecific", "v2v": "ReferenceClipSpecific"}

EVAL_PROMPT_PATH = HERE / "instruction_following_prompt.txt"
JUDGE_PROMPT = EVAL_PROMPT_PATH.read_text(encoding="utf-8")


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def discover(test_root: Path, tasks: set[str], only_ids: set[str] | None = None,
             models: set[str] | None = None) -> list[dict]:
    rows = []
    if not test_root.is_dir():
        raise FileNotFoundError(f"Generated-video directory not found: {test_root}")
    model_dirs = (sorted(test_root / name for name in models) if models is not None
                  else sorted(p for p in test_root.iterdir() if p.is_dir()))
    for model_dir in model_dirs:
        for task in sorted(tasks):
            task_dir = model_dir / "results" / task
            if not task_dir.is_dir():
                continue
            sample_dirs = (sorted(task_dir / sample_id for sample_id in only_ids) if only_ids is not None
                           else sorted(p for p in task_dir.iterdir() if p.is_dir()))
            for sample_dir in sample_dirs:
                if only_ids is not None and sample_dir.name not in only_ids:
                    continue
                out_dir = sample_dir / "LLM" / "project" / "out"
                if not out_dir.is_dir():
                    continue
                videos = sorted(p for p in out_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
                primary = [p for p in videos if not p.name.startswith("0web_")]
                if primary:
                    videos = primary
                for video in videos:
                    rows.append({"sample_id": sample_dir.name, "task": task,
                                 "generated_model": model_dir.name,
                                 "video_path": video.resolve(),
                                 "video_relative": video.relative_to(test_root).as_posix()})
    return rows


def make_items(sample_id: str, task: str, input_root: Path) -> tuple[list[dict], dict, Path, dict]:
    qa_path = input_root / sample_id / "qa.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Checklist missing: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    items = []

    def walk(node: Any, path: str) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            if isinstance(node.get("question"), str) and node["question"].strip():
                if node.get("expected_answer") != "Yes":
                    raise ValueError(f"Checklist expected_answer must be Yes at {path}")
                item = {"checklist_path": path, "question": node["question"].strip(),
                        "expected_answer": node["expected_answer"]}
                for field in ("id", "dimension", "name"):
                    if field in node:
                        item[field] = node[field]
                items.append(item)
            else:
                for key, child in node.items():
                    walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{path}[{i}]")

    sections = ("SharedChecklist", SPECIFIC[task])
    for section in sections:
        if section in qa:
            walk(qa[section], section)
    if not items:
        raise ValueError(f"No checklist questions apply to {task}: {qa_path}")
    checklist_payload = [{"number": idx, "question": item["question"]} for idx, item in enumerate(items, 1)]
    # Category metadata is optional in the release layout and never affects scoring.
    meta_path = input_root / sample_id / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    fields = meta.get("fields") or {}
    category = {"category_ids": meta.get("final_category_ids", []), "subtype_ids": meta.get("subtype_ids", []),
                "category": fields.get("category", ""), "subtype": fields.get("subtype", ""),
                "complexity": {k: fields[k] for k in ("construction_complexity", "dynamic_complexity", "dependency_span") if fields.get(k)}}
    return items, category, qa_path, checklist_payload


def resolve_generation_prompt_path(sample_id: str, task: str, input_root: Path) -> Path:
    for path in (input_root / sample_id / "prompt" / f"{task}.txt",
                 input_root / sample_id / f"{task}.txt"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Generation prompt missing for {sample_id} ({task}); checked {input_root / sample_id / 'prompt' / (task + '.txt')}")


def load_sample_list_keys(paths: list[Path], tasks: set[str]) -> list[tuple[str, str]]:
    """Read one or more CSV scopes as exact (sample_id, task) pairs.

    A CSV with a task column selects only that task for each row. A CSV without
    task selects every requested task for each ID.
    """
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(key: tuple[str, str]) -> None:
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Sample-list CSV not found: {path}")
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = {str(k).strip().lower() for k in (reader.fieldnames or [])}
            if "sample_id" not in fields:
                raise ValueError(f"Sample-list CSV needs a sample_id column: {path}")
            has_task = "task" in fields or "task_type" in fields
            for line_no, raw in enumerate(reader, start=2):
                row = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                sample_id = row.get("sample_id", "")
                if not sample_id:
                    continue
                task_value = row.get("task", "") or row.get("task_type", "")
                if has_task and task_value:
                    task = task_value.lower()
                    task = {"t2v": "t2v", "i2v": "i2v", "v2v": "v2v",
                            "text-to-video": "t2v", "image-to-video": "i2v",
                            "video-to-video": "v2v"}.get(task, task)
                    if task not in SPECIFIC:
                        raise ValueError(f"Unknown task {task_value!r} at {path}:{line_no}")
                    if task in tasks:
                        add((sample_id, task))
                else:
                    for task in sorted(tasks):
                        add((sample_id, task))
    return keys


def load_sample_model_scope(paths: list[Path], tasks: set[str]) -> dict[tuple[str, str], set[str] | None]:
    """An optional model column restricts each CSV row to that model only.

    Existing sample/task-only lists keep their all-selected-models behavior.
    Empty cells in an explicit model column are rejected to avoid accidental
    paid calls to other models.
    """
    scope = {}
    aliases = {"text-to-video": "t2v", "image-to-video": "i2v", "video-to-video": "v2v"}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = {str(k).strip().lower() for k in (reader.fieldnames or [])}
            model_field = next((k for k in ("generated_model", "model", "tested_model") if k in fields), None)
            for line_no, raw in enumerate(reader, 2):
                row = {str(k).strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                sid = row.get("sample_id")
                if not sid:
                    continue
                model = row.get(model_field, "") if model_field else None
                if model_field and not model:
                    raise ValueError(f"Empty {model_field} at {path}:{line_no}")
                task = (row.get("task") or row.get("task_type") or "").lower()
                task = aliases.get(task, task)
                for t in ([task] if task else sorted(tasks)):
                    if t not in tasks:
                        continue
                    key = (sid, t)
                    if model is None:
                        scope[key] = None
                    elif key not in scope:
                        scope[key] = {model}
                    elif scope[key] is not None:
                        scope[key].add(model)
    return scope


def parse_response(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Response has no JSON object")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Response JSON root must be an object")
    return result


def validate(result: dict, job: dict, duration: float) -> None:
    expected = job["items"]
    rows = result.get("qa")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} QA answers")
    yes_count = 0
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict) or type(row.get("question")) is not int or row["question"] != idx:
            raise ValueError(f"QA question number changed or reordered at item {idx}")
        if row.get("answer") not in {"Yes", "No"}:
            raise ValueError(f"Invalid binary answer at item {idx}")
        # Rationale is not part of checklist scoring. If the model adds one
        # despite the prompt, discard it without rejecting an otherwise valid
        # assessment (including unusually long rationales).
        if set(row) - {"question", "answer", "reason"} or not {"question", "answer"} <= set(row):
            raise ValueError(f"QA item {idx} must contain question and answer only")
        row.pop("reason", None)
        yes_count += row["answer"] == "Yes"
    if set(result) != {"qa"}:
        raise ValueError("Response must contain exactly qa")
    result["checklist_summary"] = {"yes_count": yes_count, "total_items": len(expected),
                                   "score_0_1": round(yes_count / len(expected), 6) if expected else None}


def job_signature(job: dict, cfg: dict, input_root: Path) -> tuple[str, str]:
    video_hash = sha256(job["video_path"])
    items, category, qa_path, checklist_payload = make_items(job["sample_id"], job["task"], input_root)
    generation_prompt_path = resolve_generation_prompt_path(job["sample_id"], job["task"], input_root)
    generation_prompt = generation_prompt_path.read_text(encoding="utf-8").strip()
    job.update(items=items, category=category, checklist_path=str(qa_path.resolve()), checklist_sha256=sha256(qa_path))
    job.update(checklist_payload=checklist_payload, generation_prompt=generation_prompt,
               generation_prompt_path=str(generation_prompt_path.resolve()),
               generation_prompt_sha256=sha256(generation_prompt_path))
    key = digest({"video_sha256": video_hash, "checklist_sha256": job["checklist_sha256"], "items": items,
                  "generation_prompt_sha256": job["generation_prompt_sha256"],
                  "task": job["task"], "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
                  "model": cfg["model"]["name"], "api_protocol": cfg["api_protocol"],
                  "request": cfg["request"], "compression": cfg["compression"],
                  "run_tag": cfg.get("run_tag", "")})
    job["job_key"] = key; job["video_sha256"] = video_hash
    return key, video_hash


def saved_assessment(job: dict, out_dir: Path) -> dict | None:
    """Reuse the same cache lookup during planning and immediately before a call."""
    path = out_dir / "records" / f"{job['job_key']}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("status") == "scored":
        return record
    return None


def score_one(job: dict, cfg: dict, run_id: str, out_dir: Path) -> dict:
    started = time.monotonic(); adapter = None
    result = {k: job[k] for k in ("job_key", "sample_id", "task", "generated_model", "video_path", "video_relative",
                                  "video_sha256", "checklist_path", "checklist_sha256", "category")}
    result["items"] = job["items"]
    result["generation_prompt_path"] = job["generation_prompt_path"]
    result["generation_prompt_sha256"] = job["generation_prompt_sha256"]
    result["judge_prompt_sha256"] = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    result.update(run_id=run_id, model=cfg["model"]["name"], status="processing_error")
    try:
        cached = saved_assessment(job, out_dir)
        if cached:
            cached.update(status="cached", run_id=run_id, api_requests=0, latency_s=0)
            return cached
        info = probe_video(job["video_path"]); duration = float(info["duration"])
        upload_path, compression = compress_for_inline(job["video_path"], job["video_sha256"], out_dir,
                                                       policy=cfg["compression"], fps=cfg["request"]["fps"])
        upload_hash = sha256(upload_path)
        prompt = JUDGE_PROMPT.replace("{generation_prompt}", job["generation_prompt"])
        prompt = prompt.replace("{qa_checklist}", json.dumps(job["checklist_payload"], ensure_ascii=False, indent=2))
        prompt = prompt.replace("{generated_video}", "The submitted video is attached to this request.")
        adapter = GeminiNativeVideoAdapter(cfg, out_dir)
        response = adapter.score(upload_path, upload_hash, prompt,
            {"sample_id": job["sample_id"], "task": job["task"], "duration_s": duration,
             "sampling_fps": cfg["request"]["fps"], "coverage_mode": "full_video",
             "evaluation": "instruction_following"}, {})
        result.update(raw_response=response["raw_response"], usage=response["usage"], actual_model=response["actual_model"],
                      finish_status=response["finish_status"], request_settings=response["request_settings"],
                      api_requests=adapter.http_requests, api_attempts=adapter.attempts,
                      upload_video_path=str(upload_path), upload_video_sha256=upload_hash,
                      upload_compression=compression, video_probe=info)
        assessment = parse_response(response["text"])
        validate(assessment, job, duration)
        result.update(status="scored", assessment=assessment)
    except Exception as exc:
        result["error"] = redact(f"{type(exc).__name__}: {exc}")
        if adapter:
            result["api_requests"] = adapter.http_requests
            result["api_attempts"] = adapter.attempts
    finally:
        if adapter: adapter.close()
        result["latency_s"] = round(time.monotonic() - started, 3)
    write_json(out_dir / "records" / f"{job['job_key']}.json", result)
    if result.get("raw_response"):
        write_json(out_dir / "raw" / f"{job['job_key']}.json", result["raw_response"])
    return result


def export_scores(out_dir: Path) -> None:
    """Rebuild the human-readable CSVs from the already-saved per-job records."""
    all_records = []
    for path in sorted((out_dir / "records").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("job_key"):
            all_records.append(record)
    latest = {}
    for rec in all_records:
        identity = (rec.get("sample_id"), rec.get("generated_model"), rec.get("task"))
        if identity not in latest or rec.get("run_id", "") > latest[identity].get("run_id", ""):
            latest[identity] = rec
    records = sorted(latest.values(), key=lambda r: (r.get("generated_model", ""), r.get("task", ""), r.get("sample_id", "")))
    fields = ["job_key", "sample_id", "tested_model", "judge_model", "generated_model", "task", "video_relative", "status",
              "yes_count", "total_items", "instruction_following_0_1", "record_path"]
    rows = []
    for rec in records:
        summary = (rec.get("assessment") or {}).get("checklist_summary") or {}
        rows.append({"job_key": rec.get("job_key", ""), "sample_id": rec.get("sample_id", ""),
                     "tested_model": rec.get("generated_model", ""),
                     "judge_model": rec.get("actual_model") or rec.get("model", ""),
                     "generated_model": rec.get("generated_model", ""), "task": rec.get("task", ""),
                     "video_relative": rec.get("video_relative", ""), "status": rec.get("status", ""),
                     "yes_count": summary.get("yes_count", ""), "total_items": summary.get("total_items", ""),
                     "instruction_following_0_1": summary.get("score_0_1", ""),
                     "record_path": str(Path("records") / f"{rec.get('job_key','')}.json")})
    out_dir.mkdir(parents=True, exist_ok=True)
    temp = out_dir / "scores.csv.tmp"
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, out_dir / "scores.csv")
    tmp_jsonl = out_dir / "scores.jsonl.tmp"
    with tmp_jsonl.open("w", encoding="utf-8") as f:
        for rec in all_records: f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp_jsonl, out_dir / "scores.jsonl")
    item_fields = ["job_key", "sample_id", "generated_model", "task", "video_relative", "status",
                   "item_order", "checklist_path", "name", "question", "answer"]
    item_rows = []
    for rec in records:
        for idx, item in enumerate((rec.get("assessment") or {}).get("qa", []), 1):
            src = (rec.get("items") or [])
            source_item = src[idx - 1] if idx <= len(src) else {}
            item_rows.append({"job_key": rec.get("job_key", ""), "sample_id": rec.get("sample_id", ""),
                "generated_model": rec.get("generated_model", ""), "task": rec.get("task", ""),
                "video_relative": rec.get("video_relative", ""), "status": rec.get("status", ""),
                "item_order": idx, "checklist_path": source_item.get("checklist_path", ""),
                "name": source_item.get("name", ""), "question": source_item.get("question", ""),
                "answer": item.get("answer", "")})
    item_tmp = out_dir / "checklist_answers.csv.tmp"
    with item_tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=item_fields); writer.writeheader(); writer.writerows(item_rows)
    os.replace(item_tmp, out_dir / "checklist_answers.csv")


def main() -> None:
    # If Ctrl+C interrupts this batch, its finished per-job JSON records are
    # still consolidated into scores.csv/checklist_answers.csv/scores.jsonl.
    parser = argparse.ArgumentParser(
        description="Score generated videos against QA checklists with Gemini (Instruction Following, IF)")
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--input-root", type=Path, default=HERE.parent.parent / "inputs")
    parser.add_argument("--out-dir", type=Path, default=HERE / "checklist_scores")
    parser.add_argument("--tasks", nargs="+", choices=sorted(SPECIFIC), default=["t2v", "i2v", "v2v"])
    parser.add_argument("--samples", nargs="+", help="Optional sample IDs to restrict this run")
    parser.add_argument("--sample-list", action="append", type=Path,
                        help="Optional custom CSV scope; repeat to combine. A task column selects exact "
                             "sample/task pairs; an optional generated_model column restricts each row "
                             "to that model. Without task, each sample selects all requested tasks.")
    parser.add_argument("--limit-samples", "--limit-sample", dest="limit_samples", type=int,
                        help="Select the first N unique sample IDs from the chosen scope; includes all "
                             "listed tasks for each selected sample. --limit separately caps new video jobs.")
    parser.add_argument("--models", nargs="+", help="Generated-model folder names to restrict this run")
    parser.add_argument("--limit", type=int, help="Maximum new video jobs submitted in this run")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum parallel Gemini requests")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Requested video sampling FPS in videoMetadata (default: 2)")
    parser.add_argument("--model", default="gemini-3.8-flash",
                        help="Judge model for the generateContent endpoint (default: %(default)s)")
    parser.add_argument("--media-resolution", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--video-encoder", default="libx264",
                        help="FFmpeg encoder for >40 MB upload derivatives (paper runs: h264_nvenc)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect/plan without making API calls")
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    atexit.register(export_scores, out_dir)
    if (args.concurrency < 1 or (args.limit is not None and args.limit < 1)
            or (args.limit_samples is not None and args.limit_samples < 1)
            or not 0 < args.fps <= 24):
        parser.error("concurrency, limit, and limit-samples must be positive; fps must be in (0,24]")
    load_env()
    if not JUDGE_PROMPT.strip():
        parser.error(f"Judge prompt is empty: {EVAL_PROMPT_PATH}")
    cfg = {"provider": {"api_key_env": "GEMINI_API_KEY", "base_url_env": "GEMINI_API_BASE"},
           "model": {"name": args.model},
           "api_protocol": "gemini_generate_content_inline_data",
           "request": {"fps": args.fps, "media_resolution": args.media_resolution,
                       "max_inline_video_bytes": 40_000_000, "max_completion_tokens": 10000,
                       "max_retries": 2, "fps_control_verified": False,
                       "media_resolution_control_verified": False},
           "compression": {"enabled": True, "threshold_bytes": 40_000_000, "target_bytes": 35_000_000,
                           "quality_floor_psnr_db": 25.0, "allow_below_quality_floor": True,
                           "audio_bitrate": "96k", "video_encoder": args.video_encoder}}
    selected_tasks = set(args.tasks)
    sample_list_paths = [p.resolve() for p in args.sample_list] if args.sample_list else None
    try:
        if sample_list_paths:
            ordered_sample_keys = load_sample_list_keys(sample_list_paths, selected_tasks)
            sample_model_scope = load_sample_model_scope(sample_list_paths, selected_tasks)
        else:
            ordered_sample_keys, sample_model_scope = [], {}
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if not sample_list_paths:
        ordered_sample_keys = None
    if args.samples:
        requested_ids = set(args.samples)
        if ordered_sample_keys is None:
            ordered_sample_keys = [(sid, task) for sid in dict.fromkeys(args.samples)
                                   for task in sorted(selected_tasks)]
        else:
            ordered_sample_keys = [key for key in ordered_sample_keys if key[0] in requested_ids]
    ordered_ids = (list(dict.fromkeys(sample_id for sample_id, _ in ordered_sample_keys))
                   if ordered_sample_keys is not None else None)
    if ordered_ids is not None and args.limit_samples is not None:
        ordered_ids = ordered_ids[:args.limit_samples]
    selected_id_set = set(ordered_ids) if ordered_ids is not None else None
    sample_keys = ({key for key in ordered_sample_keys if key[0] in selected_id_set}
                   if ordered_sample_keys is not None else None)
    discovery_models = set(args.models) if args.models else None
    print(f"[prepare] Scanning videos: samples={'all' if selected_id_set is None else len(selected_id_set)}, "
          f"models={','.join(sorted(discovery_models)) if discovery_models is not None else 'all'}, "
          f"tasks={','.join(sorted(selected_tasks))}. No API requests yet.", flush=True)
    rows = discover(args.test_root, selected_tasks, only_ids=selected_id_set, models=discovery_models)
    if ordered_sample_keys is None and args.limit_samples is not None:
        # No CSV or explicit IDs: select from the discovered model/task scope.
        selected_id_set = set(dict.fromkeys(row["sample_id"] for row in rows))
        selected_id_set = set(sorted(selected_id_set)[:args.limit_samples])
        rows = [row for row in rows if row["sample_id"] in selected_id_set]
    if ordered_sample_keys is not None:
        rows = [row for row in rows if (row["sample_id"], row["task"]) in sample_keys]
    if args.models:
        selected_models = set(args.models)
        rows = [row for row in rows if row["generated_model"] in selected_models]
    rows = [row for row in rows if sample_model_scope.get((row["sample_id"], row["task"])) is None
            or row["generated_model"] in sample_model_scope[(row["sample_id"], row["task"])]]
    print(f"[prepare] Found {len(rows)} videos; validating checklists and hashing media for cache lookup.", flush=True)
    jobs = []
    issues = []
    for preparation_index, row in enumerate(rows, 1):
        if preparation_index == 1 or preparation_index % 25 == 0:
            print(f"[prepare] {preparation_index}/{len(rows)} "
                  f"{row['generated_model']}/{row['task']}/{row['sample_id']}", flush=True)
        try:
            key, video_hash = job_signature(row, cfg, args.input_root)
            row.update(job_key=key, video_sha256=video_hash)
            jobs.append(row)
        except Exception as exc:
            issues.append({"sample_id": row["sample_id"], "task": row["task"], "video": row["video_relative"],
                           "error": f"{type(exc).__name__}: {exc}"})
    print(f"[prepare] Checked {len(rows)} videos: ready={len(jobs)}, input_errors={len(issues)}. "
          "Checking saved results.", flush=True)
    pending = []
    for job in jobs:
        if saved_assessment(job, out_dir) is None:
            pending.append(job)
    selected = pending[:args.limit] if args.limit is not None else pending
    print(f"[plan] cached={len(jobs)-len(pending)}, pending={len(pending)}, "
          f"selected={len(selected)}, concurrency={args.concurrency}, dry_run={args.dry_run}", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = out_dir / "runs" / run_id
    summary = {"run_id": run_id, "mode": "dry_run" if args.dry_run else "execute",
               "metric": "Instruction Following", "requested_model": cfg["model"]["name"], "tasks": args.tasks,
               "discovered_videos": len(rows), "checklist_ready_jobs": len(jobs),
               "missing_or_invalid_checklist": len(issues), "compatible_completed": len(jobs)-len(pending),
               "pending": len(pending), "selected_this_run": len(selected),
               "api_calls_planned": 0 if args.dry_run else len(selected),
               "concurrency": args.concurrency, "limit": args.limit, "requested_fps": args.fps,
               "limit_samples": args.limit_samples,
               "scope": "custom" if sample_list_paths else ("explicit" if args.samples else "all"),
               "sample_lists": [str(p) for p in (sample_list_paths or [])],
               "media_resolution": args.media_resolution, "fps_effective_verified": False,
               "api_protocol": cfg["api_protocol"],
               "media_resolution_effective_verified": False,
               "video_encoder": args.video_encoder,
               "judge_prompt": str(EVAL_PROMPT_PATH),
               "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()}
    write_json(run_dir / "run_summary.json", summary)
    if issues: write_json(run_dir / "input_issues.json", issues)
    write_json(run_dir / "planned_jobs.json", [{"sample_id": j["sample_id"], "task": j["task"],
        "generated_model": j["generated_model"], "video": j["video_relative"],
        "checklist_question_count": len(j["items"]), "category": j["category"]} for j in selected])
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return
    if not selected:
        export_scores(out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return
    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(score_one, job, cfg, run_id, out_dir): job for job in selected}
        for future in as_completed(futures):
            job = futures[future]
            try:
                rec = future.result()
                status = rec.get("status", "processing_error"); statuses[status] = statuses.get(status, 0) + 1
                write_json(run_dir / "records" / f"{job['job_key']}.json", rec)
                if status == "cached":
                    # Another process may have finished after our plan. The
                    # existing result is reused without a request or log spam.
                    continue
                if rec.get("raw_response"):
                    write_json(run_dir / "raw" / f"{job['job_key']}.json", rec["raw_response"])
                # Persist the aggregate after every completed request, rather
                # than waiting for the full batch. Per-job JSON is already
                # durable in score_one(); this keeps the CSVs current even if
                # the batch is interrupted.
                export_scores(out_dir)
                if status == "scored":
                    checklist = rec["assessment"]["checklist_summary"]
                    print(f"{job['sample_id']} {job['task']} tested_model={rec['generated_model']} "
                          f"judge_model={rec.get('actual_model') or rec['model']} "
                          f"IF={checklist['yes_count']}/{checklist['total_items']} "
                          f"({checklist['score_0_1']:.3f}) {status}", flush=True)
                else:
                    print(f"{job['sample_id']} {job['task']} tested_model={rec['generated_model']} "
                          f"judge_model={rec['model']} {status}: {rec.get('error', '')}", flush=True)
            except Exception as exc:
                statuses["processing_error"] = statuses.get("processing_error", 0) + 1
                print(f"{job['sample_id']} {job['task']} {job['generated_model']} error: {redact(exc)}", flush=True)
    summary.update(statuses=statuses, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(run_dir / "run_summary.json", summary)
    export_scores(out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if statuses.get("processing_error"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
