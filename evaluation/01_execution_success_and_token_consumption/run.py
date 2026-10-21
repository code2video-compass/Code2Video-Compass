#!/usr/bin/env python3
"""Execution Success (ES) and Token Consumption (TC) from generation metadata.

ES: fraction of attempted tasks whose code produced a rendered video.
TC: mean provider-reported output tokens per attempted task.

Reads generated/<model>/results/<task>/<sample_id>/LLM/meta.json plus the
rendered-video discovery used by every metric; writes per-model-task rows.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TASKS = ("t2v", "i2v", "v2v")


def discover(test_root: Path) -> dict[tuple[str, str, str], dict]:
    """One record per (model, task, sample_id) with attempt and export state."""
    records: dict[tuple[str, str, str], dict] = {}
    if not test_root.is_dir():
        raise FileNotFoundError(f"test root not found: {test_root}")
    for model_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        for task in TASKS:
            task_root = model_dir / "results" / task
            if not task_root.is_dir():
                continue
            for sample_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
                llm_dir = sample_dir / "LLM"
                if not llm_dir.is_dir():
                    continue
                out_dir = llm_dir / "project" / "out"
                videos = []
                if out_dir.is_dir():
                    videos = sorted(p for p in out_dir.iterdir()
                                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
                # 0web_* files are web transcodes of the same render; they are
                # not additional executions.
                main_videos = [p for p in videos if not p.name.startswith("0web_")]
                meta_path = llm_dir / "meta.json"
                records[(model_dir.name, task, sample_dir.name)] = {
                    "model": model_dir.name, "task": task, "sample_id": sample_dir.name,
                    "meta_path": meta_path if meta_path.is_file() else None,
                    "video_exports": [p.name for p in main_videos],
                    "executed": bool(main_videos),
                }
    return records


def output_tokens(meta: dict) -> int | None:
    """Return output/completion token usage; missing stays None, never zero."""
    usage_candidates = [
        meta.get("usage"), meta.get("usageMetadata"), meta.get("usage_metadata"),
        meta.get("token_usage"), meta.get("response_usage"),
    ]
    for container in (meta.get("response"), meta.get("response_metadata")):
        if isinstance(container, dict):
            usage_candidates.extend((container.get("usage"), container.get("usageMetadata"),
                                     container.get("token_usage")))
    for usage in usage_candidates:
        if not isinstance(usage, dict):
            continue
        # Gemini reports candidate text and internal thoughts separately;
        # count both, matching OpenAI completion_tokens / Anthropic output_tokens.
        candidates = usage.get("candidatesTokenCount")
        thoughts = usage.get("thoughtsTokenCount", 0)
        if (isinstance(candidates, (int, float)) and not isinstance(candidates, bool)
                and candidates >= 0 and isinstance(thoughts, (int, float))
                and not isinstance(thoughts, bool) and thoughts >= 0):
            return int(candidates) + int(thoughts)
        for key in ("completion_tokens", "output_tokens", "candidatesTokenCount",
                    "outputTokens", "completionTokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                return int(value)
        total, prompt = usage.get("total_tokens"), usage.get("prompt_tokens")
        if (isinstance(total, (int, float)) and isinstance(prompt, (int, float))
                and total >= prompt >= 0):
            return int(total - prompt)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    records = discover(args.test_root)
    if not records:
        raise SystemExit(f"no attempted tasks under {args.test_root}")
    per_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records.values():
        meta = None
        if record["meta_path"] is not None:
            try:
                meta = json.loads(record["meta_path"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = None
        tokens = output_tokens(meta) if isinstance(meta, dict) else None
        per_cell[(record["model"], record["task"])].append({**record, "output_tokens": tokens})
    rows = []
    for (model, task), items in sorted(per_cell.items()):
        attempted = len(items)
        executed = sum(1 for item in items if item["executed"])
        token_values = [item["output_tokens"] for item in items
                        if item["output_tokens"] is not None]
        rows.append({
            "model": model, "task": task,
            "attempted_tasks": attempted,
            "executed_tasks": executed,
            "execution_success": round(executed / attempted, 6) if attempted else None,
            "tasks_with_token_record": len(token_values),
            "mean_output_tokens": (round(sum(token_values) / len(token_values), 2)
                                   if token_values else None),
        })
    payload = {
        "metric": "Execution Success; Token Consumption",
        "metric_protocol": "generation_meta_usage_v1",
        "definitions": {
            "execution_success": "executed_tasks / attempted_tasks; an execution is a rendered main video under LLM/project/out (0web_ transcodes are the same execution)",
            "mean_output_tokens": "mean provider-reported output tokens per attempted task; tasks without a usage record are excluded from the mean, not counted as zero",
        },
        "test_root": str(args.test_root),
        "results": rows,
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(f"models×tasks={len(rows)} output={args.output}")
    for row in rows:
        es = "-" if row["execution_success"] is None else f"{row['execution_success']:.3f}"
        tc = "-" if row["mean_output_tokens"] is None else f"{row['mean_output_tokens']:.0f}"
        print(f"  {row['model']:<28} {row['task']:<4} ES={es} TC={tc} "
              f"({row['executed_tasks']}/{row['attempted_tasks']})")


if __name__ == "__main__":
    main()
