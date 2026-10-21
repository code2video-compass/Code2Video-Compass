#!/usr/bin/env python3
"""Global TVAlign (G-TVA): Qwen3-VL-Embedding-8B full-video semantic cosine.

Setup:
    git clone https://github.com/QwenLM/Qwen3-VL-Embedding third_party/Qwen3-VL-Embedding
    # model weights download automatically from Qwen/Qwen3-VL-Embedding-8B on
    # first run (or pass --qwen-model with an existing snapshot directory)

The Global Description and Event texts of the sample are joined into one
semantic text and encoded independently from the video (at most 64 uniformly
sampled frames). The score is the cosine between the two normalized embeddings.
Sequences that would hit the context limit are refused rather than truncated,
and preprocessing failures never produce a fabricated score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TASKS = ("t2v", "i2v", "v2v")
MAX_FRAMES = 64
MAX_LENGTH = 32768


def uniform_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count < 1 or maximum < 1:
        raise ValueError("frame_count and maximum must be positive")
    if frame_count <= maximum:
        return list(range(frame_count))
    if maximum == 1:
        return [frame_count // 2]
    return sorted({round(i * (frame_count - 1) / (maximum - 1)) for i in range(maximum)})


def discover(test_root: Path):
    rows = []
    for model_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        for task in TASKS:
            task_root = model_dir / "results" / task
            if not task_root.is_dir():
                continue
            for sample_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
                out_dir = sample_dir / "LLM" / "project" / "out"
                if not out_dir.is_dir():
                    continue
                videos = [p for p in sorted(out_dir.iterdir())
                          if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                          and not p.name.startswith("0web_")]
                for video in videos:
                    rows.append({"task": task, "model": model_dir.name, "sample_id": sample_dir.name,
                                 "generated_video_relative": video.relative_to(test_root).as_posix(),
                                 "generated_video": str(video)})
    return rows


def load_sample_text(sample_id: str, input_root: Path) -> dict:
    annotation_path = input_root / sample_id / "annotation.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"missing annotation.json: {annotation_path}")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation = data.get("source_annotation") or data.get("annotation") or data
    summary = str(annotation.get("summary") or annotation.get("global_description") or "").strip()
    events = annotation.get("events") or []
    texts = []
    for idx, event in enumerate(events, 1):
        if isinstance(event, str):
            desc = event.strip()
        elif isinstance(event, dict):
            desc = str(event.get("description") or event.get("text") or event.get("summary") or "").strip()
        else:
            continue
        if desc:
            texts.append(desc)
    if not summary:
        raise ValueError(f"{sample_id}: annotation has no Global Description/summary")
    if not texts:
        raise ValueError(f"{sample_id}: annotation has no Event descriptions")
    return {"summary": summary, "events": texts, "annotation_path": str(annotation_path)}


def read_video_frames(path: Path, max_frames: int):
    import decord
    import numpy as np
    from PIL import Image
    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(path), num_threads=1)
    n = len(reader)
    source_fps = float(reader.get_avg_fps())
    if n < 1 or source_fps <= 0:
        raise ValueError(f"invalid video stream length={n}, fps={source_fps}")
    indices = uniform_indices(n, int(max_frames))
    frames_arr = reader.get_batch(indices).asnumpy()
    frames = [Image.fromarray(np.asarray(a).astype("uint8")).convert("RGB") for a in frames_arr]
    timestamps = [float(i / source_fps) for i in indices]
    return frames, timestamps


def load_embedder(qwen_repo: Path, model_path: Path):
    sys.path.insert(0, str(qwen_repo))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402
    if not model_path.is_dir():
        from huggingface_hub import snapshot_download
        model_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download("Qwen/Qwen3-VL-Embedding-8B", local_dir=str(model_path))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    embedder = Qwen3VLEmbedder(model_name_or_path=str(model_path), max_length=MAX_LENGTH,
                               max_frames=MAX_FRAMES, torch_dtype=dtype,
                               low_cpu_mem_usage=True)
    revision = getattr(getattr(embedder.model, "config", None), "_commit_hash", None) or str(model_path)
    return embedder, dtype, revision


def qwen_encode(embedder, *, text=None, frames=None):
    import torch.nn.functional as F
    spec = {"text": text} if text is not None else {"video": frames}
    conversation = embedder.format_model_input(**spec)
    inputs = embedder._preprocess_inputs([conversation])
    token_count = int(inputs["input_ids"].shape[1])
    # Refuse sequences that hit max_length so a truncated prompt or frame
    # sequence cannot get a fake score.
    if token_count >= MAX_LENGTH:
        raise ValueError(f"input reached max_length={MAX_LENGTH}; refusing possibly truncated embedding")
    if frames is not None and "pixel_values_videos" not in inputs:
        raise ValueError("processor produced no video tensor; refusing empty-video score")
    inputs = {k: v.to(embedder.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = embedder.forward(inputs)
        emb = embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
        emb = F.normalize(emb.float(), p=2, dim=-1)
    if not torch.isfinite(emb).all():
        raise ValueError("embedding contains non-finite values")
    return emb[0].cpu(), token_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--input-root", type=Path, default=HERE.parent.parent / "inputs")
    parser.add_argument("--qwen-repo", type=Path,
                        default=HERE / "third_party" / "Qwen3-VL-Embedding")
    parser.add_argument("--qwen-model", type=Path,
                        default=HERE / "checkpoints" / "Qwen3-VL-Embedding-8B")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    if not args.qwen_repo.is_dir():
        raise FileNotFoundError(f"Qwen3-VL-Embedding repository not found: {args.qwen_repo}")
    embedder, dtype, revision = load_embedder(args.qwen_repo, args.qwen_model)
    rows = [r for r in discover(args.test_root) if r["task"] in set(args.tasks)]
    previous = []
    if args.only_unscored and args.output.is_file():
        previous = json.loads(args.output.read_text()).get("results", [])
        done = {r["generated_video_relative"] for r in previous}
        rows = [r for r in rows if r["generated_video_relative"] not in done]
    rows = rows[: args.limit or None]
    print(f"device={embedder.model.device} videos={len(rows)}", flush=True)
    results, errors = [], []
    for i, row in enumerate(rows, 1):
        try:
            info = load_sample_text(row["sample_id"], args.input_root)
            semantic_text = " ".join([info["summary"], *info["events"]])
            frames, timestamps = read_video_frames(Path(row["generated_video"]), MAX_FRAMES)
            text_embedding, text_tokens = qwen_encode(embedder, text=semantic_text)
            video_embedding, video_tokens = qwen_encode(embedder, frames=frames)
            score = float(torch.dot(text_embedding, video_embedding).item())
            results.append({**row, "status": "ok",
                            "qwen_vl_embedding_global": score,
                            "global_sampled_timestamps_s": timestamps,
                            "frame_count": len(frames),
                            "text_token_count": text_tokens,
                            "video_input_token_count": video_tokens,
                            "semantic_source_type": "global_description_and_events",
                            "global_description": info["summary"],
                            "events": info["events"]})
            print(f"{i}/{len(rows)} {row['model']} {row['sample_id']} "
                  f"G-TVA={score:.4f}", flush=True)
        except Exception as exc:
            errors.append({**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i}/{len(rows)} ERROR {row['generated_video_relative']}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "metric": "Global TVAlign",
        "metric_protocol": "qwen3_vl_embedding_global_v2",
        "model": {"name": "Qwen3-VL-Embedding-8B", "repository": "Qwen/Qwen3-VL-Embedding-8B",
                  "revision": revision, "precision": str(dtype),
                  "implementation": "official Qwen3VLEmbedder; independent text and video encodes"},
        "sampling": {"max_frames": MAX_FRAMES, "max_length": MAX_LENGTH},
        "device": str(embedder.model.device),
        "n_results": len(merged), "n_errors": len(errors), "errors": errors,
        "results": sorted(merged.values(), key=lambda r: r["generated_video_relative"]),
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(f"finished={len(results)}/{len(rows)} errors={len(errors)} output={args.output}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
