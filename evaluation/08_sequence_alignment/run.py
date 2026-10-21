#!/usr/bin/env python3
"""Sequence Alignment (SA) for V2V: DINOv3 subsequence DTW localization.

The source keyclip (inputs/<sample_id>/keyclip.mp4) and the full generated
video are both sampled at 4 FPS and encoded by DINOv3
(timm/vit_base_patch16_dinov3.lvd1689m, downloaded automatically). A monotone
open-end subsequence DTW locates the generated interval that best matches the
whole reference in order. The score is the path-mean DINOv3 cosine. Paths must
span at least 25% of the reference's sampled-frame count (and at least 2
frames), which rules out a single lucky frame while allowing shorter passages;
the realized ratio is recorded per result. This is order/appearance alignment,
not exact synchronization or motion similarity.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
SAMPLE_FPS = 4.0
BATCH_SIZE = 16
MIN_GENERATED_SPAN_RATIO = 0.25
MODEL_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
MODEL_NAME = f"hf-hub:timm/vit_base_patch16_dinov3.lvd1689m@{MODEL_REVISION}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_TRANSFORM = None


def _load_model():
    global _MODEL, _TRANSFORM
    if _MODEL is None:
        _MODEL = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval().to(DEVICE)
        data_config = timm.data.resolve_model_data_config(_MODEL)
        _TRANSFORM = timm.data.create_transform(**data_config, is_training=False)
    return _MODEL, _TRANSFORM


def encode_pil_batch(images: list[Image.Image]) -> np.ndarray:
    model, transform = _load_model()
    inputs = torch.stack([transform(image) for image in images]).to(DEVICE)
    with torch.inference_mode():
        features = model(inputs).float()
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    embeddings = features.cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(embeddings).all():
        raise ValueError("DINOv3 produced NaN or Inf embeddings")
    return embeddings


def discover(test_root: Path):
    rows = []
    for model_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        task_root = model_dir / "results" / "v2v"
        if not task_root.is_dir():
            continue
        for sample_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
            out_dir = sample_dir / "LLM" / "project" / "out"
            if not out_dir.is_dir():
                continue
            for video in sorted(p for p in out_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                rows.append({"model": model_dir.name, "sample_id": sample_dir.name,
                             "task": "v2v",
                             "video_variant": "0web_export" if video.name.startswith("0web_") else "main_export",
                             "generated_video_relative": video.relative_to(test_root).as_posix(),
                             "generated_video": str(video)})
    return rows


def sample_frames_4fps(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        cap.release()
        raise ValueError(f"invalid video metadata fps={fps} count={count}: {path}")
    indices = np.unique(np.rint(np.arange(0, count, max(1.0, fps / SAMPLE_FPS))).astype(int))
    target = set(int(i) for i in indices)
    frames, decoded = [], []
    idx = 0
    try:
        while idx <= int(indices[-1]):
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in target:
                frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                decoded.append(idx)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames decoded: {path}")
    times = [float(i / fps) for i in decoded]
    return frames, times, fps, count


def bounded_subsequence_dtw(sim_ref_gen: np.ndarray) -> dict:
    """Open-end subsequence DTW with a span floor, avoiding one-frame collapse."""
    n, m = sim_ref_gen.shape
    if not n or not m:
        raise ValueError("empty DINOv3 similarity matrix")
    cost = 1.0 - sim_ref_gen.astype(np.float32, copy=False)
    dp = np.empty_like(cost)
    parent = np.zeros((n, m), dtype=np.uint8)
    dp[0] = cost[0]
    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
        parent[i, 0] = 2
        for j in range(1, m):
            vals = (dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
            move = int(np.argmin(vals))
            dp[i, j] = vals[move] + cost[i, j]
            parent[i, j] = (1, 2, 3)[move]
    # The quarter-span floor is intentionally permissive: it rules out a
    # single lucky generated frame while allowing clips whose matching passage
    # is much shorter than the source.
    min_span = max(2, int(np.ceil(n * MIN_GENERATED_SPAN_RATIO)))
    candidates = []
    for end_j in range(m):
        i, j, path = n - 1, end_j, []
        while True:
            path.append((i, j))
            if i == 0:
                break
            move = int(parent[i, j])
            if move == 1:
                i -= 1; j -= 1
            elif move == 2:
                i -= 1
            else:
                j -= 1
        path.reverse()
        span = path[-1][1] - path[0][1] + 1
        if span < min_span:
            continue
        vals = np.asarray([sim_ref_gen[a, b] for a, b in path], dtype=np.float32)
        candidates.append((float(vals.mean()), path, float(vals.min())))
    if not candidates:
        raise ValueError(f"no subsequence path spans at least {min_span} sampled generated frames")
    mean, path, minimum = max(candidates, key=lambda x: (x[0], len(x[1])))
    return {"similarity_mean": mean, "similarity_min": minimum, "path_length": len(path),
            "reference_start_index": path[0][0], "reference_end_index": path[-1][0],
            "generated_start_index": path[0][1], "generated_end_index": path[-1][1],
            "generated_span_frames": path[-1][1] - path[0][1] + 1,
            "generated_span_ratio_of_reference": (path[-1][1] - path[0][1] + 1) / max(n, 1),
            "minimum_generated_span_ratio": MIN_GENERATED_SPAN_RATIO}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--input-root", type=Path, default=HERE.parent.parent / "inputs")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    rows = discover(args.test_root)
    previous = []
    if args.only_unscored and args.output.is_file():
        previous = json.loads(args.output.read_text()).get("results", [])
        done = {r["generated_video_relative"] for r in previous}
        rows = [r for r in rows if r["generated_video_relative"] not in done]
    rows = rows[: args.limit or None]
    print(f"device={DEVICE} videos={len(rows)}", flush=True)
    _load_model()
    refs: dict[str, tuple] = {}
    results, errors = [], []
    started = time.time()
    for i, row in enumerate(rows, 1):
        try:
            sid = row["sample_id"]
            if sid not in refs:
                clip_path = args.input_root / sid / "keyclip.mp4"
                if not clip_path.is_file():
                    raise FileNotFoundError(f"missing source keyclip: {clip_path}")
                frames, times, fps, count = sample_frames_4fps(clip_path)
                refs[sid] = (clip_path, frames, times, encode_pil_batch(frames))
            clip_path, ref_frames, ref_times, ref_emb = refs[sid]
            gen_frames, gen_times, gen_fps, gen_count = sample_frames_4fps(
                Path(row["generated_video"]))
            gen_emb_chunks = [encode_pil_batch(gen_frames[s:s + BATCH_SIZE])
                              for s in range(0, len(gen_frames), BATCH_SIZE)]
            gen_emb = np.concatenate(gen_emb_chunks)
            sim = ref_emb @ gen_emb.T  # [reference, generated]
            dtw = bounded_subsequence_dtw(sim)
            dtw["generated_start_timestamp_s"] = float(gen_times[dtw["generated_start_index"]])
            dtw["generated_end_timestamp_s"] = float(gen_times[dtw["generated_end_index"]])
            results.append({"task": "v2v", "model": row["model"], "sample_id": sid,
                            "variant": row["video_variant"],
                            "generated_video_relative": row["generated_video_relative"],
                            "generated_video": row["generated_video"],
                            "source_keyclip": str(clip_path),
                            "video_metadata": {"fps": gen_fps, "frame_count": gen_count,
                                               "duration_s": gen_count / gen_fps,
                                               "sample_fps": SAMPLE_FPS,
                                               "sampled_frames": len(gen_frames)},
                            "metric_protocol": "v2v_dinov3_subsequence_dtw_v2_minspan25",
                            "subsequence_dtw": dtw, "success": True})
            print(f"{i}/{len(rows)} {row['model']} {sid} SA={dtw['similarity_mean']:.4f} "
                  f"window={dtw['generated_start_timestamp_s']:.2f}-"
                  f"{dtw['generated_end_timestamp_s']:.2f}s", flush=True)
        except Exception as exc:
            errors.append({"model": row["model"], "sample_id": row["sample_id"],
                           "generated_video_relative": row["generated_video_relative"],
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i}/{len(rows)} ERROR {row['generated_video_relative']}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "task": "v2v",
        "metric": "Sequence Alignment",
        "metric_protocol": "v2v_dinov3_subsequence_dtw_v2_minspan25",
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION, "device": str(DEVICE)},
        "sampling_fps": SAMPLE_FPS,
        "minimum_generated_span_ratio": MIN_GENERATED_SPAN_RATIO,
        "span_policy": "most-similar monotone path with at least 25% of the source "
                       "sampled-frame span and at least 2 generated samples",
        "n_success": len(merged), "n_errors": len(errors), "errors": errors,
        "results": sorted(merged.values(), key=lambda r: r["generated_video_relative"]),
    }
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(f"finished={len(results)}/{len(rows)} errors={len(errors)} "
          f"elapsed_min={(time.time() - started) / 60:.1f} output={args.output}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
