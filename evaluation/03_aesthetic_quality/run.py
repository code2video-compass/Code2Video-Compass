#!/usr/bin/env python3
"""Aesthetic Quality (AQ): Aesthetic Predictor V2.5 over 1-FPS video frames.

Setup:
    pip install aesthetic-predictor-v2-5 decord
    # predictor weights (aesthetic_predictor_v2_5.pth) from the official
    # discus0434/aesthetic-predictor-v2-5 release, under checkpoints/aesthetic/

Each video is sampled at one frame per second (uniform bin midpoints); every
frame is scored by Aesthetic Predictor V2.5 on the SigLIP SO400M backbone, and
the reported score is the frame-mean. The predictor scale is the official 1-10.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import decord
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TASKS = ("t2v", "i2v", "v2v")
AESTHETIC_SAMPLE_FPS = 1.0
AESTHETIC_BATCH_SIZE = 32
SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"


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
                for video in sorted(p for p in out_dir.iterdir()
                                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                    rows.append({
                        "task": task, "model": model_dir.name, "sample_id": sample_dir.name,
                        "video_variant": "0web_export" if video.name.startswith("0web_") else "main_export",
                        "generated_video_relative": video.relative_to(test_root).as_posix(),
                        "generated_video": str(video),
                    })
    return rows


def read_frames_at_fps(video_path: Path, sample_fps: float):
    """Uniformly sample one frame per time bin; the count scales with duration."""
    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(video_path), num_threads=1)
    length = len(reader)
    source_fps = float(reader.get_avg_fps())
    if length < 1 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"invalid video metadata length={length} fps={source_fps}: {video_path}")
    duration = length / source_fps
    n_bins = max(1, int(np.ceil(duration * sample_fps)))
    times = np.minimum((np.arange(n_bins, dtype=np.float64) + 0.5) / sample_fps,
                       max(0.0, duration - 0.5 / source_fps))
    indices = np.clip(np.rint(times * source_fps).astype(int), 0, length - 1)
    indices = np.unique(indices)
    frames = [Image.fromarray(frame).convert("RGB")
              for frame in reader.get_batch(indices.tolist()).asnumpy()]
    timestamps = [float(i / source_fps) for i in indices]
    return [int(i) for i in indices], timestamps, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--predictor", type=Path,
                        default=HERE / "checkpoints" / "aesthetic" / "aesthetic_predictor_v2_5.pth")
    parser.add_argument("--encoder", default=SIGLIP_MODEL_ID,
                        help="SigLIP backbone: HF hub id or local snapshot directory")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not args.predictor.is_file():
        raise FileNotFoundError(f"predictor weights not found: {args.predictor} (see module docstring)")
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    model, processor = convert_v2_5_from_siglip(
        predictor_name_or_path=str(args.predictor),
        encoder_model_name=args.encoder,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device).eval()
    rows = discover(args.test_root)
    previous = []
    if args.only_unscored and args.output.is_file():
        previous = json.loads(args.output.read_text()).get("results", [])
        done = {r["generated_video_relative"] for r in previous}
        rows = [r for r in rows if r["generated_video_relative"] not in done]
    rows = rows[: args.limit or None]
    print(f"device={device} videos={len(rows)}", flush=True)
    results, errors = [], []
    started = time.time()
    for i, row in enumerate(rows, 1):
        try:
            indices, timestamps, frames = read_frames_at_fps(Path(row["generated_video"]),
                                                             AESTHETIC_SAMPLE_FPS)
            frame_scores = []
            for offset in range(0, len(frames), AESTHETIC_BATCH_SIZE):
                pixels = processor(images=frames[offset:offset + AESTHETIC_BATCH_SIZE],
                                   return_tensors="pt").pixel_values.to(device)
                with torch.inference_mode():
                    values = model(pixels).logits.squeeze(-1).float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Aesthetic Predictor returned non-finite scores")
                frame_scores.extend(float(x) for x in values)
            results.append({**row,
                            "aesthetic_v25": {
                                "sampling_fps": AESTHETIC_SAMPLE_FPS,
                                "sampled_frame_indices": indices,
                                "sampled_frame_timestamps_s": timestamps,
                                "frame_scores": frame_scores,
                                "mean": float(np.mean(frame_scores)),
                                "std_population": float(np.std(frame_scores)),
                                "min": float(np.min(frame_scores)),
                            }})
            print(f"{i}/{len(rows)} {row['model']} {row['sample_id']} "
                  f"AQ={np.mean(frame_scores):.3f} ({len(frame_scores)} frames)", flush=True)
        except Exception as exc:
            errors.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i}/{len(rows)} ERROR {row['generated_video_relative']}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "metric": "Aesthetic Quality",
        "metric_protocol": "aesthetic_predictor_v2_5_frame_mean_v1",
        "model": {"predictor": "Aesthetic Predictor V2.5", "backbone": args.encoder,
                  "predictor_weights": str(args.predictor),
                  "implementation": "https://github.com/discus0434/aesthetic-predictor-v2-5"},
        "sampling": {"fps": AESTHETIC_SAMPLE_FPS,
                     "scheme": "uniform one-frame-per-second bin midpoint",
                     "batch_size": AESTHETIC_BATCH_SIZE},
        "device": str(device),
        "n_videos": len(merged), "n_errors": len(errors), "errors": errors,
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
