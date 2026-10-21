#!/usr/bin/env python3
"""Technical Quality (TQ): official DOVER++ technical branch, calibrated 0-1.

Setup (once):
    git clone https://github.com/VQAssessment/DOVER third_party/DOVER
    # place DOVER_plus_plus.pth (official release) under checkpoints/dover/

Reads generated videos, applies the official DOVER++ technical branch, and maps
the raw output through the official calibration
sigmoid((raw - 0.1107) / 0.07355). The aesthetic branch is loaded by the joint
model but only the technical field is reported.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TASKS = ("t2v", "i2v", "v2v")
CHECKPOINT_SHA256 = "7997cac660573daf54bd489cfc4d5d067ebd8bb32e68015e6ca155ae64972b9e"
REPOSITORY_COMMIT = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"


def normalize_technical_score(raw_score: float) -> float:
    """Official DOVER technical calibration to a [0, 1] score."""
    z = (float(raw_score) - 0.1107) / 0.07355
    return 1.0 / (1.0 + math.exp(-z))


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
                        "video_path": str(video),
                    })
    return rows


def load_dover(repo: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo))
    from dover.models import DOVER  # noqa: E402
    opt = yaml.safe_load((repo / "dover.yml").read_text())
    model = DOVER(**opt["model"]["args"]).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    return model, opt


def infer_one(model, opt, video: Path, device):
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    sample_types = opt["data"]["val-l1080p"]["args"]["sample_types"]
    samplers = {}
    for view, cfg in sample_types.items():
        if "t_frag" not in cfg:
            samplers[view] = UnifiedFrameSampler(cfg["clip_len"], cfg["num_clips"], cfg["frame_interval"])
        else:
            samplers[view] = UnifiedFrameSampler(cfg["clip_len"] // cfg["t_frag"], cfg["t_frag"],
                                                 cfg["frame_interval"], cfg["num_clips"])
    # Deterministic per-video sampling seed, as in the official evaluation.
    seed = sum((i + 1) * b for i, b in enumerate(str(video).encode())) % 2 ** 31
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    views, frame_inds = spatial_temporal_view_decomposition(str(video), sample_types, samplers)
    mean = torch.tensor([123.675, 116.28, 103.53]); std = torch.tensor([58.395, 57.12, 57.375])
    for name, value in views.items():
        clips = sample_types[name].get("num_clips", 1)
        views[name] = (((value.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
                       .reshape(value.shape[0], clips, -1, *value.shape[2:]).transpose(0, 1).to(device))
    with torch.inference_mode():
        outputs = model(views)
    raw = {}
    for name, tensor in zip(views, outputs):
        vals = tensor.detach().float().cpu().numpy()
        raw[name] = {"mean_score": float(np.mean(vals)), "std_score": float(np.std(vals)),
                     "shape": list(vals.shape), "all_finite": bool(np.isfinite(vals).all())}
    return {"sampling_seed": seed,
            "sampled_frame_indices_by_view": {k: np.asarray(v).astype(int).tolist()
                                              for k, v in frame_inds.items()},
            "raw_output_fields": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--repo", type=Path, default=HERE / "third_party" / "DOVER",
                        help="local clone of VQAssessment/DOVER")
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "checkpoints" / "dover" / "DOVER_plus_plus.pth")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not args.repo.is_dir():
        raise FileNotFoundError(f"DOVER repository not found: {args.repo} (see module docstring)")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"DOVER++ checkpoint not found: {args.checkpoint}")
    rows = discover(args.test_root)
    previous = []
    if args.only_unscored and args.output.is_file():
        previous = json.loads(args.output.read_text()).get("results", [])
        done = {r["generated_video_relative"] for r in previous}
        rows = [r for r in rows if r["generated_video_relative"] not in done]
    rows = rows[: args.limit or None]
    if not rows:
        print("No videos to score")
        return
    model, opt = load_dover(args.repo, args.checkpoint, device)
    print(f"device={device} videos={len(rows)}", flush=True)
    results, errors = [], []
    started = time.time()
    for i, row in enumerate(rows, 1):
        try:
            infer = infer_one(model, opt, Path(row["video_path"]), device)
            technical_raw = infer["raw_output_fields"]["technical"]["mean_score"]
            technical_norm = normalize_technical_score(technical_raw)
            results.append({**row, "device": str(device),
                            **infer,
                            "technical_raw_score": technical_raw,
                            "technical_quality_0_1": technical_norm,
                            "success": all(v["all_finite"] for v in infer["raw_output_fields"].values())})
            print(f"{i}/{len(rows)} {row['model']} {row['sample_id']} "
                  f"TQ={technical_norm:.4f} (raw {technical_raw:.4f})", flush=True)
        except Exception as exc:
            errors.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i}/{len(rows)} ERROR {row['generated_video_relative']}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "metric": "Technical Quality",
        "metric_protocol": "dover_plus_plus_technical_calibrated_v1",
        "normalization": {"raw_field": "raw_output_fields.technical.mean_score",
                          "z_score": "(raw - 0.1107) / 0.07355",
                          "score_0_1": "sigmoid(z_score)",
                          "source": "VQAssessment/DOVER evaluate_a_set_of_videos.py fuse_results"},
        "repository": "https://github.com/VQAssessment/DOVER",
        "repository_commit": REPOSITORY_COMMIT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
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
