#!/usr/bin/env python3
"""Appearance Fidelity (AF) for V2V: DreamSim inside the DINOv3-located window.

Setup:
    pip install dreamsim

Reads the matched interval produced by 08_sequence_alignment (results.json).
Both the full source keyclip and the matched generated interval are sampled at
4 FPS and embedded by the official DreamSim ensemble (DINO + CLIP + OpenCLIP
ViT-B/16). A closed-endpoint monotone Sakoe-Chiba DTW (band 0.15, widened just
enough when the generated passage is shorter) minimizes DreamSim cosine
distance; the score is 1 - mean aligned distance. This is appearance
similarity within the matched passage, not a motion score.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from dreamsim import dreamsim

HERE = Path(__file__).resolve().parent
SAMPLE_FPS = 4.0
BATCH = 32
BAND_FRACTION = 0.15


def decode_uniform(path: Path, start_s: float | None = None, end_s: float | None = None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or n <= 0:
        cap.release()
        raise ValueError(f"invalid metadata fps={fps} frames={n}: {path}")
    step = max(1.0, fps / SAMPLE_FPS)
    inds = np.unique(np.rint(np.arange(0, n, step)).astype(int))
    times = inds / fps
    mask = np.ones(len(inds), dtype=bool)
    if start_s is not None:
        mask &= times >= start_s - 1e-6
    if end_s is not None:
        mask &= times <= end_s + 1e-6
    inds = inds[mask]
    if len(inds) < 2:
        cap.release()
        raise ValueError(f"matched window has <2 sampled frames: {path} {start_s}..{end_s}")
    target = set(map(int, inds))
    frames, decoded = [], []
    i = 0
    try:
        while i <= int(inds[-1]):
            ok, bgr = cap.read()
            if not ok:
                break
            if i in target:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                decoded.append(i)
            i += 1
    finally:
        cap.release()
    if len(frames) < 2:
        raise ValueError(f"no frames decoded in selected window: {path}")
    return frames, [float(i / fps) for i in decoded], decoded, fps, n


def embed_frames(model, preprocess, frames, device):
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(frames), BATCH):
            batch = torch.cat([preprocess(Image.fromarray(im))
                               for im in frames[start:start + BATCH]], dim=0).to(device)
            z = model.embed(batch).float()
            z = torch.nn.functional.normalize(z, dim=-1)
            chunks.append(z.cpu())
    return torch.cat(chunks).numpy()


def constrained_dtw(cost, band_fraction=BAND_FRACTION):
    n, m = cost.shape
    dp = np.full((n, m), np.inf)
    prev = np.full((n, m, 2), -1, dtype=np.int32)
    # Keep the default narrow band for similar-length clips. When DINOv3 finds
    # a substantially shorter generated passage, widen the normalized-time band
    # enough to connect both endpoints instead of dropping the sample.
    length_gap = 1.0 - (min(n, m) - 1) / max(max(n, m) - 1, 1)
    effective_band = max(band_fraction, 0.5 * length_gap)
    for i in range(n):
        for j in range(m):
            if (abs(i / max(n - 1, 1) - j / max(m - 1, 1)) > effective_band
                    and (i, j) != (0, 0) and (i, j) != (n - 1, m - 1)):
                continue
            c = float(cost[i, j])
            if i == j == 0:
                dp[i, j] = c
                continue
            opts = []
            if i:
                opts.append((dp[i - 1, j], i - 1, j))
            if j:
                opts.append((dp[i, j - 1], i, j - 1))
            if i and j:
                opts.append((dp[i - 1, j - 1], i - 1, j - 1))
            if opts:
                v, pi, pj = min(opts, key=lambda x: x[0])
                if np.isfinite(v):
                    dp[i, j] = v + c
                    prev[i, j] = [pi, pj]
    if not np.isfinite(dp[-1, -1]):
        raise RuntimeError(f"no constrained DTW path {n}x{m}, band={effective_band}")
    path = []
    i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == j == 0:
            break
        i, j = map(int, prev[i, j])
        if i < 0 or j < 0:
            raise RuntimeError("broken DTW backtrace")
    path.reverse()
    vals = np.asarray([cost[i, j] for i, j in path], dtype=np.float64)
    mean_dist = float(vals.mean())
    return path, mean_dist, float(1.0 - mean_dist), float(vals.sum()), float(dp[-1, -1]), float(effective_band)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--input-root", type=Path, default=HERE.parent.parent / "inputs")
    parser.add_argument("--alignment", type=Path,
                        default=HERE.parent / "08_sequence_alignment" / "results.json",
                        help="results.json written by 08_sequence_alignment")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    model, preprocess = dreamsim(pretrained=True, device=str(device), dreamsim_type="ensemble")
    model.eval()
    if not args.alignment.is_file():
        raise FileNotFoundError(f"sequence-alignment results not found: {args.alignment} "
                                "(run 08_sequence_alignment first)")
    source = json.loads(args.alignment.read_text())
    rows = source.get("results", [])
    previous = []
    if args.only_unscored and args.output.is_file():
        previous = json.loads(args.output.read_text()).get("results", [])
        done = {r["generated_video_relative"] for r in previous}
        rows = [r for r in rows if r["generated_video_relative"] not in done]
    rows = rows[: args.limit or None]
    print(f"device={device} videos={len(rows)}", flush=True)
    results, errors = [], []
    started = time.time()
    for n, row in enumerate(rows, 1):
        try:
            dtw = row.get("subsequence_dtw") or {}
            start = float(dtw["generated_start_timestamp_s"])
            end = float(dtw["generated_end_timestamp_s"])
            sid = row["sample_id"]
            ref_path = args.input_root / sid / "keyclip.mp4"
            gen_path = args.test_root / row["generated_video_relative"]
            ref_frames, ref_times, ref_inds, ref_fps, ref_n = decode_uniform(ref_path)
            gen_frames, gen_times, gen_inds, gen_fps, gen_n = decode_uniform(gen_path, start, end)
            ref_z = embed_frames(model, preprocess, ref_frames, device)
            gen_z = embed_frames(model, preprocess, gen_frames, device)
            cost = 1.0 - ref_z @ gen_z.T
            path, mean_dist, mean_sim, sum_dist, cum_dist, effective_band = constrained_dtw(cost)
            results.append({
                "task": "v2v", "model": row["model"], "sample_id": sid,
                "variant": row.get("variant", "main_export"),
                "generated_video_relative": row["generated_video_relative"],
                "source_clip": str(ref_path),
                "metric_protocol": "v2v_dreamsim_dino_aligned_window_v1",
                "appearance_fidelity": mean_sim,
                "sampling_fps": SAMPLE_FPS,
                "reference_frame_count": len(ref_inds),
                "generated_window_frame_count": len(gen_inds),
                "dino_matched_generated_window_s": {"start": start, "end": end},
                "alignment_window": "full source keyclip vs DINOv3 subsequence-DTW generated interval",
                "dtw_algorithm": "closed-endpoint monotone Sakoe-Chiba DTW minimizing DreamSim cosine distance",
                "dtw_band_fraction": effective_band,
                "dtw_default_band_fraction": BAND_FRACTION,
                "dtw_path_length": len(path),
                "dtw_aligned_distance_mean": mean_dist,
                "dtw_aligned_similarity_mean": mean_sim,
                "dtw_source_start_s": ref_times[path[0][0]],
                "dtw_source_end_s": ref_times[path[-1][0]],
                "dtw_generated_start_s": gen_times[path[0][1]],
                "dtw_generated_end_s": gen_times[path[-1][1]],
                "reference_video_metadata": {"fps": ref_fps, "frame_count": ref_n},
                "generated_video_metadata": {"fps": gen_fps, "frame_count": gen_n},
                "device": str(device), "success": bool(np.isfinite(mean_sim)),
            })
            print(f"{n}/{len(rows)} {row['model']} {sid} AF={mean_sim:.4f}", flush=True)
        except Exception as exc:
            errors.append({"generated_video_relative": row.get("generated_video_relative"),
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"{n}/{len(rows)} ERROR {row.get('generated_video_relative')}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "task": "v2v", "metric": "Appearance Fidelity",
        "metric_protocol": "v2v_dreamsim_dino_aligned_window_v1",
        "model": {"name": "DreamSim ensemble (DINO ViT-B/16 + CLIP ViT-B/16 + OpenCLIP ViT-B/16)",
                  "source": "https://github.com/ssundaram21/dreamsim", "package_version": "0.2.1"},
        "requires": "08_sequence_alignment matched windows",
        "device": str(device),
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
