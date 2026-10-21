#!/usr/bin/env python3
"""Motion Fidelity (MF) for V2V: CoTracker3 tracklets inside the DINOv3 window.

Setup:
    git clone https://github.com/facebookresearch/co-tracker third_party/CoTracker3
    cd third_party/CoTracker3 && git checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
    # scaled_offline.pth from huggingface.co/facebook/cotracker3 under
    # checkpoints/cotracker3/

Reads the matched interval produced by 08_sequence_alignment. The full source
keyclip and the matched generated interval are each sampled at up to 24
uniformly spaced frames (resized so the longer side is at most 512 px);
CoTracker3 tracks a regular 20x20 point grid on both. Each consecutive-step
tracklet (displacement vectors with both endpoints visible) is compared by
cosine plus magnitude ratio; a pair needs at least 25% of the steps jointly
visible. The score is the symmetric mean of per-tracklet best matches in both
directions. Frames are sampled proportionally within each interval — this is
not exact framewise synchronization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N_FRAMES = 24
GRID_SIZE = 20
MIN_VALID_STEP_FRACTION = 0.25
REPOSITORY_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"


def window_indices(path: Path, start_s: float | None, end_s: float | None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError(f"invalid video metadata fps={fps} count={count}: {path}")
    lo = 0 if start_s is None else max(0, min(count - 1, int(round(start_s * fps))))
    hi = count - 1 if end_s is None else max(lo, min(count - 1, int(round(end_s * fps))))
    return fps, count, lo, hi


def uniform_video_window(path: Path, nframes: int, start_s: float | None = None, end_s: float | None = None):
    fps, count, lo, hi = window_indices(path, start_s, end_s)
    take = min(nframes, hi - lo + 1)
    idx = np.unique(np.rint(np.linspace(lo, hi, take)).astype(int))
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"cannot decode frame {i} from {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(1.0, 512 / max(h, w))
            if scale < 1:
                rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    return np.stack(frames), idx.tolist(), fps, count, (lo / fps, hi / fps)


def run_tracks(model, path: Path, nframes: int, device, start_s=None, end_s=None):
    arr, idx, fps, count, window = uniform_video_window(path, nframes, start_s, end_s)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        tracks, vis = model(video, grid_size=GRID_SIZE)
    xy = tracks[0].detach().float().cpu().numpy()
    visible = vis[0].detach().bool().cpu().numpy()
    h, w = arr.shape[1:3]
    xy[..., 0] /= max(w - 1, 1)
    xy[..., 1] /= max(h - 1, 1)
    return {"tracks": xy, "visible": visible, "sampled_frame_indices": idx,
            "fps": fps, "frame_count": count, "window_s": window,
            "duration_s": (window[1] - window[0])}


def chamfer_tracklet_score(ref, gen):
    a, b = ref["tracks"], gen["tracks"]
    va, vb = ref["visible"], gen["visible"]
    if a.shape[0] != b.shape[0]:
        raise ValueError("paired tracklets must use same normalized window sample count")
    da = np.diff(a, axis=0)
    db = np.diff(b, axis=0)
    ma = va[:-1] & va[1:]
    mb = vb[:-1] & vb[1:]
    na = np.linalg.norm(da, axis=-1)
    nb = np.linalg.norm(db, axis=-1)
    needed = max(1, int(math.ceil((a.shape[0] - 1) * MIN_VALID_STEP_FRACTION)))
    valid = ma[:, :, None] & mb[:, None, :]
    dot = np.sum(da[:, :, None, :] * db[:, None, :, :], axis=-1)
    sa = na[:, :, None]
    sb = nb[:, None, :]
    den = sa * sb
    cos = np.divide(dot, den, out=np.zeros_like(dot), where=den > 1e-8)
    static = (sa <= 1e-8) & (sb <= 1e-8)
    cos[static] = 1.0
    ratio = np.divide(np.minimum(sa, sb), np.maximum(sa, sb), out=np.zeros_like(den),
                      where=np.maximum(sa, sb) > 1e-8)
    ratio[static] = 1.0
    pair = 0.5 * (cos + ratio)
    counts = valid.sum(axis=0)
    sims = np.full(counts.shape, np.nan, dtype=np.float32)
    good = counts >= needed
    sums = np.where(valid, pair, 0.0).sum(axis=0)
    sims[good] = (sums[good] / counts[good]).astype(np.float32)
    valid_a = np.isfinite(sims).any(axis=1)
    valid_b = np.isfinite(sims).any(axis=0)
    if not valid_a.any() or not valid_b.any():
        raise ValueError("no valid tracklet pair has enough jointly visible steps")
    score = float(0.5 * (np.nanmax(sims[valid_a], axis=1).mean() + np.nanmax(sims[:, valid_b], axis=0).mean()))
    return {"tracklet_chamfer_similarity": score,
            "reference_tracklet_count": int(a.shape[1]),
            "generated_tracklet_count": int(b.shape[1]),
            "valid_reference_tracklet_count": int(valid_a.sum()),
            "valid_generated_tracklet_count": int(valid_b.sum()),
            "valid_reference_tracklet_coverage": float(valid_a.mean()),
            "valid_generated_tracklet_coverage": float(valid_b.mean()),
            "valid_pair_count": int(np.isfinite(sims).sum()),
            "pairwise_tracklet_matrix_shape": list(sims.shape),
            "minimum_jointly_visible_steps": needed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--input-root", type=Path, default=HERE.parent.parent / "inputs")
    parser.add_argument("--alignment", type=Path,
                        default=HERE.parent / "08_sequence_alignment" / "results.json",
                        help="results.json written by 08_sequence_alignment")
    parser.add_argument("--cotracker-repo", type=Path, default=HERE / "third_party" / "CoTracker3")
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "checkpoints" / "cotracker3" / "scaled_offline.pth")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"{args.checkpoint} not found (see module docstring)")
    if not args.cotracker_repo.is_dir():
        raise FileNotFoundError(f"{args.cotracker_repo} not found (see module docstring)")
    if not args.alignment.is_file():
        raise FileNotFoundError(f"sequence-alignment results not found: {args.alignment} "
                                "(run 08_sequence_alignment first)")
    sys.path.insert(0, str(args.cotracker_repo))
    model = torch.hub.load(str(args.cotracker_repo), "cotracker3_offline",
                           source="local", pretrained=False).to(device).eval()
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.model.load_state_dict(weights)
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
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
    cache = {}
    started = time.time()
    for n, row in enumerate(rows, 1):
        try:
            sid = row["sample_id"]
            ref = args.input_root / sid / "keyclip.mp4"
            gen = args.test_root / row["generated_video_relative"]
            dtw = row.get("subsequence_dtw") or {}
            start = float(dtw["generated_start_timestamp_s"])
            end = float(dtw["generated_end_timestamp_s"])
            rf, rc, rlo, rhi = window_indices(ref, None, None)
            gf, gc, glo, ghi = window_indices(gen, start, end)
            t = min(N_FRAMES, rhi - rlo + 1, ghi - glo + 1)
            if t < 3:
                raise ValueError(f"DINO matched segment too short for CoTracker: {t} frames")
            kr = (str(ref), None, None, t)
            kg = (str(gen), start, end, t)
            if kr not in cache:
                cache[kr] = run_tracks(model, ref, t, device)
            if kg not in cache:
                cache[kg] = run_tracks(model, gen, t, device, start, end)
            a, b = cache[kr], cache[kg]
            match = chamfer_tracklet_score(a, b)
            results.append({
                "task": "v2v", "model": row["model"], "sample_id": sid,
                "variant": row.get("variant", "main_export"),
                "generated_video_relative": row["generated_video_relative"],
                "source_clip": str(ref),
                "metric_protocol": "v2v_cotracker_dino_aligned_window_v1",
                "repository": "https://github.com/facebookresearch/co-tracker",
                "repository_commit": REPOSITORY_COMMIT,
                "checkpoint_name": "scaled_offline.pth",
                "checkpoint_sha256": checkpoint_sha256,
                "sampling": f"{t} uniformly spaced frames per clip window; source full keyclip "
                            "and generated DINOv3-matched interval",
                "temporal_alignment": "DINOv3 subsequence DTW locates the generated time interval; "
                                      "CoTracker uses normalized sampling within each interval and does "
                                      "not warp tracklets by the DTW path. This is not exact framewise "
                                      "synchronization.",
                "dino_matched_generated_window_s": {"start": start, "end": end},
                "grid_size": GRID_SIZE,
                "min_valid_step_fraction": MIN_VALID_STEP_FRACTION,
                "spatial_roi": "none; full frame",
                "point_initialization": "regular grid from official CoTracker3 API",
                **match,
                "reference_window_s": {"start": a["window_s"][0], "end": a["window_s"][1]},
                "generated_window_s": {"start": b["window_s"][0], "end": b["window_s"][1]},
                "reference_duration_s": a["duration_s"],
                "generated_duration_s": b["duration_s"],
                "reference_sampled_frame_indices": a["sampled_frame_indices"],
                "generated_sampled_frame_indices": b["sampled_frame_indices"],
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "success": True,
            })
            print(f"{n}/{len(rows)} {row['model']} {sid} MF={match['tracklet_chamfer_similarity']:.4f}", flush=True)
        except Exception as exc:
            errors.append({"generated_video_relative": row.get("generated_video_relative"),
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"{n}/{len(rows)} ERROR {row.get('generated_video_relative')}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "task": "v2v", "metric": "Motion Fidelity",
        "protocol": "v2v_cotracker_dino_aligned_window_v1",
        "requires": "08_sequence_alignment matched windows",
        "repository_commit": REPOSITORY_COMMIT,
        "checkpoint_sha256": checkpoint_sha256,
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
