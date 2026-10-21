#!/usr/bin/env python3
"""Motion Smoothness (MS): VBench-style AMT-S interpolation at 12 FPS, 512 short side.

Setup:
    git clone https://github.com/Vchitect/VBench third_party/VBench
    pip install scenedetect opencv-python-headless
    # download amt-s.pth (see the VBench/AMT release) under checkpoints/amt/

Each output video is resampled to uniform 12-FPS timestamps and downscaled only
when needed so its short side is at most 512 pixels. Odd sampled frames are
hidden, and the official VBench-vendored AMT-S predicts them from the
neighboring even frames. Interpolation targets whose endpoints straddle a
detected hard cut (PySceneDetect ContentDetector) are excluded. Score is
1 - mean(valid-frame RGB MAE on 0-255) / 255.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_THREADS", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scenedetect import open_video, SceneManager  # noqa: E402
from scenedetect.detectors import ContentDetector  # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = ("t2v", "i2v", "v2v")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TARGET_FPS = 12.0
SHORT_SIDE = 512
CUT_THRESHOLD = 27.0
CUT_MIN_SCENE_LEN = 15
BATCH_SIZE = 4
VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
AMT_CHECKPOINT_SOURCE = ("https://huggingface.co/lalala125/AMT/resolve/"
                         "31d0420ef7788f6dc22aad3445148705007141bb/amt-s.pth")


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
                for path in sorted(p for p in out_dir.iterdir()
                                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                    rows.append({"task": task, "model": model_dir.name, "sample_id": sample_dir.name,
                                 "variant": "0web_export" if path.name.startswith("0web_") else "main_export",
                                 "path": path,
                                 "generated_video_relative": path.relative_to(test_root).as_posix()})
    return rows


def load_model(vbench_repo: Path, checkpoint: Path, device):
    sys.path.insert(0, str(vbench_repo))
    sys.path.insert(0, str(vbench_repo / "vbench" / "third_party" / "amt"))
    from vbench.third_party.amt.utils.build_utils import build_from_cfg  # noqa: E402
    network_cfg = yaml.safe_load((vbench_repo / "vbench/third_party/amt/cfgs/AMT-S.yaml").read_text())["network"]
    model = build_from_cfg(network_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def detect_hard_cuts(path: Path):
    started = time.time()
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=CUT_THRESHOLD, min_scene_len=CUT_MIN_SCENE_LEN))
    manager.detect_scenes(video, show_progress=False)
    scenes = manager.get_scene_list()
    return sorted({int(start.frame_num) for start, _ in scenes[1:]}), time.time() - started


def vbench_scale(height: int, width: int, device: torch.device) -> float:
    """Official VBench VRAM-aware resolution scale for the interpolator."""
    if device.type != "cuda":
        return 1.0
    vram = float(torch.cuda.get_device_properties(device).total_memory)
    anchor_resolution = 1024 * 512
    anchor_memory = 1500 * 1024 ** 2
    anchor_memory_bias = 2500 * 1024 ** 2
    scale = anchor_resolution / (height * width) * math.sqrt(max(vram - anchor_memory_bias, 1) / anchor_memory)
    scale = min(1.0, scale)
    denom = math.floor(1.0 / math.sqrt(scale) * 16)
    return 16.0 / max(denom, 1)


def is_cut_crossing(left_source_frame: int, right_source_frame: int, cut_frames: list[int]) -> bool:
    # A cut at the left endpoint starts a new shot and is safe; any cut after
    # that endpoint through the right endpoint makes the interpolation target
    # straddle the shot boundary.
    return any(left_source_frame < cut <= right_source_frame for cut in cut_frames)


def score_video(model, path: Path, device: torch.device, batch_size: int = BATCH_SIZE,
                target_fps: float = TARGET_FPS, short_side: int = SHORT_SIDE):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_width, source_height = width, height
    if not math.isfinite(source_fps) or source_fps <= 0 or source_count < 3:
        cap.release()
        raise ValueError(f"invalid video metadata fps={source_fps}, frames={source_count}")
    target_fps = source_fps if target_fps <= 0 else target_fps
    if source_fps < target_fps - 1e-3:
        cap.release()
        raise ValueError(f"source FPS {source_fps:g} is below requested {target_fps:g} FPS; "
                         "avoid manufacturing duplicate frames")
    if short_side > 0 and min(width, height) > short_side:
        resize_ratio = short_side / min(width, height)
        width = max(1, round(width * resize_ratio))
        height = max(1, round(height * resize_ratio))

    target_count = int(math.floor((source_count - 1) * target_fps / source_fps)) + 1
    source_to_targets = {}
    for target_idx in range(target_count):
        source_idx = int(round(target_idx * source_fps / target_fps))
        if source_idx >= source_count:
            continue
        source_to_targets.setdefault(source_idx, []).append(target_idx)
    cut_frames, cut_detection_seconds = detect_hard_cuts(path)
    scale = vbench_scale(height, width, device)
    padding = max(1, int(16 / scale))
    from vbench.third_party.amt.utils.utils import InputPadder  # noqa: E402
    padder = InputPadder((1, 3, height, width), padding)
    embt = torch.tensor(0.5, dtype=torch.float32, device=device).view(1, 1, 1, 1)

    candidate_count = len(range(1, target_count - 1, 2))
    valid_scores: list[float] = []
    valid_timestamps: list[float] = []
    cut_timestamps: list[float] = []
    pending: list = []
    inference_started = time.time()

    def flush_pending():
        if not pending:
            return
        # Transfer compact uint8 frames once; RGB conversion, normalization,
        # padding, uint8 output quantization, and MAE reduction run on the GPU.
        packed = np.stack([[item[0], item[2], item[1]] for item in pending])
        frames = torch.from_numpy(packed).to(device).permute(0, 1, 4, 2, 3)
        frames = frames[:, :, [2, 1, 0], :, :]  # OpenCV BGR -> model RGB, targets included
        in0 = padder.pad(frames[:, 0].float().div_(255.0))
        in1 = padder.pad(frames[:, 1].float().div_(255.0))
        embt_batch = embt.expand(len(pending), -1, -1, -1)
        with torch.inference_mode():
            pred = padder.unpad(model(in0, in1, embt_batch, scale_factor=scale, eval=True)["imgt_pred"])
            # Official tensor2img: multiply by 255, clip, then truncate to
            # uint8. Integer sums reproduce NumPy's MAE without copying
            # full-resolution predicted frames back to the CPU.
            predicted_u8 = pred.mul(255.0).clamp_(0, 255).to(torch.uint8)
            sums = (predicted_u8.to(torch.int16) - frames[:, 2].to(torch.int16)).abs().sum(
                dim=(1, 2, 3), dtype=torch.int64)
            maes = (sums.double() / (height * width * 3)).cpu().tolist()
        for item, mae in zip(pending, maes):
            if not math.isfinite(mae):
                raise ValueError("non-finite AMT interpolation MAE")
            valid_scores.append(mae)
            valid_timestamps.append(item[3])
        pending.clear()

    last_even = None
    middle = None
    try:
        source_idx = 0
        while source_idx < source_count and source_idx <= max(source_to_targets):
            if not cap.grab():
                break
            targets = source_to_targets.get(source_idx)
            if targets:
                ok, bgr = cap.retrieve()
                if not ok:
                    raise OSError(f"failed to decode selected source frame {source_idx}")
                if (height, width) != (source_height, source_width):
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                for target_idx in targets:
                    if target_idx % 2 == 0:
                        if last_even is not None and middle is not None:
                            left_idx, left_target, left_rgb = last_even
                            mid_idx, mid_target, mid_rgb = middle
                            if is_cut_crossing(left_idx, source_idx, cut_frames):
                                cut_timestamps.append(mid_target / target_fps)
                            else:
                                pending.append((left_rgb, mid_rgb, bgr, mid_target / target_fps))
                                if len(pending) >= batch_size:
                                    flush_pending()
                        last_even = (source_idx, target_idx, bgr)
                        middle = None
                    else:
                        middle = (source_idx, target_idx, bgr)
            source_idx += 1
    finally:
        cap.release()
    flush_pending()
    inference_seconds = time.time() - inference_started
    if len(valid_scores) == 0:
        raise ValueError(f"no valid interpolation targets after hard-cut exclusion "
                         f"({candidate_count} candidates)")
    mean_mae = float(np.mean(valid_scores))
    score = float(1.0 - mean_mae / 255.0)
    if not math.isfinite(score):
        raise ValueError("non-finite Motion Smoothness score")
    return {
        "source_fps": source_fps, "target_fps": target_fps, "source_frame_count": source_count,
        "source_width": source_width, "source_height": source_height,
        "resize_short_side_limit": short_side, "resampled_frame_count": target_count,
        "width": width, "height": height, "amt_scale_factor": scale,
        "scene_detector": "PySceneDetect ContentDetector", "scene_cut_threshold": CUT_THRESHOLD,
        "scene_cut_min_scene_len_source_frames": CUT_MIN_SCENE_LEN,
        "detected_hard_cut_count": len(cut_frames),
        "cut_detection_elapsed_seconds": round(cut_detection_seconds, 3),
        "interpolation_elapsed_seconds": round(inference_seconds, 3),
        "detected_hard_cut_source_frame_indices": cut_frames,
        "candidate_interpolated_frame_count": candidate_count,
        "hard_cut_excluded_frame_count": len(cut_timestamps),
        "hard_cut_excluded_timestamps_s": cut_timestamps,
        "valid_interpolated_frame_count": len(valid_scores),
        "effective_frame_coverage": float(len(valid_scores) / candidate_count) if candidate_count else 0.0,
        "mean_mae_0_255": mean_mae, "motion_smoothness": score,
        "frame_level_mae": [
            {"timestamp_s": float(t), "mae_0_255": float(v), "smoothness": float(1.0 - v / 255.0)}
            for t, v in zip(valid_timestamps, valid_scores)
        ],
    }


def protocol_name(target_fps=TARGET_FPS, short_side=SHORT_SIDE):
    if target_fps == 24 and short_side == 0:
        return "vbench_motion_smoothness_amt_s_24fps_v1"
    cadence = "native" if target_fps <= 0 else f"{target_fps:g}fps"
    return f"codevideo_motion_smoothness_amt_s_{cadence}_short{short_side}_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--vbench-repo", type=Path, default=HERE / "third_party" / "VBench")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints" / "amt" / "amt-s.pth")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS, help="0: native cadence")
    parser.add_argument("--short-side", type=int, default=SHORT_SIDE,
                        help="0: original dimensions; otherwise downscale only")
    args = parser.parse_args()
    if args.batch_size < 1 or args.target_fps < 0 or args.short_side < 0:
        raise SystemExit("invalid runtime settings")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    if not args.vbench_repo.is_dir():
        raise FileNotFoundError(f"VBench repository not found: {args.vbench_repo}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"AMT-S checkpoint not found: {args.checkpoint}")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rows = [r for r in discover(args.test_root) if r["task"] in set(args.tasks)]
    if args.sample_ids is not None:
        rows = [r for r in rows if r["sample_id"] in set(args.sample_ids)]
    previous = (json.loads(args.output.read_text())
                if args.only_unscored and args.output.is_file() else {})
    if previous and previous.get("metric_protocol") != protocol_name(args.target_fps, args.short_side):
        raise SystemExit("Cannot resume a different FPS/resolution protocol into the same output file")
    results = {r["generated_video_relative"]: r for r in previous.get("results", [])}
    errors = {r["generated_video_relative"]: r for r in previous.get("errors", [])}
    rows = [r for r in rows if r["generated_video_relative"] not in results]
    rows = rows[: args.limit or None]
    if not rows:
        print("No unscored videos")
        return
    model = load_model(args.vbench_repo, args.checkpoint, device)
    print(f"device={device} batch={args.batch_size} target_fps={args.target_fps} "
          f"short_side={args.short_side} videos={len(rows)}", flush=True)
    started = time.time()
    for i, row in enumerate(rows, 1):
        key = row["generated_video_relative"]
        item_started = time.time()
        try:
            scores = score_video(model, row["path"], device, args.batch_size,
                                 args.target_fps, args.short_side)
            results[key] = {**{k: v for k, v in row.items() if k != "path"},
                            "generated_video": str(row["path"]), "device": str(device),
                            "metric_protocol": protocol_name(args.target_fps, args.short_side),
                            **scores, "elapsed_seconds": time.time() - item_started, "success": True}
            errors.pop(key, None)
            print(f"{i}/{len(rows)} {row['task']} {row['model']} {row['sample_id']} "
                  f"MS={scores['motion_smoothness']:.4f} "
                  f"coverage={scores['effective_frame_coverage']:.1%}", flush=True)
        except Exception as exc:
            errors[key] = {**{k: v for k, v in row.items() if k != "path"},
                           "error": f"{type(exc).__name__}: {exc}"}
            print(f"{i}/{len(rows)} ERROR {key}: {exc}", flush=True)
        payload = {"metric": "Motion Smoothness",
                   "metric_protocol": protocol_name(args.target_fps, args.short_side),
                   "repository": "https://github.com/Vchitect/VBench",
                   "repository_commit": VBENCH_COMMIT,
                   "amt_source": "https://github.com/mcg-nku/amt",
                   "checkpoint_source": AMT_CHECKPOINT_SOURCE,
                   "checkpoint_path": str(args.checkpoint),
                   "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   "sampling_fps": args.target_fps, "resize_short_side_limit": args.short_side,
                   "score_formula": "1 - mean(valid-frame RGB MAE on 0-255 scale) / 255",
                   "cut_exclusion": {"detector": "PySceneDetect ContentDetector",
                                     "threshold": CUT_THRESHOLD,
                                     "min_scene_len_source_frames": CUT_MIN_SCENE_LEN,
                                     "rule": "exclude targets whose endpoints straddle a detected cut"},
                   "device": str(device), "n_scored": len(results), "n_errors": len(errors),
                   "errors": sorted(errors.values(), key=lambda r: r["generated_video_relative"]),
                   "results": sorted(results.values(), key=lambda r: r["generated_video_relative"])}
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(args.output)
    print(f"finished={len(results)} errors={len(errors)} "
          f"elapsed_min={(time.time() - started) / 60:.1f} output={args.output}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
