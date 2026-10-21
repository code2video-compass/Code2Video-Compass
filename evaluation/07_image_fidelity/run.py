#!/usr/bin/env python3
"""Image fidelity for I2V: DINOv3 Perceptual Fidelity (PF) and same-frame SSIM (SF).

The input keyframe (inputs/<sample_id>/keyframe.png) is compared against the
whole generated video sampled at 4 FPS. DINOv3
(timm/vit_base_patch16_dinov3.lvd1689m, downloaded automatically on first run)
finds the canonical matching frame with the highest embedding cosine — that
cosine is PF. On the same frame, SSIM is computed after resizing the generated
frame to the reference dimensions — that is SF. DINO and SSIM are reported
separately and never combined; PSNR is saved only as a diagnostic.
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
SAMPLE_FPS = 4.0
BATCH_SIZE = 16
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
        task_root = model_dir / "results" / "i2v"
        if not task_root.is_dir():
            continue
        for sample_dir in sorted(p for p in task_root.iterdir() if p.is_dir()):
            out_dir = sample_dir / "LLM" / "project" / "out"
            if not out_dir.is_dir():
                continue
            for video in sorted(p for p in out_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                rows.append({"model": model_dir.name, "sample_id": sample_dir.name,
                             "task": "i2v",
                             "video_variant": "0web_export" if video.name.startswith("0web_") else "main_export",
                             "generated_video_relative": video.relative_to(test_root).as_posix(),
                             "generated_video": str(video)})
    return rows


def load_keyframe(sample_id: str, input_root: Path):
    path = input_root / sample_id / "keyframe.png"
    if not path.is_file():
        raise FileNotFoundError(f"No keyframe image found for sample {sample_id}: {path}")
    with Image.open(path) as image:
        return path, image.convert("RGB")


def pixel_metrics(a_rgb: np.ndarray, b_rgb: np.ndarray) -> dict:
    h, w = a_rgb.shape[:2]
    resized = b_rgb.shape[:2] != (h, w)
    if resized:
        b_rgb = cv2.resize(b_rgb, (w, h), interpolation=cv2.INTER_AREA)
    psnr = float(peak_signal_noise_ratio(a_rgb, b_rgb, data_range=255))
    ssim = float(structural_similarity(a_rgb, b_rgb, data_range=255, channel_axis=-1))
    return {"ssim": ssim,
            "psnr_db": psnr if math.isfinite(psnr) else None,
            "generated_frame_resized_to_reference": resized}


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot reopen video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"cannot decode frame {frame_index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def score_one(row: dict, keyframe_path: Path, keyframe: Image.Image) -> dict:
    path = Path(row["generated_video"])
    key_rgb = np.asarray(keyframe, dtype=np.uint8)
    key_emb = encode_pil_batch([keyframe])[0]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open generated video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"invalid generated video metadata: {path} fps={fps} count={frame_count}")
    interval = max(1.0, fps / SAMPLE_FPS)
    target_indices = np.unique(np.rint(np.arange(0, frame_count, interval)).astype(np.int64))
    target_set = set(int(i) for i in target_indices)
    batches, batch_indices, decoded_indices = [], [], []

    def flush():
        if not batches:
            return
        emb = encode_pil_batch(batches)
        sims.extend(float(x) for x in emb @ key_emb)
        decoded_indices.extend(batch_indices)
        batches.clear()
        batch_indices.clear()

    sims: list[float] = []
    last_read = int(target_indices[-1]) if len(target_indices) else -1
    try:
        for idx in range(last_read + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            if idx not in target_set:
                continue
            batches.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            batch_indices.append(idx)
            if len(batches) >= BATCH_SIZE:
                flush()
        flush()
    finally:
        cap.release()
    if not decoded_indices:
        raise ValueError(f"no sampled frames decoded from: {path}")
    actual_indices = np.asarray(decoded_indices, dtype=np.int64)
    key_score = np.asarray(sims, dtype=np.float32)
    best_pos = int(np.argmax(key_score))
    best_index = int(actual_indices[best_pos])
    best_time = float(best_index / fps)
    best_rgb = read_frame(path, best_index)
    pixels = pixel_metrics(key_rgb, best_rgb)
    return {
        "canonical_frame": {"frame_index": best_index, "timestamp_s": best_time,
                            "dino_cosine_similarity": float(key_score[best_pos]),
                            "ssim": pixels["ssim"]},
        "keyframe_perceptual_fidelity": float(key_score[best_pos]),
        "keyframe_structural_fidelity": pixels["ssim"],
        "keyframe_mean_dino_similarity": float(key_score.mean()),
        "best_frame_psnr_db": pixels["psnr_db"],
        "video_metadata": {"fps": fps, "frame_count": frame_count,
                           "duration_s": frame_count / fps,
                           "sample_fps": SAMPLE_FPS, "sampled_frames": len(actual_indices)},
        "keyframe_path": str(keyframe_path),
    }


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
    refs: dict[str, tuple[Path, Image.Image]] = {}
    results, errors = [], []
    started = time.time()
    for i, row in enumerate(rows, 1):
        try:
            sid = row["sample_id"]
            if sid not in refs:
                refs[sid] = load_keyframe(sid, args.input_root)
            record = score_one(row, *refs[sid])
            results.append({**row, "metric_protocol": "i2v_canonical_frame_v1",
                            "sampling_fps": SAMPLE_FPS, **record, "success": True})
            cf = record["canonical_frame"]
            print(f"{i}/{len(rows)} {row['model']} {sid} "
                  f"PF={cf['dino_cosine_similarity']:.4f} SF={cf['ssim']:.4f} "
                  f"@{cf['timestamp_s']:.2f}s", flush=True)
        except Exception as exc:
            errors.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i}/{len(rows)} ERROR {row['generated_video_relative']}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "task": "i2v",
        "metric": "Perceptual Fidelity (PF); Structural Fidelity (SF)",
        "metric_protocol": "i2v_canonical_frame_v1",
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION,
                  "device": str(DEVICE)},
        "sampling_fps": SAMPLE_FPS,
        "note": "whole-video 4 FPS search; SSIM on the DINOv3-selected canonical frame; "
                "PSNR saved as a diagnostic only and not a benchmark metric",
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
