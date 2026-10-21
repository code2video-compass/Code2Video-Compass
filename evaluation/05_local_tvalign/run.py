#!/usr/bin/env python3
"""Local TVAlign (L-TVA): sliding-window ViCLIP-L against Global Description and Events.

Setup:
    git clone https://github.com/Vchitect/VBench third_party/VBench
    # ViCLIP-L_InternVid-FLT-10M.pth and bpe_simple_vocab_16e6.txt.gz from the
    # official VBench/ViCLIP release, under checkpoints/viclip/

The video is sampled at 4 FPS; consecutive 8-frame windows with stride 4 are
encoded by VBench's ViCLIP-L. The Global Description and every Event text are
split into 30-token chunks, each window takes its maximum cosine over all text
chunks, and windows aggregate LongAV-style: per-window s = clip(2.5 * max(cos, 0), 0, 1),
final = 0.8 * mean + 0.2 * P25. This is the project's sliding-window protocol;
it checks coarse semantics only, not exact text, numbers, layout, or ordering.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metric_core import chunk_text, longav_aggregate, sliding_windows  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TASKS = ("t2v", "i2v", "v2v")
SAMPLE_FPS = 4.0
WINDOW_SIZE = 8
STRIDE = 4
BATCH_SIZE = 8


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
                # 0web_ files are web transcodes/duplicates of the same output.
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
    normalized_events = []
    for idx, event in enumerate(events, 1):
        if isinstance(event, str):
            desc, event_id = event.strip(), f"event_{idx:02d}"
        elif isinstance(event, dict):
            desc = str(event.get("description") or event.get("text") or event.get("summary") or "").strip()
            event_id = str(event.get("event_id") or event.get("id") or f"event_{idx:02d}")
        else:
            continue
        if desc:
            normalized_events.append({"event_id": event_id, "text": desc})
    if not summary:
        raise ValueError(f"{sample_id}: annotation has no Global Description/summary")
    if not normalized_events:
        raise ValueError(f"{sample_id}: annotation has no Event descriptions")
    return {"summary": summary, "events": normalized_events,
            "annotation_path": str(annotation_path)}


def read_video_frames(path: Path, fps: float):
    import decord
    import numpy as np
    from PIL import Image
    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(str(path), num_threads=1)
    n = len(reader)
    source_fps = float(reader.get_avg_fps())
    if n < 1:
        raise ValueError(f"empty video: {path}")
    duration = n / source_fps
    count = max(1, int(np.ceil(duration * fps)))
    target_times = np.minimum((np.arange(count) + 0.5) / fps,
                              max(0.0, duration - 0.5 / source_fps))
    indices = sorted({int(x) for x in np.clip(np.rint(target_times * source_fps), 0, n - 1)})
    if len(indices) < fps:  # preserve the full short-video interval
        indices = list(dict.fromkeys(
            [int(x) for x in np.rint(np.linspace(0, n - 1, max(len(indices), 8))).astype(int)]))
    frames_arr = reader.get_batch(indices).asnumpy()
    frames = [Image.fromarray(np.asarray(a).astype("uint8")).convert("RGB") for a in frames_arr]
    timestamps = [float(i / source_fps) for i in indices]
    return frames, timestamps


def load_viclip(vbench_repo: Path, checkpoint: Path, bpe_path: Path, device):
    """Load the official VBench ViCLIP-L with its checkpoint's EasyDict metadata."""
    import torchvision.transforms as T
    sys.path.insert(0, str(vbench_repo))
    from vbench.third_party.ViCLIP.viclip import ViCLIP  # noqa: E402
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer  # noqa: E402

    tokenizer = SimpleTokenizer(str(bpe_path))
    torch_load = torch.load

    class EasyDict(dict):
        """Compatibility shim for the training-only EasyDict in the checkpoint."""
        __module__ = "utils.easydict"

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

    utils_mod = types.ModuleType("utils"); utils_mod.__path__ = []
    easy_mod = types.ModuleType("utils.easydict"); easy_mod.EasyDict = EasyDict
    old_utils, old_easydict = sys.modules.get("utils"), sys.modules.get("utils.easydict")
    sys.modules["utils"], sys.modules["utils.easydict"] = utils_mod, easy_mod

    def load_official_checkpoint(path, *args, **kwargs):
        # PyTorch >= 2.6 defaults to weights_only=True, which rejects the
        # checkpoint's EasyDict training metadata. Opt out for this fixed
        # local official checkpoint only.
        if str(path) == str(checkpoint):
            kwargs.setdefault("weights_only", False)
        return torch_load(path, *args, **kwargs)

    torch.load = load_official_checkpoint
    try:
        model = ViCLIP(tokenizer=tokenizer, pretrain=str(checkpoint),
                       freeze_text=True).to(device).eval()
    finally:
        torch.load = torch_load
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
        if old_easydict is None:
            sys.modules.pop("utils.easydict", None)
        else:
            sys.modules["utils.easydict"] = old_easydict
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=False),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    return model, tokenizer, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=HERE.parent.parent / "generated")
    parser.add_argument("--input-root", type=Path, default=HERE.parent.parent / "inputs")
    parser.add_argument("--vbench-repo", type=Path, default=HERE / "third_party" / "VBench")
    parser.add_argument("--viclip-checkpoint", type=Path,
                        default=HERE / "checkpoints" / "viclip" / "ViCLIP-L_InternVid-FLT-10M.pth")
    parser.add_argument("--bpe", type=Path,
                        default=HERE / "checkpoints" / "viclip" / "bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--only-unscored", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    import numpy as np  # noqa: F401  (frames path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    for path in (args.vbench_repo, args.viclip_checkpoint, args.bpe):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (see module docstring)")
    model, tokenizer, transform = load_viclip(args.vbench_repo, args.viclip_checkpoint,
                                              args.bpe, device)
    rows = [r for r in discover(args.test_root) if r["task"] in set(args.tasks)]
    previous = []
    if args.only_unscored and args.output.is_file():
        previous = json.loads(args.output.read_text()).get("results", [])
        done = {r["generated_video_relative"] for r in previous}
        rows = [r for r in rows if r["generated_video_relative"] not in done]
    rows = rows[: args.limit or None]
    print(f"device={device} videos={len(rows)}", flush=True)
    import torch.nn.functional as F
    results, errors = [], []
    for i, row in enumerate(rows, 1):
        try:
            info = load_sample_text(row["sample_id"], args.input_root)
            sources = ([{"source_type": "global_description", "source_id": "global",
                         "text": info["summary"]}]
                       + [{"source_type": "event", "source_id": e["event_id"], "text": e["text"]}
                          for e in info["events"]])
            chunks, chunk_meta = [], []
            for source in sources:
                pieces, _ = chunk_text(source["text"], tokenizer.encode, tokenizer.decode, 30)
                for part, text in enumerate(pieces, 1):
                    chunk_meta.append({"text_chunk_id": len(chunks), "source_type": source["source_type"],
                                       "source_id": source["source_id"], "part": part, "text": text})
                    chunks.append(text)
            frames, timestamps = read_video_frames(Path(row["generated_video"]), SAMPLE_FPS)
            win_indices = sliding_windows(len(frames), WINDOW_SIZE, STRIDE)
            # Very short clips repeat the last boundary frame to satisfy the
            # 8-frame ViCLIP API.
            if len(frames) < WINDOW_SIZE:
                idx = win_indices[0]
                win_indices = [[idx[min(j, len(idx) - 1)] for j in range(WINDOW_SIZE)]]
            text_embs = []
            for start in range(0, len(chunks), BATCH_SIZE):
                with torch.inference_mode():
                    emb = model.encode_text(chunks[start:start + BATCH_SIZE]).float()
                    text_embs.append(F.normalize(emb, dim=-1))
            text_features = torch.cat(text_embs, dim=0).to(device)
            window_features = []
            half = max(1, BATCH_SIZE // 2)
            for start in range(0, len(win_indices), half):
                batch = win_indices[start:start + half]
                tensors = torch.stack([torch.stack([transform(frames[j]) for j in ids])
                                       for ids in batch]).to(device)
                with torch.inference_mode():
                    feature = model.encode_vision(tensors, test=True).float()
                    window_features.append(F.normalize(feature, dim=-1))
            video_features = torch.cat(window_features, dim=0)
            sims = (video_features @ text_features.T).float().cpu().numpy()
            per_window = sims.max(axis=1)
            local_events = []
            for event_id in sorted({s["source_id"] for s in sources if s["source_type"] == "event"}):
                cols = [x["text_chunk_id"] for x in chunk_meta
                        if x["source_type"] == "event" and x["source_id"] == event_id]
                sub = sims[:, cols]
                best = np.unravel_index(int(np.argmax(sub)), sub.shape)
                local_events.append({"event_id": event_id,
                                     "best_cosine_similarity": float(sub[best]),
                                     "best_window_index": int(best[0])})
            local_scaled, local_raw, saturation = longav_aggregate(per_window.tolist())
            results.append({**row, "status": "ok",
                            "viclip_sw_tvalign_local": local_scaled,
                            "viclip_sw_tvalign_local_raw": local_raw,
                            "local_saturation_ratio": saturation,
                            "window_best_cosines_raw": [float(x) for x in per_window],
                            "local_event_scores": local_events,
                            "sampling": {"fps": SAMPLE_FPS, "window_size": WINDOW_SIZE,
                                         "stride": STRIDE, "window_count": len(win_indices)},
                            "text": {"annotation_path": info["annotation_path"],
                                     "global_description": info["summary"],
                                     "events": info["events"],
                                     "text_chunk_count": len(chunks)}})
            print(f"{i}/{len(rows)} {row['model']} {row['sample_id']} "
                  f"L-TVA={local_scaled:.4f}", flush=True)
        except Exception as exc:
            errors.append({**row, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{i}/{len(rows)} ERROR {row['generated_video_relative']}: {exc}", flush=True)
    merged = {r["generated_video_relative"]: r for r in previous}
    merged.update({r["generated_video_relative"]: r for r in results})
    payload = {
        "metric": "Local TVAlign",
        "metric_protocol": "viclip_sw_tvalign_local_v2",
        "model": {"name": "ViCLIP-L / InternVid-FLT-10M",
                  "implementation": "official VBench ViCLIP",
                  "checkpoint": str(args.viclip_checkpoint)},
        "sampling": {"fps": SAMPLE_FPS, "window_size": WINDOW_SIZE, "stride": STRIDE},
        "aggregation": "per-window s=clip(2.5*max(cosine,0),0,1); final 0.8*mean+0.2*P25; "
                       "text chunks cover the Global Description and every Event",
        "device": str(device),
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
