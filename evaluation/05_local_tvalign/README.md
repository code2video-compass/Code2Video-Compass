# 05 - Local TVAlign (L-TVA)

Sliding-window text-video alignment with ViCLIP-L. The video is sampled at
4 FPS into 8-frame windows with stride 4; the Global Description and Event
texts are split into 30-token chunks; each window takes its maximum cosine
over all chunks; windows aggregate LongAV-style
(`s = clip(2.5 * max(cos, 0), 0, 1)`, final `0.8 * mean + 0.2 * P25`).

Texts come from `inputs/<sample_id>/annotation.json` (Global Description and
Events). This checks coarse semantics only, not exact text, numbers, layout,
or ordering.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
git clone https://github.com/Vchitect/VBench third_party/VBench
```

Place `ViCLIP-L_InternVid-FLT-10M.pth` and `bpe_simple_vocab_16e6.txt.gz`
from the official VBench/ViCLIP release under `checkpoints/viclip/`.

## Run

```bash
python run.py
```

Options: `--vbench-repo`, `--viclip-checkpoint`, `--bpe`, `--device`,
`--tasks`, `--only-unscored`, `--limit`.
