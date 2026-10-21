# 04 - Motion Smoothness (MS)

AMT interpolation error: predict each odd-index frame from its two neighbors
and score `1 - mean(RGB MAE on 0-255) / 255` over valid middle-frame indices.

Protocol details that affect the score: uniform 12-FPS resampling, short side
capped at 512 px, and interpolation targets that straddle a detected hard cut
(PySceneDetect `ContentDetector`, threshold 27.0, min scene length 15) are
excluded.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
git clone https://github.com/Vchitect/VBench third_party/VBench
pip install scenedetect opencv-python-headless
```

Place the official `amt-s.pth` checkpoint (see the VBench/AMT release) under
`checkpoints/amt/`.

## Run

```bash
python run.py
```

Options: `--vbench-repo`, `--checkpoint`, `--device`, `--tasks`, `--sample-ids`, `--only-unscored`,
`--limit`.
