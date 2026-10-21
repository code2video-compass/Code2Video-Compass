# 03 - Aesthetic Quality (AQ)

Frame-level Aesthetic Predictor V2.5 on the SigLIP SO400M backbone. Videos are
sampled at one frame per second (uniform bin midpoints), every frame is scored
on the original 1-10 scale, and the video score is the frame mean.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
pip install aesthetic-predictor-v2-5 decord
```

Place `aesthetic_predictor_v2_5.pth` from the official
`discus0434/aesthetic-predictor-v2-5` release under `checkpoints/aesthetic/`.
The SigLIP encoder downloads automatically from Hugging Face on first run.

## Run

```bash
python run.py
```

Options: `--predictor`, `--encoder`, `--device`, `--only-unscored`, `--limit`.
