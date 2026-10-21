# 08 - Sequence Alignment (SA)

V2V reference-clip alignment with DINOv3 subsequence DTW. The source keyclip
(`inputs/<sample_id>/keyclip.mp4`) and the generated video are sampled at
4 FPS; an open-end monotone subsequence DTW locates the generated interval
that best matches the whole reference in order, requiring the matched span to
cover at least 25% of the reference's sampled frames (and at least 2 frames).
The score is the path-mean DINOv3 cosine.

`09_appearance_fidelity` and `10_motion_fidelity` consume the window this
script locates, so run this one first.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
pip install timm==1.0.27 opencv-python-headless
```

DINOv3 configuration and weights download automatically from the fixed
Hugging Face revision `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da` on first run.
The model identifier includes `@revision`, so this version is used during
loading as well as recorded in the results.

## Run

```bash
python run.py
```

Options: `--device`, `--only-unscored`, `--limit`.
