# 07 - Perceptual Fidelity (PF) and Structural Fidelity (SF)

I2V reference fidelity against the input keyframe
(`inputs/<sample_id>/keyframe.png`). The whole generated video is sampled at
4 FPS and encoded by DINOv3 (`timm/vit_base_patch16_dinov3.lvd1689m`);
**PF** is the highest keyframe-video cosine, and **SF** is SSIM computed on
that same canonical frame after resizing the generated frame to the reference
dimensions. PSNR is saved as a diagnostic only.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
pip install timm==1.0.27 scikit-image opencv-python-headless
```

DINOv3 configuration and weights download automatically from the fixed
Hugging Face revision `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da` on first run.
The model identifier includes `@revision`, so this version is used during
loading as well as recorded in the results.

## Run

```bash
python run.py
```

Options: `--only-unscored`, `--limit`.
