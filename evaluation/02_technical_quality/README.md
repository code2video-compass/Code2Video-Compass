# 02 - Technical Quality (TQ)

Video-quality metric from the technical branch of DOVER++. The raw branch
output is mapped through the official fixed calibration
`sigmoid((raw - 0.1107) / 0.07355)`; all models share the same mapping.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
git clone https://github.com/VQAssessment/DOVER third_party/DOVER
(cd third_party/DOVER && git checkout f1ddc96215bc7fbcf8f315c65d47905f339c3419 && pip install -r requirements.txt)
```

Place `DOVER_plus_plus.pth` from the official DOVER release under
`checkpoints/dover/` (expected SHA-256
`7997cac660573daf54bd489cfc4d5d067ebd8bb32e68015e6ca155ae64972b9e`).

## Run

```bash
python run.py
```

Options: `--repo`, `--checkpoint`, `--device`, `--only-unscored`,
`--limit`.
