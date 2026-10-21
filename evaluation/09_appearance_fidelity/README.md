# 09 - Appearance Fidelity (AF)

V2V appearance fidelity with DreamSim inside the DINOv3-located window. The
full source keyclip and the matched generated interval are sampled at 4 FPS
and embedded by the DreamSim ensemble; a closed-endpoint monotone Sakoe-Chiba
DTW (band 0.15, widened when the generated passage is shorter) minimizes
DreamSim cosine distance, and the score is `1 - mean aligned distance`.

Requires the matched window from `08_sequence_alignment` (default
`--alignment ../08_sequence_alignment/results.json`).

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
pip install dreamsim
```

The ensemble weights download automatically on first run.

## Run

```bash
python run.py
```

Options: `--alignment`, `--device`, `--only-unscored`, `--limit`.
