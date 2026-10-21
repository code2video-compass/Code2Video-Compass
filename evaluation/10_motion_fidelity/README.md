# 10 - Motion Fidelity (MF)

V2V motion fidelity with CoTracker3 tracklets inside the DINOv3-located
window. Both the full source keyclip and the matched generated interval are
sampled at up to 24 uniformly spaced frames (longer side at most 512 px) and
tracked on a regular 20x20 point grid. Consecutive-step tracklet displacements
are compared by cosine plus magnitude ratio; a pair needs at least 25% of the
steps jointly visible, and the score is the symmetric mean of per-tracklet
best matches in both directions. Frames are sampled proportionally within each
interval; this is not exact framewise synchronization.

Requires the matched window from `08_sequence_alignment` (default
`--alignment ../08_sequence_alignment/results.json`).

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
git clone https://github.com/facebookresearch/co-tracker third_party/CoTracker3
(cd third_party/CoTracker3 && git checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d && pip install -e .)
```

Place `scaled_offline.pth` from `huggingface.co/facebook/cotracker3` under
`checkpoints/cotracker3/`.

## Run

```bash
python run.py
```

Options: `--cotracker-repo`, `--checkpoint`, `--alignment`, `--device`,
`--only-unscored`, `--limit`.
