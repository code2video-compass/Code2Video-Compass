# 06 - Global TVAlign (G-TVA)

Whole-video semantic alignment with Qwen3-VL-Embedding-8B. The Global
Description and Event texts are joined into one semantic text and encoded
independently from the video (at most 64 uniformly sampled frames); the score
is the cosine of the two normalized embeddings. Sequences that would hit the
32768-token context limit are refused rather than truncated.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding third_party/Qwen3-VL-Embedding
pip install decord transformers accelerate
```

Model weights download automatically from `Qwen/Qwen3-VL-Embedding-8B` on
first run (or pass `--qwen-model` with an existing snapshot directory).

## Run

```bash
python run.py
```

Options: `--qwen-repo`, `--qwen-model`, `--tasks`, `--only-unscored`,
`--limit`.
