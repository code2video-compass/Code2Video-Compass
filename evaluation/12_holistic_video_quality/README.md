# 12 - Holistic Video Quality (NC, VD, MQ, VE)

Holistic 1-5 ratings judged by Gemini. Each generated video is submitted with
its generation prompt through a native generateContent request (inline MP4,
temperature 0, 2 FPS sampling). The judge returns four ratings with reasons
grounded in video evidence: NC (Content Coherence), VD (Visual Design),
MQ (Motion Quality), and VE (Viewer Engagement), each an integer 1-5.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

```bash
pip install requests
```

`ffmpeg` and `ffprobe` must be on PATH. Get a Gemini API key at
<https://aistudio.google.com/apikey> and set `GEMINI_API_KEY` in the
environment, or create a `.env` file beside this script (see `.env.example`):

```bash
cp .env.example .env   # then paste your key
```

`GEMINI_API_BASE` optionally overrides the endpoint root (default:
<https://generativelanguage.googleapis.com>). The judge model defaults to
`gemini-3.8-flash` (the paper's judge); `--model` selects another model.

## Run

```bash
# plan only, no API calls
python run.py --dry-run

# full run over all discovered videos
python run.py

# restrict to specific samples / models / a CSV scope
python run.py --samples sample_id_a sample_id_b
python run.py --models model_dirname
python run.py --sample-list scope.csv --limit 50
```

A `--sample-list` CSV needs a `sample_id` column; a `task` column selects exact
sample/task pairs, and an optional `generated_model` column restricts each row
to that model.

## Outputs (under `--out-dir`, default `holistic_scores/`)

- `scores.csv` — one row per video: NC/VD/MQ/VE scores and reasons.
- `scores.jsonl`, `records/*.json` — full per-job records (prompt hashes,
  request settings, usage).
- `raw/*.json` — every raw API response, keyed by a content-addressed job key
  (video bytes + prompts + request settings); re-runs reuse compatible
  successful records; unsuccessful requests may be attempted again.

## Scope and request settings

`--limit-samples N` limits unique sample IDs across the selected models and
settings; `--limit N` limits new video jobs across that scope. Neither means
N samples per model and setting. Use `--dry-run` to inspect the plan. A full
description of selection order, media compression, cache/resume behavior,
and output files is in the [evaluation workflow](../README.md).

The API receives inline MP4 bytes and a `videoMetadata.fps` sampling request.
The current `--media-resolution` option is recorded in local configuration
but is not sent as a provider resolution control. Reference images/clips
are not attached to the judge; it evaluates the submitted generated video
using the text inputs described above.
