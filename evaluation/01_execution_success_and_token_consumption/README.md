# 01 - Execution Success (ES) and Token Consumption (TC)

Operational metrics. **ES** is the fraction of attempted tasks whose generated
output directory contains a primary video file. This is a file-based proxy;
the script does not decode every file to verify playability. **TC** is the mean provider-reported
output-token count per attempted task, counting reported reasoning tokens once.

Commands in this file run from this metric's directory. By default, inputs
and model outputs are read from `<repository>/inputs/` and
`<repository>/generated/`; override the applicable root options with absolute
paths for a custom layout. See the [evaluation workflow](../README.md).

## Setup

No dependencies beyond the Python standard library.

## Run

```bash
python run.py
```

Reads `generated/<model>/results/<task>/<sample_id>/LLM/meta.json` (provider
usage records) and checks primary video files in the corresponding result
directories. A sample counts as executed when a primary video exists under
`LLM/project/out` (`0web_`-prefixed transcodes of the same output are not
counted twice). Missing usage fields are excluded from TC rather than counted
as zero.

## Attempted-task denominator

Each discovered `<model>/results/<task>/<sample_id>/LLM/` directory counts as
one attempted task, including directories without a video. ES is the number
with a primary video divided by the number of these directories. Preserve
such a directory for every attempt, including generation/render failures.
The script does not use the full HF dataset size as a denominator or infer
missing task folders. TC excludes attempts with missing token records; its
coverage is reported separately.
