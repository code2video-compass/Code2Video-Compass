# Evaluation

Metric implementations for Code2Video-Compass, evaluated on your locally
rendered model outputs. Each folder has an entry script. There is no shared
all-metric orchestration command. The Gemini scorers provide their own request
concurrency; see the workflow below.

Metrics are listed in benchmark order.

Names follow the paper. Metric order matches the result tables: Operational
(ES, TC), Video Quality (TQ, AQ, MS), Text–Video Alignment (L-TVA, G-TVA),
Image Fidelity (PF, SF), Clip Fidelity (SA, AF, MF), Instruction Following
(IF), Holistic Video Quality (NC, VD, MQ, VE).

| # | Folder | Metric(s) | Level | Settings |
|---|--------|-----------|-------|----------|
| 01 | `01_execution_success_and_token_consumption` | Execution Success (ES), Token Consumption (TC) | objective | T2V, I2V, V2V |
| 02 | `02_technical_quality` | Technical Quality (TQ) — DOVER++ | objective | T2V, I2V, V2V |
| 03 | `03_aesthetic_quality` | Aesthetic Quality (AQ) — Aesthetic Predictor V2.5 | objective | T2V, I2V, V2V |
| 04 | `04_motion_smoothness` | Motion Smoothness (MS) — AMT interpolation error | objective | T2V, I2V, V2V |
| 05 | `05_local_tvalign` | Local TVAlign (L-TVA) — sliding-window ViCLIP-L | objective | T2V, I2V, V2V |
| 06 | `06_global_tvalign` | Global TVAlign (G-TVA) — Qwen3-VL-Embedding-8B | objective | T2V, I2V, V2V |
| 07 | `07_image_fidelity` | Perceptual Fidelity (PF), Structural Fidelity (SF) — DINOv3 + SSIM | objective | I2V |
| 08 | `08_sequence_alignment` | Sequence Alignment (SA) — DINOv3 subsequence DTW | objective | V2V |
| 09 | `09_appearance_fidelity` | Appearance Fidelity (AF) — DreamSim aligned window | objective | V2V |
| 10 | `10_motion_fidelity` | Motion Fidelity (MF) — CoTracker3 tracklets | objective | V2V |
| 11 | `11_instruction_following` | Instruction Following (IF) — Gemini QA checklist | subjective | T2V, I2V, V2V |
| 12 | `12_holistic_video_quality` | Content Coherence (NC), Visual Design (VD), Motion Quality (MQ), Viewer Engagement (VE) — Gemini judge | subjective | T2V, I2V, V2V |

Metrics 11 and 12 are two independent Gemini calls with separate prompts: 11
scores the QA checklist (IF), while 12 rates holistic quality without seeing
the checklist.

## 1. Prepare tasks and generate videos

Run evaluation after generating and rendering your own model outputs. The task
texts and annotations come from
[code2video-compass/Code2Video-Compass](https://huggingface.co/datasets/code2video-compass/Code2Video-Compass):

```python
from datasets import load_dataset

t2v = load_dataset("code2video-compass/Code2Video-Compass", "t2v", split="train")
i2v = load_dataset("code2video-compass/Code2Video-Compass", "i2v", split="train")
v2v = load_dataset("code2video-compass/Code2Video-Compass", "v2v", split="train")
```

Install `datasets` to run this loading example. The HF `train` split is the
loading name for each complete task table, not a separately held-out training
partition. The complete tables contain 718 T2V, 693 I2V, and 693 V2V tasks;
your evaluation may use a declared subset.

- **T2V:** pass the row's `prompt` to your generation model.
- **I2V:** pass `prompt` and the prescribed keyframe. When the reference is a
  video, extract the frame at `keyframe_timestamp_s` from that source. A
  standalone image has no video timestamp.
- **V2V:** pass `prompt` and the prescribed source interval
  `[keyclip_start_s, keyclip_end_s]` as a clip.

The tables contain reference URLs and timestamps, not embedded reference
media. Use the same actual reference image/clip for generation and fidelity
scoring. Missing URLs or timestamps do not authorize substituting a different
reference; record unavailable inputs and report the covered evaluation scope.
Keep the original task text, the model request, generated code,
render outcome, and available provider token usage in your generation logs.

## 2. Organize local files

The repository does not launch generation models. It reads the local files you
prepare after generation. There are two roots: `--test-root` for model outputs
and `--input-root` for benchmark inputs. Their defaults are `generated/` and
`inputs/` under the repository root, independent of the current working
directory. A relative path explicitly supplied on the command line is resolved
relative to your current working directory.

```text
Code2Video-Compass/
  generated/                         # --test-root
    <model>/
      results/<task>/<sample_id>/LLM/
        meta.json                    # provider usage record for 01, if available
        project/out/render.mp4       # one primary rendered video per task
  inputs/                            # --input-root
    <sample_id>/
      annotation.json                # Global Description + Events for 05/06
      keyframe.png                   # the actual I2V conditioning image for 07
      keyclip.mp4                    # the actual V2V conditioning clip for 08–10
      prompt/
        t2v.txt                      # the HF t2v prompt for 11/12
        i2v.txt                      # the HF i2v prompt for 11/12
        v2v.txt                      # the HF v2v prompt for 11/12
      qa.json                        # grouped checklist questions for 11
```

`<task>` is `t2v`, `i2v`, or `v2v`. `<sample_id>` must be the same ID in the
HF row, input folder, output folder, and any scope CSV. Source-paired tasks
share an ID across settings; their prompts remain distinct. Create only the
prompt and reference files required by that sample's actual settings.

**Keep a `<sample_id>/LLM/` directory for each attempted generation, including
failed attempts with no rendered video.** ES uses these directories as its
attempted-task denominator. It does not infer attempts from the full HF table.
Do not create attempt directories for tasks you did not attempt.

### Map the HF fields to input files

The text files contain the row's `prompt`, without an evaluation rubric or
model response. The semantic annotation file uses this schema; the strings
below illustrate the format and must be replaced with the actual HF fields:

```json
{
  "global_description": "The row's complete global_description.",
  "events": ["First event from the row's events list", "Second event"]
}
```

The HF `qa` field is a flat list with a `section` field on every entry. Group
these entries by `section` when writing `qa.json`; a bare copy of the flat list
is not the input schema expected by scorer 11. The accepted grouped form is:

```json
{
  "SharedChecklist": [
    {"dimension": "Content.Text", "question": "An original shared QA question", "expected_answer": "Yes"}
  ],
  "FullVideoSpecific": [
    {"question": "An original T2V-specific question", "expected_answer": "Yes"}
  ],
  "KeyframeSpecific": [
    {"question": "An original I2V-specific question", "expected_answer": "Yes"}
  ],
  "ReferenceClipSpecific": [
    {"question": "An original V2V-specific question", "expected_answer": "Yes"}
  ]
}
```

Preserve each question's text, expected answer, metadata, and within-section
order. Include only sections present in the source data. For paired tasks,
store shared questions once and combine the setting-specific sections from the
three rows; verify that the shared lists agree before deduplicating them. Scorer
11 uses `SharedChecklist` plus exactly one setting-specific section. Existing
nested `SharedChecklist` dictionaries are also accepted. Do not invent or
reword checklist questions. The quoted questions above are format placeholders,
not benchmark questions.

Only inputs required by the selected metric are needed: 01 reads generation
metadata and output paths; 02–04 read rendered videos; 05/06 also need
`annotation.json`; 07 needs the I2V image; 08–10 need the V2V clip; 11 needs the
prompt and grouped QA; 12 needs the prompt. Reference images/clips and QA are
not sent to scorer 12; scorer 11 receives the generated video, prompt, and QA.

### Primary outputs and duplicate exports

For the most consistent use of the current scripts, keep **one primary video
per attempted task** under `project/out/`. Store browser transcodes, previews,
and alternate encodes outside this directory. Supported discovery extensions
are `.mp4`, `.mov`, `.mkv`, `.webm`, and `.avi`; use a valid MP4 for Gemini.

Some objective scorers keep `0web_` exports as separate per-video rows,
whereas 01/05/06 exclude them and 11/12 prefer primary exports when present.
These are alternate encodes of the same sample, not extra benchmark tasks.
The repository does not provide a common model-level aggregation script, so
raw per-video result rows must not be averaged as independent tasks when
multiple exports are present. The single-primary-video layout avoids this
ambiguity.

## 3. Install dependencies and run metrics

Python 3.10+ is required. Install PyTorch and the metric-specific dependencies
and checkpoints listed in each metric's README. GPU inference is recommended;
CPU fallback is available for some metrics. `ffmpeg` and `ffprobe` must be on
PATH for the Gemini scorers. Per-folder `third_party/` and `checkpoints/` paths
are relative to that metric folder, not a shared directory at repository root.

Commands in this section run **from the repository root**, after the required
dependencies and checkpoints are installed. The default input/output roots are
those shown above. The folders' README files contain their setup instructions.

```bash
python evaluation/01_execution_success_and_token_consumption/run.py
python evaluation/02_technical_quality/run.py
python evaluation/03_aesthetic_quality/run.py
python evaluation/04_motion_smoothness/run.py
python evaluation/05_local_tvalign/run.py
python evaluation/06_global_tvalign/run.py
python evaluation/07_image_fidelity/run.py
python evaluation/08_sequence_alignment/run.py
python evaluation/09_appearance_fidelity/run.py
python evaluation/10_motion_fidelity/run.py
```

07 evaluates I2V only; 08–10 evaluate V2V only. **Run 08 before 09 and 10.**
Their default alignment input is
`evaluation/08_sequence_alignment/results.json`. If you change 08's output
location, pass that file to both later metrics using `--alignment`.
The alignment file carries the generated-video paths; regenerate it when
moving or replacing the videos.

For custom local roots, pass absolute paths to each command. Create the parent
directory of a custom `--output` file before running, for example:

```bash
python evaluation/05_local_tvalign/run.py \
  --test-root /absolute/path/to/generated \
  --input-root /absolute/path/to/inputs \
  --output /absolute/path/to/local_tvalign_results.json
```

Scorers 02–10 support `--limit 1` for an initial small scoring run. Use a
separate output file for such a check, and inspect the results and errors
before launching the full run. 01 processes every discovered attempt; it has
no `--limit` or `--only-unscored` option. Consult `python <entry-script> --help`
for the options supported by that specific scorer.

### Gemini IF and holistic quality

Set `GEMINI_API_KEY`, or create the metric-local `.env` using its
`.env.example`. These scripts read local MP4 files and upload them through
Gemini's native `generateContent` API. They do not use a webpage, serve the
videos publicly, or invoke the generation model again. Their local job
runners make individual concurrent API requests; they are not Vertex batch
submission/collection commands.

First inspect the scope without making API calls:

```bash
python evaluation/11_instruction_following/run.py --dry-run --limit-samples 5
python evaluation/12_holistic_video_quality/run.py --dry-run --limit-samples 5
```

A dry run reads files, hashes videos, validates prompts/checklists, and writes
local planning records; it does not decode every video or verify API access.
After checking the plan, run the desired scope:

```bash
python evaluation/11_instruction_following/run.py \
  --limit-samples 5 --concurrency 5 --fps 2
python evaluation/12_holistic_video_quality/run.py \
  --limit-samples 5 --concurrency 5 --fps 2
```

The default requested judge name is `gemini-3.8-flash`. Your API endpoint must
provide that model name. Use `--model` to select another available model and
report the change when publishing results. `GEMINI_API_BASE` changes the HTTPS
endpoint root. Authentication uses the `x-goog-api-key` header; this adapter
does not automatically use shell proxy variables.

The full generated video is uploaded as inline MP4 bytes. `--fps 2` sends a
`videoMetadata.fps` sampling request to the provider; it does not downsample
the local source video or prove the provider's effective sampling cadence.
Videos above 40,000,000 bytes receive an upload-only compressed derivative
with full duration, resolution, and frame rate, targeting 35,000,000 bytes.
The source is preserved, and derivative details are stored in the record.
The default encoder is `libx264`; `--video-encoder h264_nvenc` requires a
working NVIDIA encoder. Compression can reduce visual detail; inspect the
saved compression diagnostics when interpreting readability scores.

The current `--media-resolution` option is recorded in local configuration
but is not sent as a provider resolution control. It does not guarantee a
particular video input resolution at the API. Requested and verified settings
are distinguished in the saved records.

## 4. Choose an evaluation scope

Without a scope option, each scorer discovers local outputs under its test
root; the release does not automatically load the authors' internal evaluation
CSV files. You can control the experiment by preparing the desired output
folders. Scorers 11/12 also accept explicit IDs, model filters, and scope CSVs:

```csv
sample_id,task,generated_model
C2VC-0001,t2v,my_model
C2VC-0001,i2v,my_model
C2VC-0002,v2v,another_model
```

```bash
python evaluation/11_instruction_following/run.py \
  --sample-list /absolute/path/to/scope.csv --dry-run
python evaluation/11_instruction_following/run.py \
  --sample-list /absolute/path/to/scope.csv --concurrency 5
```

Use the same scope arguments with scorer 12. Omit `--models` to select all
discovered model folders; `--models all` is not a special all-models selector
in these release scripts. Repeat `--sample-list` to combine CSVs. A CSV needs
`sample_id`; `task` selects an exact setting. If the task column or a task cell is absent, all requested settings for that ID are
selected. An optional `generated_model` column restricts rows to those model
folders; all its cells must be nonempty when that column is present.

| Option in 11/12 | Meaning |
|---|---|
| `--samples ID1 ID2` | Restrict to the listed sample IDs |
| `--models model_a model_b` | Restrict to the listed model-folder names |
| `--tasks t2v i2v v2v` | Restrict to the listed settings |
| `--limit-samples N` | Select at most N unique sample IDs, including their selected settings and models |
| `--limit N` | Submit at most N new, uncached video jobs across the selected scope |
| `--concurrency N` | Run at most N simultaneous API scoring jobs |
| `--dry-run` | Save the local plan without calling the API |

CSV ID selection follows first occurrence in the combined files; explicit ID
selection follows `--samples` order. With neither, `--limit-samples` uses
lexicographically sorted discovered IDs within the selected model/task scope.
It is not random sampling. Five IDs across two models and three settings can
produce up to 30 video jobs; `--limit-samples 5` does not mean five videos per
model or five total API calls. Missing videos do not create Gemini calls.

## 5. Read outputs, resume, and summarize

01 writes model-by-setting ES/TC rows to its `results.json`. Scorers 02–10
write per-video results, protocol information, and errors to `results.json`
beside each entry script, or the path passed using `--output`.

For 02–10, resume using the same command and `--only-unscored`. Their skip
lookup uses generated-video relative paths; it does not establish that the
video bytes or metric configuration are unchanged. Use a fresh output file
when replacing videos, changing weights or settings, or conducting a new
experiment. Without `--only-unscored`, an output file is not a resume cache.
Objective metric results are written at the end of a run; do not assume every
in-progress sample was saved before an interruption.

11 writes to `evaluation/11_instruction_following/checklist_scores/` by
default; 12 writes to
`evaluation/12_holistic_video_quality/holistic_scores/`.

- `scores.csv`: latest per-model/sample/setting results, including status.
  IF is a Yes fraction in `[0, 1]`; NC/VD/MQ/VE are integer ratings in `[1, 5]`
  before aggregation.
- `records/*.json`: durable per-job records with assessments, hashes, usage,
  and request settings; successfully completed jobs are cached.
- `raw/*.json`: raw returned API responses when available.
- `scores.jsonl`: saved job records; use `job_key` and status when inspecting
  history rather than treating every historical row as a distinct sample.
- Scorer 11 also writes `checklist_answers.csv`.
- `runs/<run_id>/`: per-run plans, summary, and input issues.

The Gemini runners save each completed job and refresh CSV outputs during
the run. Repeat the same command to reuse compatible successful records;
failed requests can be attempted again. Cache keys include video contents,
prompt(s), checklist where applicable, and scoring configuration. A changed
video/prompt/configuration may therefore create a new paid request. For an
independent repeated assessment, choose a fresh `--out-dir` for each repeat.
Do not run multiple writers against the same output directory concurrently.

### ES, token consumption, and missing outputs

For each model and setting, let `A` be the number of discovered
`results/<task>/<sample_id>/LLM/` directories and `R` the number containing
at least one primary video file under `project/out/`:

- **ES = R / A.** A failed attempt with an `LLM/` directory but no primary
  video is included in `A` and contributes zero to execution success. An
  attempt with no such directory cannot be inferred by this scorer.
- Execution detection is file-based; 01 does not decode every video to verify
  playability. Keep complete, valid renders as primary outputs.
- **TC** averages available provider-reported output/completion token counts
  over the attempted tasks with valid usage records, counting reported
  reasoning tokens once. Missing usage is excluded rather than imputed as
  zero; `tasks_with_token_record` reports the coverage.

Scorers 02–12 compute video-level metrics for discovered outputs. They do not
automatically insert metric zeros for missing renders or build a combined
leaderboard. When reporting full-cohort means, use the same attempted-task
scope, count missing renders as zero for applicable quality/alignment/IF
metrics, and keep valid judge-assigned zeros. Report the attempted and
scored counts. A decoding, input, checkpoint, or API error for an existing
video is an uncomputed metric, not evidence of a zero-quality model output;
resolve it or report the missing measurement explicitly.

For rendered-only means, use successful metric measurements and state that
render failures are excluded. A nonzero-only analysis additionally excludes
legitimate zero scores and is a different statistic. Do not label it
rendered-only. Combine settings by actual applicable sample counts rather
than an unweighted mean of T2V/I2V/V2V means. PF/SF are I2V-only; SA/AF/MF
are V2V-only, so other settings must not be inserted as zeros for them.

## 6. Protocol and reproducibility notes

- IF (11) and holistic quality (12) use **separate Gemini requests and
  prompts**. Temperature 0 alone does not guarantee identical API responses
  across runs or provider model updates.
- DINOv3 scorers 07/08 load both configuration and weights from
  `timm/vit_base_patch16_dinov3.lvd1689m` at revision
  `c6a5fb7d12bbd3cf3b0079253141c3332aaed7da` using the Hub identifier's
  `@revision` suffix. Their setup instructions specify `timm==1.0.27`.
- Record the dataset revision, selected sample/model/task scope, generation
  settings and provider model version, renderer environment, metric package
  versions, checkpoint versions/hashes, judge prompts, and actual returned
  judge model version for the experiment you report. Not every dependency
  or upstream repository in this release is version-locked; record your
  installed environment rather than assuming an unpinned install is identical.
- Keep a single primary video per sample/task while scoring. If handling
  alternate exports yourself, collapse them to one measurement per
  `(model, task, sample_id)` before averaging and declare the selection policy.

## Citing the underlying tools

DOVER++, Aesthetic Predictor V2.5, VBench (ViCLIP-L, AMT-S), Qwen3-VL-Embedding,
DINOv3, DreamSim, and CoTracker3 are used through their official implementations;
see each folder's README for repository links.
