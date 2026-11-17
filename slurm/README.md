# Mercury overnight encoder batch

This workflow prepares ten independent FP32 experiments. It does **not** submit anything until the explicit `--submit` command. Existing experiments, results and datasets are never replaced. Source references: `src/20_train_deberta_run02.py` and the user-reported successful five-update FP32/SDPA test `src/21_test_modernbert.py`. No existing project SLURM scripts were found.

## Fixed experiment matrix

The ordered, machine-readable source is `configs/overnight_runs.json`.

| Order | Family | Run | Learning rate | Maximum epochs | Planned steps | Warmup steps |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | deberta | run_03 | 2e-5 | 7 | 11172 | 559 |
| 2 | deberta | run_04 | 4e-5 | 7 | 11172 | 559 |
| 3 | deberta | run_05 | 5e-5 | 7 | 11172 | 559 |
| 4 | deberta | run_06 | 3e-5 | 5 | 7980 | 399 |
| 5 | modernbert | run_01 | 3e-5 | 5 | 7980 | 399 |
| 6 | modernbert | run_02 | 3e-5 | 7 | 11172 | 559 |
| 7 | modernbert | run_03 | 4e-5 | 5 | 7980 | 399 |
| 8 | modernbert | run_04 | 4e-5 | 7 | 11172 | 559 |
| 9 | modernbert | run_05 | 5e-5 | 5 | 7980 | 399 |
| 10 | modernbert | run_06 | 5e-5 | 7 | 11172 | 559 |

Each job requests `students`, one `gpu:a100:1`, one task, and a maximum of `02:00:00`. CPU/RAM allocation follows site defaults; no unverified memory/CPU limits were invented. Queue access and actual runtime are not verified locally. The worst-case requested GPU time is **20 hours**, plus queue delays; “overnight” is not a promise of completion by morning. Early stopping may shorten runs. A two-hour timeout remains a failed/incomplete experiment; later jobs still proceed.

## Training protocol

Shared implementation: `src/22_train_encoder.py`. Both families use fixed SUM train (51,072) and validation (10,944), labels ANSWERABLE=0/UNANSWERABLE=1, dynamic padding/truncation 256, train/eval batch 32, accumulation 1, seed/data seed 42, explicit FP32 weights, FP16/BF16 disabled, AdamW (`adamw_torch`), decay 0.01, linear schedule and `ceil(planned_steps*0.05)` warmup. Validation/save/logging each epoch; maximum positive-class F1 selects the checkpoint; restore best; patience 2/threshold 0; save limit 2. No SUM test or UMWP input.

DeBERTa uses microsoft/deberta-v3-base and its tokenizer. ModernBERT uses answerdotai/ModernBERT-base, its tokenizer and explicit FP32 **SDPA** attention; optional reference compilation is disabled when supported, matching the compatibility test. There is no Flash Attention dependency or silent precision/batch fallback. Five successful technical updates do not guarantee full-run memory fit or stability. Dependencies must already exist in cds1092; this workflow installs nothing.

The trainer preserves prior data validation, metric definitions, provenance, two-stage overwrite checks and late exclusive directory creation. It additionally checks expected train/validation counts and exact pair/text isolation between these two splits, records dataset/script/matrix hashes, requires one visible CUDA GPU, and disables inherited Accelerate mixed precision. It does not regenerate splits or inspect held-out data. Pretrained revision is recorded when supplied by the loader; an unavailable hash remains unknown. Each queued job uses a frozen copy of the trainer and matrix, while its working directory, datasets and outputs remain the actual repository root.

## One submission command

On Mercury, from the repository root with the existing Conda installation available:

```bash
cd ~/projects/CDS1092_NLP_Project
python src/23_submit_overnight.py --submit
```

The submission helper uses only Python's standard library and SLURM CLI tools. It need not import torch. If `python` is only available inside the environment, first activate `cds1092` in the login shell. Each GPU job independently initializes Conda and activates `cds1092`. It tries the exported `CONDA_EXE`, `conda` on PATH, then common installation locations. If Mercury uses a different installation, export its actual executable before submitting; do not guess or install a replacement:

```bash
export CONDA_EXE=/actual/conda/installation/bin/conda
```

The script is submitted through `sbatch`, so executable permission on the shell file is not required. Files must have LF line endings. To inspect the matrix without submitting:

```bash
python src/23_submit_overnight.py
bash -n slurm/train_encoder.sh
```

No shell/SLURM commands in this documentation were used to launch jobs during implementation.

## Queue, results and cancellation

```bash
python src/23_submit_overnight.py --status
squeue -u "$USER"
python src/23_submit_overnight.py --cancel
```

The status helper prints the manifest, active batch-tagged job IDs, and sacct accounting when available. The cancel helper requests cancellation **only for active jobs with this batch's exact comment tag**, including a job accepted immediately before a submission-process crash. It does not delete artifacts or clear the permanent reservation. For a confirmed individual ID, ordinary `scancel JOB_ID` is also possible. No cancellation is performed automatically now.

Operational files are created on explicit submission under:

```text
slurm_batches/cds1092-overnight-v1/
  submission.json                   Job IDs, dependency chain, timestamps, source hashes
  snapshot/src/22_train_encoder.py   Submitted code snapshot
  snapshot/configs/overnight_runs.json
  logs/<family>_<run>-<jobid>.out
  logs/<family>_<run>-<jobid>.err
  status/<family>_<run>_<jobid>.tsv   Start/end UTC, exit status, elapsed seconds
```

Every run writes to independent canonical experiment directories:

```text
models/<family>/<run>/checkpoints/
models/<family>/<run>/best_model/    Best weights and matching tokenizer
results/<family>/<run>/config.json
results/<family>/<run>/training_history.csv
results/<family>/<run>/validation_metrics.json
```

No experiment directories are created by submission. They are reserved by the trainer only after successful data/model/tokenizer and TrainingArguments initialization. Existing partial or completed run directories cause refusal. A pre-training config alone is not completion evidence; inspect final reports, saved weights/tokenizer, exit status and accounting together.

## Dependency and duplicate-submission safety

1. Submission checks all ten target experiment paths and refuses if any already exists; it does not silently skip or overwrite a run.
2. An exclusive permanent batch directory protects against concurrent/repeated submission of this same batch, including after completion/cancellation/failure. Do not delete it as a routine retry mechanism.
3. All jobs are initially submitted held, with each job after the first using `afterany:<previous_job_id>`. IDs are durably recorded after each accepted submission.
4. Only after the entire chain is recorded are holds released. A training failure, timeout or cancellation does not prevent the next job's afterany dependency from clearing.
5. Partial submission/release failure is recorded and the reservation remains. Use --status/--cancel, inspect SLURM and artifacts, and make an explicit recovery plan. There is no automatic resubmission. A crash before recording a job ID can be recovered by its unique batch comment tag.

The EXIT/TERM/INT traps record normal exits, errors and the requested pre-timeout termination where the shell can run. SIGKILL, node loss, or cancellation before job start may prevent a TSV from being written; **sacct is the fallback source for terminal status**. Logs use unique job IDs and append mode to avoid truncation if a site requeues a job. A requeued job still refuses any existing experiment path rather than overwriting it. Logs, snapshots and submission state are operational records, not existing scientific results; do not commit them blindly.

## Local verification

Python compilation and offline mocked submission tests can be run without training or SLURM:

```bash
python -B -m unittest discover -s tests -p test_overnight_workflow.py
bash -n slurm/train_encoder.sh
```

Tests intercept every SLURM call, use temporary filesystem fixtures, and check chaining, hold/release order, frozen snapshots, overwrite refusal, duplicate protection and partial-submission recovery. Syntax/static checks do not establish live Mercury permissions, package loading or full training stability.
