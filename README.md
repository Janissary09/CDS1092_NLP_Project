# CDS1092 NLP Project --- Detecting Unanswerable Mathematical Word Problems

**FH Graubünden (FHGR) --- NLP Research & Applied Project**

## Project Overview

This project investigates whether NLP models can reliably distinguish
between **answerable** and **unanswerable** mathematical word problems.

A problem is **ANSWERABLE** when the provided information is sufficient
to determine a unique answer. It is **UNANSWERABLE** when required
information is missing, ambiguous, contradictory, or otherwise
insufficient.

The project compares:

-   **Task-specific encoder models** fine-tuned for binary
    classification.
-   **General-purpose Large Language Models (LLMs)** evaluated in a
    zero-shot setting.

The goal is not to solve the mathematical problems themselves, but to
determine whether they contain enough information to be solved reliably.

## Research Design

Planned encoder models:

-   BERT-base
-   MathBERT
-   DeBERTa-v3-base
-   ModernBERT-base

General-purpose LLMs are evaluated separately with the same
classification task and a controlled zero-shot prompt. Exact LLM
versions may be updated before the final experiments depending on
availability.

Datasets:

-   **SUM (Synthetic Unanswerable Math)** --- main training, validation,
    and internal test dataset.
-   **UMWP (Unanswerable Math Word Problems)** --- external evaluation
    dataset for cross-dataset generalization.
-   A possible fresh/private evaluation set to reduce
    benchmark-contamination risk.

The positive class is **UNANSWERABLE**.

Primary metrics are Accuracy, Precision, Recall, F1 score, and the
confusion matrix.

**The test set is kept untouched during model development and
hyperparameter selection.**

------------------------------------------------------------------------

## SUM Dataset

Source: `lime-nlp/Synthetic_Unanswerable_Math` on Hugging Face.

The original SUM dataset contains **36,480 problem pairs**. Each pair
contains an answerable version, an unanswerable version, and a
mathematical ground-truth answer. The ground-truth answer is not used as
model input.

Each pair becomes two classification examples:

``` text
answerable_question   -> ANSWERABLE
unanswerable_question -> UNANSWERABLE
```

This produces **72,960 classification examples**.

### Leakage-Safe Split

  Split               Pairs     Examples      Share
  ------------ ------------ ------------ ----------
  Train              25,536       51,072        70%
  Validation          5,472       10,944        15%
  Test                5,472       10,944        15%
  **Total**      **36,480**   **72,960**   **100%**

Random seed: **42**

Splitting is performed at pair level before expansion. Pairs connected
by identical question text are additionally grouped transitively before
splitting, preventing identical question text from appearing in
different splits.

Final validation confirmed:

-   zero pair leakage across splits,
-   zero exact-text leakage across splits,
-   exactly 50/50 ANSWERABLE/UNANSWERABLE balance in every split.

The publisher's separate 284-row `test.parquet` is not used because it
represents a different evaluation setup.

------------------------------------------------------------------------

## Repository Structure

``` text
CDS1092_NLP_Project/
|
|-- data/
|   `-- processed/
|       `-- sum/
|           |-- train.csv
|           |-- validation.csv
|           `-- test.csv
|
|-- notebooks/
|   `-- 01_sum_dataset_analysis.ipynb
|
|-- src/
|   |-- inspect_sum.py
|   |-- analyze_sum_duplicates.py
|   |-- prepare_sum.py
|   |-- train_bert.py
|   `-- train_bert_run02.py
|
|-- results/
|   `-- bert/
|       |-- run_01/
|       `-- run_02/
|
|-- models/                 # Local/Mercury only; ignored by Git
|   `-- bert/
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Directory and File Guide

### `data/`

Contains experiment datasets. The canonical SUM splits are in:

``` text
data/processed/sum/
```

-   `train.csv` --- encoder fine-tuning.
-   `validation.csv` --- model development and hyperparameter selection.
-   `test.csv` --- final evaluation only; never use it for tuning.

These fixed files should not be regenerated or changed casually after
experiments have started.

### `notebooks/`

Contains exploratory analysis.

`01_sum_dataset_analysis.ipynb` is used to inspect SUM distributions,
labels, text lengths, and other dataset properties without mixing
exploratory work into training code.

### `src/`

Reproducible Python scripts:

-   `inspect_sum.py` --- initial SUM inspection.
-   `analyze_sum_duplicates.py` --- duplicate-text analysis; used to
    detect leakage in the first split attempt.
-   `prepare_sum.py` --- creates the final leakage-safe 70/15/15 SUM
    split.
-   `train_bert.py` --- BERT Run 01.
-   `train_bert_run02.py` --- BERT Run 02 with its own configuration and
    output directories.

Run-specific scripts are currently kept separate so completed
experiments remain easy to reproduce and inspect.

### `results/`

Contains compact experiment artifacts suitable for GitHub, for example:

``` text
results/bert/run_02/
|-- config.json
|-- training_history.csv
`-- validation_metrics.json
```

### `models/`

Contains trained model weights and checkpoints. These files are large
and intentionally excluded from GitHub via `.gitignore`.

On Mercury, a run can contain:

``` text
models/bert/run_02/best_model/
models/bert/run_02/checkpoints/
```

GitHub is **not** used as storage for model weights.

------------------------------------------------------------------------

## Requirements

Main Python dependencies:

``` text
torch
transformers
datasets
pandas
numpy
scikit-learn
accelerate
```

  Package          Purpose
  ---------------- ------------------------------------------------
  `torch`          Neural-network training and CUDA/GPU execution
  `transformers`   Tokenizers and Transformer models such as BERT
  `datasets`       Dataset loading and processing
  `pandas`         CSV/data manipulation and result storage
  `numpy`          Numerical operations
  `scikit-learn`   Classification metrics
  `accelerate`     Hugging Face/PyTorch training support

Exact versions can differ between development machines. Important final
experimental environments should be recorded.

------------------------------------------------------------------------

## Local Setup

Clone the repository:

``` bash
git clone https://github.com/Janissary09/CDS1092_NLP_Project.git
cd CDS1092_NLP_Project
```

Create or activate a suitable Python environment, then install
dependencies:

``` bash
pip install -r requirements.txt
```

On GPU systems, use the appropriate CUDA-enabled PyTorch build for that
machine rather than blindly replacing an already working GPU
installation.

Prepare SUM only when intentionally recreating the dataset:

``` bash
python src/prepare_sum.py
```

All model comparisons should use the same fixed train/validation/test
files.

------------------------------------------------------------------------

## FHGR Mercury and GPU Training

Full encoder fine-tuning is performed on FHGR Mercury because it
provides access to NVIDIA GPUs.

Project Conda environment:

``` bash
conda activate cds1092
```

Repository location on Mercury:

``` text
~/projects/CDS1092_NLP_Project
```

### Request an A100 GPU

FHGR GPU calculations must be run through Slurm. Example interactive
allocation:

``` bash
srun --partition=students --gres=gpu:a100:1 --time=02:00:00 --pty bash
```

Then:

``` bash
conda activate cds1092
cd ~/projects/CDS1092_NLP_Project
```

Verify CUDA:

``` bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print('CUDA:', torch.cuda.is_available())"
```

### Run BERT Experiments

Run 01:

``` bash
python src/train_bert.py
```

Run 02:

``` bash
python src/train_bert_run02.py
```

Training scripts store large model artifacts under `models/` and compact
experiment records under `results/`.

FHGR workstations may be restarted periodically, so code and compact
results should be pushed to GitHub regularly.

------------------------------------------------------------------------

## Current BERT Experiments

  Run      Model                   Learning Rate   Max Epochs   Best Validation F1
  -------- --------------------- --------------- ------------ --------------------
  Run 01   `bert-base-uncased`            `2e-5`            5             \~0.8431
  Run 02   `bert-base-uncased`            `3e-5`            7         **\~0.8517**

Run 02 currently has the best validation F1.

During Run 02, training loss continued to decrease while validation loss
increased in later epochs. This is being monitored as a possible sign of
overfitting.

**The SUM test set has not been used for model selection or
hyperparameter tuning.**

------------------------------------------------------------------------

## Git and GitHub Workflow

Typical workflow:

``` bash
git pull
git status
git add <files>
git commit -m "Describe the change"
git push
```

When working from multiple machines, pull recent changes before starting
new work to avoid diverging branches.

Mercury is configured for GitHub SSH authentication, so normal
`git pull` and `git push` operations do not require a GitHub password or
personal access token.

Mercury remote:

``` text
git@github.com:Janissary09/CDS1092_NLP_Project.git
```

### Stored in GitHub

-   source code,
-   notebooks,
-   fixed SUM CSV splits,
-   configuration files,
-   training histories,
-   validation metrics,
-   documentation.

### Not Stored in GitHub

-   trained model weights,
-   large checkpoints,
-   Hugging Face caches,
-   Python caches,
-   virtual/Conda environments,
-   temporary files.

Always check `.gitignore` before committing new generated artifacts.

------------------------------------------------------------------------

## Reproducibility Rules

1.  Use the same fixed SUM train, validation, and test splits for all
    encoder comparisons.
2.  Keep the test set untouched until final evaluation.
3.  Make hyperparameter decisions using training and validation data
    only.
4.  Do not overwrite completed runs; create a new run for a new
    experiment.
5.  Record configuration and validation metrics for important runs.
6.  Keep large model files outside GitHub.
7.  Document random seeds and relevant model/software versions.
8.  Do not tune SUM-trained models using the external evaluation
    dataset.

------------------------------------------------------------------------

## Evaluation Strategy

After encoder development:

1.  Select configurations using validation performance.
2.  Evaluate selected models on the untouched SUM test set.
3.  Evaluate SUM-trained encoders on UMWP without retraining.
4.  Evaluate general-purpose LLMs on the same evaluation questions with
    a controlled zero-shot prompt.
5.  Compare Accuracy, Precision, Recall, and F1.
6.  Analyze both major error directions:
    -   UNANSWERABLE classified as ANSWERABLE,
    -   ANSWERABLE classified as UNANSWERABLE.
7.  Where possible, analyze UMWP results by unanswerability category.

This provides both **in-domain performance** and **cross-dataset
generalization** results.

------------------------------------------------------------------------

## Project Status

Completed so far:

-   project repository and structure,
-   SUM download and inspection,
-   identification of leakage in the initial split,
-   leakage-safe fixed SUM split,
-   dataset-analysis notebook,
-   FHGR Mercury environment,
-   CUDA-enabled PyTorch/A100 verification,
-   BERT Run 01,
-   BERT Run 02,
-   GitHub storage of compact experiment results,
-   Mercury-to-GitHub SSH authentication.

Next steps include further BERT experimentation, final BERT
configuration selection, remaining encoder models, UMWP evaluation, and
zero-shot LLM evaluation.

------------------------------------------------------------------------

## Project Team

**FH Graubünden (FHGR) --- CDS1092 NLP Research & Applied Project**

-   Murat Arikan
-   Sinan Özdere

Supervisor: **Prof. Corsin Capol**
