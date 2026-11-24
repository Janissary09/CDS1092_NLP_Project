# CDS1092 NLP Project ? Detecting Unanswerable Mathematical Word Problems

FH Graub?nden (FHGR), NLP Research & Applied Project.

The project compares SUM-fine-tuned encoders with general-purpose zero-shot LLMs on mathematical word problem answerability. A question is ANSWERABLE when the information supports a unique answer and UNANSWERABLE when it is insufficient, ambiguous or contradictory. Models receive question text and predict one label. UNANSWERABLE is the positive class.

## Start here

- [Project progress and handover](PROJECT_PROGRESS.md) tracks completed and pending work, current Git state and the next steps.
- [Technical context](PROJECT_CONTEXT.md) describes datasets, protocols, checkpoints, results and repository layout.
- [Notebook guidelines](NOTEBOOK_GUIDELINES.md) describes the analysis and reporting conventions.

## Datasets and current results

| Dataset | Role | Status |
| --- | --- | --- |
| SUM | Encoder training, validation and internal test | Fixed pair-grouped CSV splits contain 51,072 train, 10,944 validation and 10,944 test rows. Identical text groups are kept within a single split. |
| UMWP | External evaluation | The complete official 5,200-row benchmark is prepared, including its duplicate and conflicting-label rows. |
| MATE | External evaluation | The local raw CSV has 10,000 audited rows, balanced across labels in 5,000 pairs. Cross-dataset overlap checks, processing and evaluations remain pending. |

Four validation-selected encoders have completed SUM Test and UMWP evaluations. Qwen, Apertus, Mistral, Gemma and GPT-6.1 Terra have full strict original-prompt results on both tests. Standard Phi-4 has a strict original-prompt UMWP result and separate prompt and parser sensitivity results. The Meta Llama evaluation remains pending on Mercury.

Use SUM train and validation for encoder development. Keep test benchmarks, existing run outputs and the frozen LLM prompt fixed for the primary comparison. Report prompt and parser sensitivity as separate conditions.

## Working with the repository

- `data/` contains fixed benchmarks and local raw inputs. The audited MATE replacement is at `data/raw/mate/MATE Dataset.csv` and is currently ignored by Git.
- `src/`, `configs/`, `slurm/` and `tests/` contain preparation, training, evaluation and diagnostic workflows.
- `notebooks/` contains dataset, encoder and LLM analyses. `results/` contains compact artifact files. Model weights and checkpoints live under the ignored `models/` tree when available.

For GPU execution, Mercury uses Slurm and the `cds1092` Conda environment from `~/projects/CDS1092_NLP_Project`. The documented Mercury Git remote uses SSH at `git@github.com:Janissary09/CDS1092_NLP_Project.git`. The training guide covers the shared trainer and manifests. [Mercury training guide](slurm/README_training.md)

Local dependencies are listed in `requirements.txt`. Encoder training also needs `accelerate` and a PyTorch build compatible with the CUDA environment. Notebook analysis uses Jupyter and matplotlib. The hosted OpenAI evaluator uses separate dependencies in `requirements-api.txt`. [OpenAI evaluation guide](OPENAI_EVALUATION.md)

## Team

Murat Arikan and Sinan ?zdere. Supervisor Prof. Corsin Capol.
