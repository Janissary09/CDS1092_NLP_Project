# CDS1092 NLP Project: Notebook Guidelines

Last updated: **2026-11-19** (Europe/Zurich).

Start with [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md); use [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for data, metric and experiment evidence. This standard is based on the revised `notebooks/03_bert_training_analysis.ipynb`. It applies to the four encoder training-analysis notebooks and any future encoder analysis notebooks.

**Maintenance:** Update when a reusable notebook convention is agreed or a real logging/schema difference requires explicit handling. Keep experiment outcomes in the technical context and current status in the progress file. Do not rewrite conventions merely to fit an unsupported assumption. User instruction is required for notebook changes; this document does not authorize automatic edits or execution.

## 1. Principles

- Use English headings, explanations, tables and figure labels, even when discussion with the user is Turkish.
- Write concise, factual scientific prose. Separate observations, hypotheses and unresolved limitations. Do not claim significance, causation or superiority without evidence.
- Analysis notebooks read saved artifacts. No training, model loading, inference, GPU work or dataset preparation. Do not import/execute a training script to obtain metadata.
- Read actual JSON/CSV results; never manually copy available metrics into notebook code. Derive interpretation numbers from loaded records so prose cannot silently drift.
- Be self-contained and fresh-kernel Run All compatible. Include imports, root discovery, helper functions and definitions before use. Saved output is not proof of reproducibility by itself.
- Compare runs in shared tables/figures rather than six repetitive run sections. Avoid duplicated code, unnecessary complexity, decorative charts and repeated shared settings.
- Never invent missing configuration/metrics. Explain supported fallbacks and stop or clearly mark unavailable data when required evidence is absent. Do not silently substitute another run's settings.
- Keep each encoder's experimental history in its own notebook. Use the cross-model report for comparisons between selected encoders.

## 2. Standard training-analysis structure

### 1. Introduction

**Purpose:** identify the research task, model and scope in one short paragraph. State answerability classification rather than solving; ANSWERABLE=0, UNANSWERABLE=1 and positive class UNANSWERABLE; SUM training/validation/test and UMWP external evaluation. Explain that saved experiments are analyzed, no model is run.

**Data:** model identity and authoritative label/dataset protocol from configuration/scripts. No figure is needed. Avoid a literature review, unsupported model claims or a repeated project history.

### 2. Experiment Configuration

**Purpose:** make differences between available runs visible. **Data:** each run's config.json and validation_metrics.json.configuration, plus history/checkpoint metadata for best epoch.

Use one compact shared table, preferably Run, Learning Rate, Max Epochs, Best Epoch. Include batch size, warmup steps or other columns only when essential to comparability or when they differ. Explain common model/length/batch/seed/optimizer/scheduler/decay/warmup/early-stopping settings once in a short paragraph, sourced from artifacts or explicitly attributed to scripts. Check shared values before describing them as shared.

For BERT Run 01, config.json is absent: explicitly use report.configuration, not inferred defaults. If overlapping config/report fields conflict, surface the inconsistency. A change in max epochs also changes the linear schedule/warmup horizon; do not describe a longer-budget run as a simple continuation.

Avoid large repeated parameter tables, assuming identical tokenizers across models, or filling undocumented library defaults as facts.

### 3. Validation Results

**Purpose:** compare the evidence used for checkpoint/model selection. **Data:** saved best-checkpoint F1, checkpoint path/step, original epoch validation events and post-training reevaluation.

One table for all runs: Run, Best Epoch, Accuracy, Precision (UNANSWERABLE), Recall (UNANSWERABLE), F1 (UNANSWERABLE). Use original metrics corresponding to the selected checkpoint event, with saved checkpoint F1 as the primary criterion. Verify checkpoint step/F1 agreement with history; do not combine restored accuracy/precision/recall with a different original F1 without explicit labeling. Mark the selected row modestly (e.g. bold).

Show a small discrepancy table only when necessary. For Run 06, preserve epoch-5 checkpoint F1 separately from post-training restored evaluation. Its final epoch metadata does not turn reevaluation into another training epoch. Do not silently reconcile differing values or rank models using loss instead of the agreed positive-class F1.

### 4. Training Analysis

**Purpose:** explain logged trajectories. **Data:** actual epoch train-loss and training-time validation-loss/F1 records, excluding restored-model evaluation.

Include training loss, validation loss and validation F1 over **actual logged epoch coordinates**. Use one coherent color per run across figures, consistent names, readable legends, labeled axes/titles and simple grids. Shared x-axis panels are suitable for a small set of runs. If many runs overcrowd plots, use meaningful subsets/facets with explicit inclusion rather than unreadable overlays. Do not invent epoch sequences, smooth away observations or extend stopped runs.

Follow with a short observed-trends paragraph. Loss can worsen while F1 improves; report both without treating this as proof of a specific mechanism. Avoid repeated descriptions of every curve, excessive precision in prose or untested causal explanations.

### 5. Model Selection

**Purpose:** identify selected run/checkpoint and defend the agreed validation-only criterion. **Data:** original best-checkpoint validation F1 and provenance/chronology; no test metrics as selection inputs.

State the winning checkpoint/epoch and, when useful, its margin over the runner-up in percentage points. Small differences from one seed do not establish statistical significance. Report tie/ambiguity or checkpoint discrepancies explicitly.

Preserve chronology: BERT Runs 05/06 were exploratory additions following initial experiments, with their commits after the Run 02 test-results commit. Commit order is not exact execution time. Do not imply all six runs were prespecified or completed before test evaluation. Cross-model search budgets may differ even if one configuration is matched.

### 6. Test Evaluation

**Purpose:** present frozen internal and external performance after validation-based selection. **Data:** selected model's saved SUM test/UMWP metrics.json and confusion_matrix.csv (predictions only for separately justified error analysis).

One comparison table: Dataset, Number of Examples, Accuracy, Precision (UNANSWERABLE), Recall (UNANSWERABLE), F1 (UNANSWERABLE), Macro F1. Read the saved definitions, not a newly chosen calculation. Show both confusion matrices with **Actual class** rows, **Predicted class** columns, explicit ANSWERABLE/UNANSWERABLE order and count annotations. Check CSV versus JSON matrices and totals. If color scales differ due to benchmark size, treat colors as within-matrix displays, not a normalized cross-dataset measure.

Report external transfer gaps in percentage points, distinguish FP (A→U) from FN (U→A), and note benchmark conflicts. Preserve official UMWP primary results; sensitivity analyses stay separately labeled. Never tune on either test dataset or run a model inside the notebook. If test results are not yet available, mark this section pending; do not evaluate the model automatically or invent placeholders that look measured.

### 7. Conclusion

**Purpose:** answer what the results show in a few sentences. **Data:** the loaded selected validation/test evidence and any derived gap.

State selected BERT/model run, SUM performance, external performance and observed generalization gap. Include key limits only where necessary. Avoid a second full results table, repetition of all settings, exaggerated claims or new recommendations unsupported by the analysis. If evaluation is pending, conclude only about available validation evidence.

## 3. Technical conventions

### Imports, paths and artifact discovery

Use pandas and matplotlib, standard-library pathlib/json/math as needed, and IPython display/Markdown for dynamic explanatory text. Avoid new dependencies unless explicitly approved. Resolve root from cwd and its ancestors using project-specific sentinels (e.g. existing source and results directories), supporting root and notebooks/ launches. Do not hardcode a user's machine path or require an earlier notebook.

Discover model-specific run directories, sort deterministically, validate intended coverage and required filenames. In a fixed six-run BERT analysis, fail clearly if the expected run set changes so the user reviews scope. A future growing experiment series may explicitly select completed runs and report exclusions; never silently ignore broken/incomplete runs. Use clear variables such as PROJECT_ROOT, runs, selected_run, validation_table and evaluation. Small load/helper functions should serve real reuse rather than a generic framework.

### Defensive loading and provenance

Validate required keys/columns, schemas, label order and counts. Preserve missing-versus-recorded distinctions. Missing Run 01 config can fall back to its report only with a stated source. Required missing files should raise a useful path-specific error; optional/pending evaluation should be labeled pending. Check overlapping settings for conflicts rather than silently overwriting. Do not convert missing values to zero metrics.

Read metadata from the artifacts belonging to that run. Identify explicit script settings separately from runtime metadata and assumptions. Keep original source paths in explanations. The notebook can validate saved report consistency without redefining or rerunning the evaluation protocol.

### Epoch events versus final reevaluation

Trainer history is an **event log**, not automatically one row per epoch. In current BERT runs, training and validation events precede one train_runtime summary; a final evaluate() event follows it and inherits the last training epoch. The reference load_run separates at that summary boundary, validates the expected structure, then matches the saved checkpoint step to an original validation event and its F1.

Do **not** use groupby(epoch).last(), or blindly drop duplicate epochs, because this may replace the real last epoch with restored-checkpoint metrics. For another logging schema, inspect events and define an explicit validated rule; do not assume the BERT boundary exists. Reject ambiguous repeated original validation events. Keep post-training metrics separately, even if numerically identical. Do not infer best epoch from final report epoch metadata.

### Tables, figures and interpretation

Use consistent Run 01-style labels and dataset names. Format tables typically to six decimals, integer counts/epochs, learning rates in scientific notation; retain full source precision in memory. Use percentages clearly, and call differences of percentages **percentage points**. Do not mix binary positive F1 with macro-F1. Use a restrained selected-row emphasis.

Plot actual epochs/metrics; close or show figures deliberately; use readable figure sizes/layout, consistent colors and modest markers. Avoid palettes, styling libraries or animation without analytical need. Compute narrative numbers from loaded data through Markdown display where useful. Distinguish evidence, possible explanations and plans in prose.

### Verification and delivery

Check notebook JSON/schema validity and code-cell syntax. If execution is authorized and dependencies already exist, run all cells in a fresh kernel using saved artifacts only, from project root and notebooks/. Inspect for errors, hidden-state dependencies and plot readability; preserve useful executed output. Do not install packages, run training/inference or regenerate datasets to make analysis pass.

If the environment blocks Jupyter or packages are absent, report the exact limitation instead of claiming execution passed. Windows ACL/kernel-launch issues are environment issues; do not weaken security settings silently. After a change, report structure, data sources, execution status and unresolved records. Preserve unrelated existing modifications. Commit/push only on separate user instruction.

## 4. One run, multiple runs and later synthesis

For an experiment series with one completed run, keep the seven-section structure but show its recorded settings and validation trajectory without an artificial ranking, invented alternatives or tuning study. Selection means the best checkpoint within that run; explain this distinction. If final/test results are unavailable, mark them pending and retain only supported conclusions.

For multiple runs, collect them programmatically and compare shared tables and trajectories. Record failed attempts and why they are excluded, as in the DeBERTa notebook. Explain scientifically meaningful differences, actual exploratory chronology and tuning budgets. New runtime/schema differences require explicit handling.

The [final cross-model comparison](results/final_test_comparison.md) uses each model's frozen selected artifacts, common labels and metrics, and fixed test questions. It should present selected configurations, test/generalization performance, resource or uncertainty evidence if actually available, and explicit comparability limits. Link to model-specific notebooks for full trajectories rather than copying all experiment histories. Do not select a model's hyperparameters using that cross-model test table, silently mix tokenizer limits, or treat unequal search budgets as controlled equality.
