# CDS1092 NLP Project: Technical Context

Last updated: **2026-11-24** (Europe/Zurich).

Read [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) **first** for current status and collaboration instructions. This is the technical reference; [NOTEBOOK_GUIDELINES.md](NOTEBOOK_GUIDELINES.md) defines analysis standards.

**Maintenance:** Update when verified code, data, methods or results change; preserve useful evidence and cite relative paths/functions. Put mutable session status in PROJECT_PROGRESS.md. Distinguish artifact-verified, user-reported, planned and unknown information. Recheck current local/Mercury state before action. No training, inference or model evaluation was executed for this update.

## 1. Research overview

The FH Graubünden CDS1092 project predicts **answerability of mathematical word problems**, not their numerical solutions. ANSWERABLE means sufficient information for a unique answer; UNANSWERABLE includes missing, ambiguous, contradictory or insufficient information. Mapping: **ANSWERABLE=0, UNANSWERABLE=1**, positive class UNANSWERABLE. Only question text is input (`README.md`; `src/05_train_bert_run02.py::LABEL_TO_ID`, `compute_metrics`).

The comparison uses the same fixed test questions for SUM-fine-tuned encoders and general-purpose zero-shot LLMs. SUM provides training, validation and held-out internal test data. UMWP and MATE are external evaluation datasets. The four selected encoders have completed SUM Test and UMWP evaluation. Qwen, Apertus, Mistral, Gemma and GPT-6.1 Terra have full strict frozen-prompt results on both datasets. Standard Phi-4 has an original-prompt result on UMWP and separate sensitivity runs. All four selected encoders have saved MATE evaluations. The local LLM MATE results in `4c155fe` cover Apertus, Gemma, Mistral, Qwen and standard Phi-4. The Meta Llama 3.1-8B-Instruct Mercury smoke test and all three benchmark runs remain pending.

The sibling LaTeX draft frames FF1 as fine-tuned encoder versus zero-shot classification, FF2 as transfer to external data, and FF3 as error analysis and implications for training and prompting. The comparison measures complete approaches with differing scale, pretraining, and access to task-specific training data. Formal directional hypotheses and a significance protocol are absent from the repository.

Gemma replaces the previously planned Gemini evaluation. The selected roster covers the models above, standard Phi-4 and Meta Llama. Team: Murat Arikan, Sinan ?zdere. Supervisor: Prof. Corsin Capol.

## 2. Repository and sources

Root: `C:/Users/Murat Arikan/dev/school/03_hs_26/NLP/CDS1092_NLP_Project`, separate Git repository. Inspection HEAD: `4c155fe`. Local origin is `https://github.com/Janissary09/CDS1092_NLP_Project.git`; README describes a Mercury SSH remote.

```text
CDS1092_NLP_Project/
  data/
    raw/umwp/StandardDataset.jsonl
    raw/mate/MATE Dataset.csv
    processed/sum/{train,validation,test}.csv
    processed/umwp/test.csv
    processed/mate/{test,excluded_pairs}.csv
  notebooks/
    01_sum_dataset_analysis.ipynb
    02_umwp_dataset_analysis.ipynb
    03_bert_training_analysis.ipynb
    04_mathbert_training_analysis.ipynb
    05_deberta_training_analysis.ipynb
    06_modernbert_training_analysis.ipynb
    07_... through 12_...             LLM analyses listed in section 15
    13_mate_dataset_analysis.ipynb
  src/                               Data, training and evaluation entry points
  configs/                           Encoder experiment configurations
  slurm/                             Batch script and training guide
  tests/                               Training, evaluation and CUDA diagnostics
  results/
    {bert,mathbert,deberta,modernbert}/run_*/
    llm/                             Local zero-shot evaluations
    api/openai/                      Hosted GPT-6.1 Terra evaluations
    encoder_training_comparison.md
    final_test_comparison.md
```

Successful runs have a training history CSV and validation report JSON. All have a separate configuration JSON except BERT Run 01, whose configuration is embedded in its report. Selected runs also have `sum_test/` and `umwp/` evaluation artifacts. Model weights are absent from this local checkout.

| Source | Purpose / functions |
| --- | --- |
| `src/01_inspect_sum.py` | `main`: publisher Parquets loaded separately because schemas differ; schema/examples, no experiment files. |
| `src/02_analyze_sum_duplicates.py` | `load_occurrences`, `has_conflict`, `main`: read-only exact duplicate/conflict/overlap inspection. |
| `src/03_prepare_sum.py` | `build_groups`, `assign_groups`, `expand`, `validate`, `main`: grouped fixed splits. |
| `src/04_train_bert_run01.py` | BERT baseline; legacy output-path exception below. |
| `src/05_train_bert_run02.py` | Selected BERT configuration, primary pipeline reference. |
| `src/06_train_bert_run03.py`, `src/07_train_bert_run04.py` | Seven-epoch LR variants. |
| `src/10_train_bert_run05.py`, `src/11_train_bert_run06.py` | Exploratory higher-LR five/seven-epoch variants. |
| All training scripts | `CONFIG`, `load_split`, `compute_metrics`, `main`; train/validation only. |
| `src/08_prepare_umwp.py` | `require`, `question_statistics`, `validate_text`, `main`: JSONL to unchanged benchmark CSV. |
| `src/09_evaluate_bert.py` | `load_dataset`, `calculate_metrics`, `main`: frozen Run 02, CLI --dataset sum or umwp. |
| `src/12_train_mathbert_run01.py` | BERT Run 02 pipeline, MathBERT tokenizer, provenance, overwrite guard. |
| `src/13_train_mathbert_run02.py` through `src/16_train_mathbert_run05.py`, `src/17_evaluate_mathbert.py` | MathBERT follow-up runs and frozen evaluation. |
| `src/18_train_deberta_run01.py` through `src/21_test_modernbert.py` | Initial DeBERTa training, diagnostics and retry, and ModernBERT compatibility checks. |
| `src/22_train_encoder.py`, `src/23_submit_training.py`, `slurm/train_encoder.sh` | Shared encoder training and SLURM submission workflow. |
| `src/24_evaluate_encoders.py` | Frozen DeBERTa and ModernBERT evaluation. |
| `results/bert/run_*/config.json` | Pre-training settings/original Mercury dataset paths; only Run 02–06. |
| `results/bert/run_*/training_history.csv` | Epoch train loss/LR/gradient norm, validation, training summary, restored-model evaluation. |
| `results/bert/run_*/validation_metrics.json` | Best checkpoint path/F1, restored metrics, settings/versions. |

Separate sibling documentation repository `../dokumentation_01/` contains README.md, .gitignore, LICENSE, `term-paper/documentation.tex`, `title.tex`, `packages_and_configuration.tex`, `library.bib`, `documentation.pdf`. Evidence uses accessible TeX/Bib; PDF was not visually verified.

## 3. SUM dataset

Dataset ID `lime-nlp/Synthetic_Unanswerable_Math`; `src/03_prepare_sum.py::SOURCE_URL` is `https://huggingface.co/datasets/lime-nlp/Synthetic_Unanswerable_Math/resolve/main/synthetic_unanswerable_math.parquet`. It contains 36,480 pairs with answerable_question, unanswerable_question, ground_truth. Mathematical answers are excluded from inputs and final CSV. Raw SUM is not stored locally; source revision/hash unpinned. Separate publisher test.parquet is inspected by script 01 but **excluded** from preparation; README calls it a distinct 284-row setup, not independently reloaded here.

`src/03_prepare_sum.py` loads only main Parquet via generic datasets.load_dataset, validates count/fields, and then:

1. `build_groups`: original index becomes pair_id; union-find across both question fields joins all exact-text-linked pairs transitively.
2. `assign_groups`: seed-42 shuffle, stable descending group-size sort, assign to largest remaining target deficit. Group integrity takes priority; this greedy heuristic cannot guarantee exact targets for arbitrary future source data.
3. `expand`: only after splitting, two examples per pair (one per class), seed-42 shuffle per split. No trim/lowercase/normalization/deletion/deduplication.
4. `validate`: original texts, complete source retention, two rows/one label each per pair, whole groups, balanced classes, disjoint pair/text sets. CSVs are written after checks, but existing CSVs can be overwritten. Do not regenerate fixed experiment files casually.

Columns: `pair_id,text,label`. Prior read-only full-CSV checks established:

| Path | Pairs | Examples | ANSWERABLE | UNANSWERABLE | Unique texts | Repeated distinct texts | Conflicting distinct texts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/processed/sum/train.csv` | 25536 | 51072 | 25536 | 25536 | 50019 | 733 | 1 |
| `data/processed/sum/validation.csv` | 5472 | 10944 | 5472 | 5472 | 10943 | 1 | 1 |
| `data/processed/sum/test.csv` | 5472 | 10944 | 5472 | 5472 | 10944 | 0 | 0 |

All 72,960 rows retained; every pair occurs twice; cross-split pair and exact-text overlaps are zero for every split pair. Within-split duplicates/conflicts remain. README describes initial split leakage, but original split files and old numerical leakage logs are unavailable. Exact-text isolation does not eliminate semantic/template/near-duplicate or pretraining contamination. No dedicated archived SUM–UMWP overlap check exists.

Only train/validation may influence training, hyperparameters, early stopping, checkpoint selection and future runs. Never tune thresholds/prompts using SUM test or UMWP. Preserve fixed data across models. Notebook 01 has inspected SUM test examples/labels/lengths, so “test data was never inspected” is too strong. Whether sequence-length choice was influenced by test statistics is unknown.

## 4. UMWP dataset

Raw `data/raw/umwp/StandardDataset.jsonl`; processed `data/processed/umwp/test.csv`. Associated publication: Sun et al., *Benchmarking Hallucination in Large Language Models Based on Unanswerable Math Word Problem*, LREC-COLING 2024 (`../dokumentation_01/term-paper/library.bib::Sun2024`). Acquisition URL/revision/commit is unknown.

`src/08_prepare_umwp.py::main` reads UTF-8-sig JSONL; validates 5,200 rows, unique non-null IDs, nonempty string questions, boolean answerable. Raw columns: id/question/answer/answerable/category/relevant_ids/source. Output: `id,text,label,category,source`; question renamed unchanged, True→ANSWERABLE, False→UNANSWERABLE. Metadata/order preserved; answer/relevant_ids excluded. `question_statistics` checks duplicates/conflicts; `validate_text` checks integrity. No row removal, corrections, dedup or train/validation split.

2,600 rows/class, 5,190 unique texts, 10 repeated distinct texts involving 20 rows, 5 conflicting-label groups. Official primary result stays all 5,200 rows; secondary sensitivity analysis must be separate.

| Source | Total | Per class |
| --- | ---: | ---: |
| GSM8K | 3400 | 1700 |
| SVAMP | 1000 | 500 |
| MultiArith | 600 | 300 |
| ASDiv | 200 | 100 |

Answerable category is null for all 2,600 rows. Unanswerable counts: code 1=840, 2=1263, 3=275, 4=103, 5=119. CSV codes are 1.0…5.0, null empty. Semantic code mapping is unknown; TeX mentions missing/ambiguous key information, unrealistic conditions, unreferenced objects and missing questions without mapping codes.

Notebook 02 records 2,598 null answers; two unanswerable answers are non-null integers; answerable answers are lists. relevant_ids is null for every answerable row, list for every unanswerable row. These label-revealing metadata patterns must not become inputs. Existing pipeline uses only text.

## 5. Common training methodology

Reference `src/05_train_bert_run02.py::load_split`, `compute_metrics`, `main`; BERT Run 03–06 differ by identity/paths, LR or budget.

- load_split accepts only train/validation; requires pair_id/text/label; rejects empty/missing data, non-string text, unknown labels. Only text/numeric labels enter model datasets.
- WORLD_SIZE must equal 1. `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")` preserves an existing SLURM mask; more than one visible GPU is rejected. Ordinary python, not torchrun. CPU fallback remains possible.
- set_seed(42) precedes model loading. BERT uses BertTokenizerFast from bert-base-uncased, truncation at 256. Dataset.from_pandas(preserve_index=False).map(batched=True,remove_columns=["text"],load_from_cache_file=False) tokenizes in memory; DataCollatorWithPadding dynamically pads.
- AutoModelForSequenceClassification(num_labels=2, explicit label2id/id2label); full encoder/head fine-tuning, no freezing.
- Train/eval batch 32, accumulation 1, effective one-GPU batch 32; adamw_torch, linear scheduler, decay .01, seed/data_seed 42, epoch evaluation/save/logging, report_to none.
- Planned steps ceil(train_count/batch_size)*max_epochs; 1,596 steps/epoch. Warmup ceil(planned_steps*.05): 5 epochs→7,980/399 planned/warmup; 7→11,172/559. Budget changes also change the full LR schedule.
- load_best_model_at_end=True; metric_for_best_model="f1"; greater_is_better=True; save_total_limit=2; early stopping patience 2/threshold 0. Binary UNANSWERABLE validation F1 selects checkpoints, not loss. CONFIG's descriptive validation_f1 differs from Trainer key f1.
- compute_metrics: argmax; accuracy; binary precision/recall/F1 with average=binary, pos_label=1, zero_division=0.
- After train, restore best, reevaluate validation, save weights and matching tokenizer, history CSV and validation JSON. Checkpoint F1 and restored evaluation remain separate. BERT Runs 02–06 and MathBERT write config before training. BERT Run 01 has only the configuration embedded in its validation report.

BERT scripts lack overwrite guards/resume CLI. Run 01 uses models/bert/best_model, models/bert/checkpoints, results/bert, although archived compact results are results/bert/run_01/. Relocation history is unknown; unchanged script does not write to that archive directory.

BERT reports record torch **2.11.0+cu128**, transformers **5.22.2**, cuda:0. Training Python, GPU model, driver, other library versions, model/tokenizer hashes and raw SUM revision are unknown. Notebook Python 3.13.7 does not establish Mercury training Python. No explicit training deterministic-algorithms/fp16/bf16 settings; do not invent defaults. Evaluation is more deterministic. Single-seed results establish neither significance nor robustness. MathBERT configs record broader runtime metadata. The [DeBERTa and ModernBERT training comparison](results/encoder_training_comparison.md) documents their recorded FP32 settings, dataset hash coverage and follow-up schedules.

## 6. Completed BERT experiments and chronology

Sources: results/bert/run_01 through run_06 history CSV/validation JSON and available config. Exact metrics, epochs, checkpoint paths and runtimes remain in section 12.

| Run | Script | LR | Max epochs | Best epoch | Best-checkpoint F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | `src/04_train_bert_run01.py` | 2e-5 | 5 | 5 | 0.8431372549019608 |
| 02 | `src/05_train_bert_run02.py` | 3e-5 | 7 | 7 | 0.8517332845513583 |
| 03 | `src/06_train_bert_run03.py` | 2e-5 | 7 | 7 | 0.8497940503432494 |
| 04 | `src/07_train_bert_run04.py` | 4e-5 | 7 | 7 | 0.8513302034428795 |
| 05 | `src/10_train_bert_run05.py` | 5e-5 | 5 | 5 | 0.8458781362007168 |
| 06 | `src/11_train_bert_run06.py` | 5e-5 | 7 | 5 | 0.8509703405346027 |

All bert-base-uncased. **Run 02 selected by validation F1**; Run 04 accuracy is higher but is not the selection criterion. F1 lead is ~0.0403 percentage points, no significance established. Run 03/04 change LR versus Run 02; Run 05 changes LR/budget; Run 06 LR. Five/seven-epoch comparisons also change warmup/decay.

Runs 05/06 were exploratory additions, not predeclared with all six. Verified commit order: 89c9fde (Run 02 final evaluation), bf04ddb (Run 05/06 code), f8d5e2d (their results), 235b47a (MathBERT code/original context). Commit order is not exact training chronology. Earlier notebook states Run 02 selected before test, consistent with frozen evaluator, but independent timestamped selection evidence is limited. Do not imply all six preceded first test evaluation; later decisions must not exploit known test outcomes.

**Run 06 discrepancy:** results/bert/run_06/validation_metrics.json records checkpoint-7980/F1 **0.8509703405346027**, matching original epoch 5 (loss .4683046042919159, accuracy .8512426900584795, precision .8525311812179017, recall .8494152046783626). Epochs 6/7 do not improve. Patience and max-budget boundaries coincide; exact stopping trigger cannot be distinguished.

Post-training restored evaluation F1 is **0.8506903172716467**, accuracy .8507858187134503, precision .851235132662397, recall .8501461988304093, loss .4714003801345825. Its epoch=7 is final Trainer state, not checkpoint epoch. Cause unknown; retain both. Later authorized investigation should inspect checkpoint restoration, serialization and numerical determinism on validation, not change results or rerun training first.

Run 02 train loss decreases .523006→.103833, validation loss minimum .374826 at epoch 3 rises to .629921 at epoch 7 while F1 improves (history CSV). Report this divergence; it does not prove a causal overfitting/calibration mechanism.

## 7. Frozen encoder evaluation

src/09_evaluate_bert.py hardcodes models/bert/run_02/best_model and results/bert/run_02/config.json. load_dataset reads only the requested processed test CSV, keep_default_na=False, validating schema/text/labels/size and UMWP balance/IDs. main checks run/mapping; local_files_only model/tokenizer; saved length 256/batch 32; seed 42, cuDNN benchmark off/determinism on, deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch. CUDA if available else CPU.

Inference uses eval/no_grad, original order, text only, padding/truncation, argmax and softmax. calculate_metrics defines accuracy, UNANSWERABLE binary precision/recall/F1, two-label macro-F1, correct/incorrect. No adaptation/training/threshold tuning.

All four selected checkpoints have saved evaluations on the fixed SUM Test (10,944 examples) and full UMWP (5,200 examples). The saved runs use maximum length 256, evaluation batch size 32 and argmax. BERT and MathBERT loaded `best_model/` exports. DeBERTa and ModernBERT loaded their recorded `checkpoint-11172/` directories.

ModernBERT leads SUM Test by UNANSWERABLE F1 (87.18%), and DeBERTa leads UMWP (68.78%). Every encoder scores lower on UMWP. The official UMWP results retain all 10 repeated texts, including five groups with conflicting labels. See the [final test comparison](results/final_test_comparison.md) for all scores, confusion matrices, transfer gaps and source links.

Prediction CSVs retain the original text and labels, predicted label, probabilities and correctness, along with SUM pair IDs or UMWP IDs and category/source metadata. UMWP duplicate analyses are saved with the evaluations. Category/source and qualitative error analysis remain open.

## 8. MathBERT and later encoder experiments

User-selected **tbs17/MathBERT-custom**, with mathematical vocabulary and its own tokenizer instead of original-vocabulary MathBERT. Domain-specific tokenization is the design rationale, not evidence of superiority to tbs17/MathBERT. No controlled variant comparison/stronger selection rationale is archived. Pretraining and tokenizer change together; vocabulary effects are not isolated.

`src/12_train_mathbert_run01.py` exists in local commit **235b47a**. Push and Mercury pull are user-reported, not live-verified here. Previously inspected public metadata indicated bert/BertForMaskedLM, 512 positions, hidden size 768, 12 layers/heads, vocab 30,522, uncased mathematical WordPiece and tokenizer.json. References for later checking: https://huggingface.co/tbs17/MathBERT-custom and its /blob/main/config.json and /tree/main pages. That was metadata-only inspection during implementation; tokenizer_config content was unavailable then. No assistant weight loading or fresh network verification occurred now. The user reports successful Mercury model/tokenizer loading; job logs are outside the local checkout.

main loads AutoConfig, AutoTokenizer(use_fast=True) from exact ID, AutoModelForSequenceClassification(num_labels=2) with the same resolved revision. Runtime checks: BERT type, position limit, all parameters trainable, matching tokenizer/embedding size, padding ID and in-range special IDs. MLM weights require new classification head and any absent pooler initialization. Full fine-tuning remains; no extra dependency or remote-code requirement introduced.

Configuration matches BERT Run 02: **3e-5, max 7 epochs, train/eval batch 32, accumulation 1, length 256, seed/data_seed 42, adamw_torch, linear, decay .01, warmup .05 (11,172/559 steps), epoch eval/save/log, positive F1 selection, restore best, patience 2/threshold 0, save limit 2**. Only fixed SUM train/validation.

Revision requested main; AutoConfig._commit_hash, when available, pins subsequent tokenizer/weights and is recorded. Otherwise null/unavailable and main fallback. Provenance: ID, requested/resolved revision/source, model/tokenizer classes, vocab size, Python and available torch/transformers/datasets/numpy/pandas/scikit-learn/accelerate/tokenizers/huggingface-hub versions (missing metadata null). The [Run 01 config](results/mathbert/run_01/config.json) records Python 3.11.18 and a requested revision of `main`, with no resolved commit hash.

Outputs models/mathbert/run_01/checkpoints/, models/mathbert/run_01/best_model/ (weights **and tokenizer**), results/mathbert/run_01/{config.json,training_history.csv,validation_metrics.json}. Before loading, either existing run directory (including partial/empty) causes failure; directories and JSON use exclusive creation. An early failure leaves directories blocking retry. No automatic resume/cleanup; obtain a decision rather than deleting/restarting.

The initial A100/CUDA launch and first-epoch F1 of approximately 0.67 were reported on 2026-11-17. Saved artifacts now establish completion of MathBERT Runs 01–05. Run 04 was selected at epoch 5 (`checkpoint-7980`, validation F1 78.16%) and evaluated by `src/17_evaluate_mathbert.py` on both test datasets. The [MathBERT notebook](notebooks/04_mathbert_training_analysis.ipynb) documents the full run history.

DeBERTa-v3-base completed Runs 02–08. Run 01 is a failed attempt with non-finite loss and zero validation F1. ModernBERT-base completed Runs 01–08. Their selected runs were DeBERTa Run 05 and ModernBERT Run 06, both at epoch 7 (`checkpoint-11172`). For both models, Run 07 tested a higher learning rate and Run 08 a longer planned schedule. Both Run 08 experiments stopped at epoch 7. Neither follow-up displaced the selected checkpoints. See the [training comparison](results/encoder_training_comparison.md) for configurations, trajectories and validation metrics.

## 9. Environment and execution

README: Mercury root ~/projects/CDS1092_NLP_Project, Conda cds1092, SSH git@github.com:Janissary09/CDS1092_NLP_Project.git. BERT absolute root /home/arikanmurat@edu.local/projects/CDS1092_NLP_Project. A100/CUDA use is documented/user-reported; BERT reports alone prove cuda:0. Job IDs, current queue, GPU memory type, login hostname, CPU/RAM requests, modules, Conda creation/full export unknown.

The original BERT and MathBERT training scripts document the interactive Mercury workflow. The current shared batch workflow is documented in [slurm/README_training.md](slurm/README_training.md) and uses `slurm/train_encoder.sh`, `src/23_submit_training.py` and the experiment manifests in `configs/`. GPU work runs through SLURM with its assigned CUDA mask.

requirements.txt lists unpinned torch/transformers/datasets/pandas/numpy/scikit-learn. accelerate is required but omitted. Analysis needs matplotlib, Jupyter/IPython; prior verification used existing local fhgr with nbformat/nbclient/ipykernel, no installs. MathBERT adds standard-library platform/importlib.metadata only. Preserve working CUDA torch; document version or precision/batch changes.

README compact-result flow: Mercury git status, explicit reviewed git add paths, commit/push, local pull. **No commit/push authorized now.** .gitignore excludes /models/, checkpoints/weights, HF caches, environments/temp files, while allowing fixed SUM/raw+processed UMWP. Git is not weight storage. Backup retention/location and real transfer history unknown. A future approved transfer may use scp -r <user@host>:<remote-project>/models/<model>/<run>/best_model <local-destination> or institutional tools with confirmed details. Compact results do not prove model availability; verify retention for all four selected models if reruns or backup transfers are needed.

## 10. Notebooks and verification history

Follow NOTEBOOK_GUIDELINES.md. This update read notebook JSON/outputs, without rerunning or loading tokenizers.

- notebooks/01_sum_dataset_analysis.ipynb: CSV/schema/label checks, whitespace words (length_statistics), full BERT tokens, 128/256/512 truncation, shortest/longest examples/histograms. Eight executed cells; fhgr/Python 3.13.7 metadata. No training/data writes; tokenizer may download. Saved train/validation/test token mean 69.43/70.61/70.95, max 2186/1029/1047; above 256: 582/138/119 (1.14%/1.26%/1.09%). HF unauthenticated/symlink/>512 warnings are recorded; untruncated measurement without inference is intentional. Test inspection qualifies unseen-test claims.
- notebooks/02_umwp_dataset_analysis.ipynb: raw fields/nulls/labels/categories/sources, duplicates/conflicts, answer/relevant_ids types, check PASS/WARNING, official-preservation decision. Twelve executed cells without errors. Future-tense proposed-schema text is historical; CSV now exists.
- notebooks/03_bert_training_analysis.ipynb: revised English seven-section analysis; programmatic six-run loading/shared tables, three loss/F1 plots, validation ranking, Run 06 discrepancy, saved JSON/CSV test metrics and actual/predicted confusion matrices. read_json/load_run helpers; ancestor root detection; explicit Run 01 fallback. load_run uses single train_runtime summary boundary, matches checkpoint step to original epoch, checks artifact consistency.

- [04_mathbert_training_analysis.ipynb](notebooks/04_mathbert_training_analysis.ipynb) covers Runs 01–05, Run 04 selection and saved tests.
- [05_deberta_training_analysis.ipynb](notebooks/05_deberta_training_analysis.ipynb) covers the failed Run 01, completed Runs 02–08, Run 05 selection and saved tests.
- [06_modernbert_training_analysis.ipynb](notebooks/06_modernbert_training_analysis.ipynb) covers Runs 01–08, Run 06 selection and saved tests.

The BERT revision fixed hidden state (undefined pd/runs/RUN_NAMES), hardcoded test metrics and groupby(epoch).last() replacing Run 06's original final epoch. These are **resolved historical defects**. Saved eight code cells have counts 1–8/no errors. In the previous notebook task fresh-kernel Run All passed from root and notebooks/ and figures were inspected. Initial sandbox launch hit Windows kernel connection-file ACL restrictions; approved outside-sandbox execution succeeded with a nonfatal event-loop warning. No model operation. Revised notebook remains locally modified, pending user review/commit.

## 11. Limitations, contradictions and pending evidence

- TeX says official SUM test preserved; actual preparation builds 70/15/15 from main Parquet, excluding publisher test. Align report later. Provisional LLM names are not final IDs.
- Run 06 restore difference, one seed, unequal search budgets, missing immutable BERT/data revisions and joint tokenizer/domain changes limit claims. Exact grouping cannot prove semantic/pretraining isolation; UMWP conflicts/unknown categories require caution.
- Remote retention of the selected checkpoints and tokenizers has not been verified.

- Comparative and qualitative analysis remains open. Keep frozen-prompt results and Phi-4 sensitivity conditions distinct in the final report.

## 12. Preserved numerical evidence

Original BERT JSON/CSV values follow. Best epoch events (12.2) and restored-model evaluations (12.3) remain separate. Every epoch table names its source; no evaluation was rerun.

### 12.1 Run configurations and best epochs

| Run / script | LR | Maximum / completed epoch | Warmup / planned steps | Best epoch / step | Best checkpoint F1 |
| --- | --- | --- | --- | --- | --- |
| run_01 / `src/04_train_bert_run01.py` | 2e-05 | 5 / 5.0 | 399 / 7980 | 5.0 / 7980 | 0.8431372549019608 |
| run_02 / `src/05_train_bert_run02.py` | 3e-05 | 7 / 7.0 | 559 / 11172 | 7.0 / 11172 | 0.8517332845513583 |
| run_03 / `src/06_train_bert_run03.py` | 2e-05 | 7 / 7.0 | 559 / 11172 | 7.0 / 11172 | 0.8497940503432494 |
| run_04 / `src/07_train_bert_run04.py` | 4e-05 | 7 / 7.0 | 559 / 11172 | 7.0 / 11172 | 0.8513302034428795 |
| run_05 / `src/10_train_bert_run05.py` | 5e-05 | 5 / 5.0 | 399 / 7980 | 5.0 / 7980 | 0.8458781362007168 |
| run_06 / `src/11_train_bert_run06.py` | 5e-05 | 7 / 7.0 | 559 / 11172 | 5.0 / 7980 | 0.8509703405346027 |

### 12.2 Original epoch event with maximum validation F1

| Source: run training_history.csv | Loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| `results/bert/run_01/training_history.csv` | 0.4437739849090576 | 0.8450292397660819 | 0.8535580524344569 | 0.83296783625731 | 0.8431372549019608 |
| `results/bert/run_02/training_history.csv` | 0.629921019077301 | 0.8518823099415205 | 0.8525911005310383 | 0.8508771929824561 | 0.8517332845513583 |
| `results/bert/run_03/training_history.csv` | 0.5452165007591248 | 0.8500548245614035 | 0.8512745277828718 | 0.8483187134502924 | 0.8497940503432494 |
| `results/bert/run_04/training_history.csv` | 0.6973326206207275 | 0.8524305555555556 | 0.8577258393618995 | 0.8450292397660819 | 0.8513302034428795 |
| `results/bert/run_05/training_history.csv` | 0.5088651180267334 | 0.846765350877193 | 0.8508042151968941 | 0.8410087719298246 | 0.8458781362007168 |
| `results/bert/run_06/training_history.csv` | 0.4683046042919159 | 0.8512426900584795 | 0.8525311812179017 | 0.8494152046783626 | 0.8509703405346027 |

### 12.3 Restored-model validation metrics from JSON

| Source: run validation_metrics.json | Loss | Accuracy | Precision | Recall | F1 | epoch metadata |
| --- | --- | --- | --- | --- | --- | --- |
| `results/bert/run_01/validation_metrics.json` | 0.4437739849090576 | 0.8450292397660819 | 0.8535580524344569 | 0.83296783625731 | 0.8431372549019608 | 5.0 |
| `results/bert/run_02/validation_metrics.json` | 0.629921019077301 | 0.8518823099415205 | 0.8525911005310383 | 0.8508771929824561 | 0.8517332845513583 | 7.0 |
| `results/bert/run_03/validation_metrics.json` | 0.5452165007591248 | 0.8500548245614035 | 0.8512745277828718 | 0.8483187134502924 | 0.8497940503432494 | 7.0 |
| `results/bert/run_04/validation_metrics.json` | 0.6973326206207275 | 0.8524305555555556 | 0.8577258393618995 | 0.8450292397660819 | 0.8513302034428795 | 7.0 |
| `results/bert/run_05/validation_metrics.json` | 0.5088651180267334 | 0.846765350877193 | 0.8508042151968941 | 0.8410087719298246 | 0.8458781362007168 | 5.0 |
| `results/bert/run_06/validation_metrics.json` | 0.4714003801345825 | 0.8507858187134503 | 0.851235132662397 | 0.8501461988304093 | 0.8506903172716467 | 7.0 |

### 12.4 Checkpoint paths and training runtimes

Absolute paths are artifact provenance; checkpoints are not present locally. train_runtime is seconds, excluding queue/setup/download time.

| Run | best_checkpoint | train_runtime (s) |
| --- | --- | --- |
| run_01 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/checkpoints/checkpoint-7980` | 2116.1642 |
| run_02 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_02/checkpoints/checkpoint-11172` | 2967.4375 |
| run_03 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_03/checkpoints/checkpoint-11172` | 2932.4643 |
| run_04 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_04/checkpoints/checkpoint-11172` | 2933.9415 |
| run_05 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_05/checkpoints/checkpoint-7980` | 2129.8538 |
| run_06 | `/home/arikanmurat@edu.local/projects/CDS1092_NLP_Project/models/bert/run_06/checkpoints/checkpoint-7980` | 2982.7804 |

### 12.5 Epoch history

Tables retain original training-time epoch evaluations. The restored-model event is separate in section 12.3; training loss is the logged epoch loss.

#### `results/bert/run_01/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5257077324659305 | 0.43079593777656555 | 0.8107638888888888 | 0.838407960199005 | 0.7699195906432749 | 0.8027055349147375 |
| 2.0 | 3192 | 0.39099182281876566 | 0.3751548230648041 | 0.8300438596491229 | 0.864553895841744 | 0.7827119883040936 | 0.8215998465375024 |
| 3.0 | 4788 | 0.3166973525121397 | 0.3837580978870392 | 0.8390899122807017 | 0.8406462272810722 | 0.8368055555555556 | 0.8387214946423666 |
| 4.0 | 6384 | 0.2578537828641428 | 0.3965548276901245 | 0.8463084795321637 | 0.8637236084452975 | 0.8223684210526315 | 0.8425388504025464 |
| 5.0 | 7980 | 0.20904292438860825 | 0.4437739849090576 | 0.8450292397660819 | 0.8535580524344569 | 0.83296783625731 | 0.8431372549019608 |

#### `results/bert/run_02/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5230060854651276 | 0.4434451162815094 | 0.8078399122807017 | 0.8156267566048342 | 0.7955043859649122 | 0.8054399111851235 |
| 2.0 | 3192 | 0.3879495491658835 | 0.380142480134964 | 0.8332419590643275 | 0.8552503409312293 | 0.8022660818713451 | 0.8279113625648279 |
| 3.0 | 4788 | 0.3062116388689008 | 0.37482649087905884 | 0.8422880116959064 | 0.8463387573964497 | 0.8364400584795322 | 0.841360294117647 |
| 4.0 | 6384 | 0.2369722256385593 | 0.3859826326370239 | 0.8490497076023392 | 0.8598342125094197 | 0.8340643274853801 | 0.8467532467532467 |
| 5.0 | 7980 | 0.1790014472521636 | 0.4656360447406769 | 0.8516081871345029 | 0.8601647323099962 | 0.8397295321637427 | 0.8498243018309599 |
| 6.0 | 9576 | 0.13814738758823328 | 0.5620740652084351 | 0.8516081871345029 | 0.8561643835616438 | 0.8452119883040936 | 0.8506529336030899 |
| 7.0 | 11172 | 0.10383264642012746 | 0.629921019077301 | 0.8518823099415205 | 0.8525911005310383 | 0.8508771929824561 | 0.8517332845513583 |

#### `results/bert/run_03/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5392013205621475 | 0.4315396547317505 | 0.8028143274853801 | 0.8231279251170047 | 0.7713815789473685 | 0.7964150943396227 |
| 2.0 | 3192 | 0.39917516110833723 | 0.3732753396034241 | 0.8309576023391813 | 0.8664508296236342 | 0.7825292397660819 | 0.8223545227578261 |
| 3.0 | 4788 | 0.32327415829613093 | 0.4045828580856323 | 0.8292214912280702 | 0.8106570098292809 | 0.8591008771929824 | 0.834176204418419 |
| 4.0 | 6384 | 0.2623334420952283 | 0.38507401943206787 | 0.8462171052631579 | 0.8721272834413671 | 0.8114035087719298 | 0.8406702641295086 |
| 5.0 | 7980 | 0.20910923702077458 | 0.4428161382675171 | 0.8469480994152047 | 0.8462520517964618 | 0.847953216374269 | 0.8471017800091283 |
| 6.0 | 9576 | 0.16914140132435582 | 0.49591901898384094 | 0.8497807017543859 | 0.8573562359970127 | 0.8391812865497076 | 0.8481714074621352 |
| 7.0 | 11172 | 0.1407810201620996 | 0.5452165007591248 | 0.8500548245614035 | 0.8512745277828718 | 0.8483187134502924 | 0.8497940503432494 |

#### `results/bert/run_04/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5229045131750274 | 0.43882375955581665 | 0.8135051169590644 | 0.8462159434914228 | 0.766264619883041 | 0.8042581758895176 |
| 2.0 | 3192 | 0.3910188603221922 | 0.37640559673309326 | 0.829952485380117 | 0.8709677419354839 | 0.774671052631579 | 0.8200019344230584 |
| 3.0 | 4788 | 0.3064057199578536 | 0.38364866375923157 | 0.8401864035087719 | 0.8412465627864345 | 0.8386330409356725 | 0.8399377688295049 |
| 4.0 | 6384 | 0.23366963474971608 | 0.3653043210506439 | 0.8516081871345029 | 0.8843387934478626 | 0.8090277777777778 | 0.8450085894254629 |
| 5.0 | 7980 | 0.16915801772497652 | 0.46096131205558777 | 0.8497807017543859 | 0.858830146231721 | 0.837171052631579 | 0.8478622987229317 |
| 6.0 | 9576 | 0.12470941376267818 | 0.569610595703125 | 0.8496893274853801 | 0.852329221137912 | 0.8459429824561403 | 0.8491240942859763 |
| 7.0 | 11172 | 0.08785945729803023 | 0.6973326206207275 | 0.8524305555555556 | 0.8577258393618995 | 0.8450292397660819 | 0.8513302034428795 |

#### `results/bert/run_05/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5162239935165062 | 0.4268994629383087 | 0.8097587719298246 | 0.8471937730438345 | 0.7558479532163743 | 0.7989182924473633 |
| 2.0 | 3192 | 0.38510713123139884 | 0.3701725900173187 | 0.8350694444444444 | 0.8689877238881063 | 0.7891081871345029 | 0.827123838712767 |
| 3.0 | 4788 | 0.29667996284656956 | 0.3823418915271759 | 0.8376279239766082 | 0.8391775289149991 | 0.835343567251462 | 0.8372561589889184 |
| 4.0 | 6384 | 0.21546532575946703 | 0.4108377993106842 | 0.8460343567251462 | 0.8561218732367877 | 0.8318713450292398 | 0.8438224117156363 |
| 5.0 | 7980 | 0.1432509840580455 | 0.5088651180267334 | 0.846765350877193 | 0.8508042151968941 | 0.8410087719298246 | 0.8458781362007168 |

#### `results/bert/run_06/training_history.csv`

| Epoch | Step | Train loss | Validation loss | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 1596 | 0.5267701113134399 | 0.4188584089279175 | 0.8114948830409356 | 0.8581634797226309 | 0.7463450292397661 | 0.798357931775975 |
| 2.0 | 3192 | 0.39814215256157975 | 0.38544726371765137 | 0.8278508771929824 | 0.8530106257378984 | 0.7922149122807017 | 0.8214894826606026 |
| 3.0 | 4788 | 0.31397298941934915 | 0.3909871578216553 | 0.8384502923976608 | 0.8414454277286135 | 0.8340643274853801 | 0.8377386196769456 |
| 4.0 | 6384 | 0.24151481303356045 | 0.38359466195106506 | 0.8494152046783626 | 0.878913991280222 | 0.8104897660818714 | 0.8433162198136528 |
| 5.0 | 7980 | 0.17342627197877505 | 0.4683046042919159 | 0.8512426900584795 | 0.8525311812179017 | 0.8494152046783626 | 0.8509703405346027 |
| 6.0 | 9576 | 0.12431351702314869 | 0.575013279914856 | 0.8489583333333334 | 0.8500458295142072 | 0.8474049707602339 | 0.8487233458405784 |
| 7.0 | 11172 | 0.08669894799253994 | 0.7083661556243896 | 0.8496893274853801 | 0.8494977168949772 | 0.8499634502923976 | 0.8497305197771079 |

### 12.6 Fixed dataset SHA-256 fingerprints

Hashes were computed from local file bytes on 2026-11-17 and verified on 2026-11-24. Different CSV line endings may change byte hashes without changing logical records. Preserve the fixed files.

| File | SHA-256 |
| --- | --- |
| `data/processed/sum/test.csv` | `7fda226cb432d6cda00dce0a6ac7a0e9aebcf4a8f874b703d2e3428c788549c2` |
| `data/processed/sum/train.csv` | `c91dd31945eb857a1528ce69f495f36582689271a5a7f5129a1f2e05f0482ed8` |
| `data/processed/sum/validation.csv` | `5478039d9c39ef5f658b532ec09ffdde41bdd06bddb63a17651f2e7e927339d0` |
| `data/processed/umwp/test.csv` | `c03cfaeff114eba47454dc09dc925117c38e8a321288c3e5e359c6cf28b568c0` |
| `data/raw/umwp/StandardDataset.jsonl` | `13adeb41e16d68cea2fcdde55b0f608b604cde4d3f0a0951fa5b89f1919189ba` |

## 13. MATE dataset audit

**MATE** is the new external benchmark. The corrected raw file is `data/raw/mate/MATE Dataset.csv`. It replaces the two legacy MATE files under `data/raw/umwp/3. dataset/`. The official UMWP benchmark remains a separate dataset.

On 2026-11-24, a read-only UTF-8 CSV audit found the exact header `pair_id,text,label,source,category`, with 10,000 data rows and no malformed or ragged records. All IDs, texts, labels and sources are populated. There are 5,000 ANSWERABLE and 5,000 UNANSWERABLE rows. Pair IDs are consecutive integers 1?5,000. Every pair has exactly two distinct question texts, one row per class, and a shared source. There are zero duplicate full rows, zero duplicate question texts and zero conflicting-label text groups, both for exact text and for text normalized using Unicode NFKC, case folding and whitespace collapsing. These checks establish structural integrity. Semantic pairing and label correctness were not adjudicated.

| Source | Rows per class | Total |
| --- | ---: | ---: |
| AQuA-RAT | 2,705 | 5,410 |
| DMath | 1,030 | 2,060 |
| GSM8K | 865 | 1,730 |
| ASDiv | 257 | 514 |
| SVAMP | 78 | 156 |
| MultiArith | 65 | 130 |

All 5,000 ANSWERABLE categories are empty. UNANSWERABLE codes 1?5 have 1,584, 2,426, 545, 198 and 247 rows respectively. Keep the original source values and category codes, with blanks preserved. The semantic mapping of codes and the dataset's release provenance remain unverified. SHA-256 of the audited raw bytes is `e2cfd8dd2cf9cb2761f68d148765e455f0cdc2cf37bfc61224e4de892e494e52`.

### Pair-level exclusion and prepared benchmark

[13_mate_dataset_analysis.ipynb](notebooks/13_mate_dataset_analysis.ipynb) verifies the recorded raw hash before loading records. It compares literal strings and keys normalized using Unicode NFKC, case folding and whitespace collapsing. Raw MATE shares 179 exact GSM8K questions and 28 additional normalized-only MultiArith questions with UMWP. The 207 questions all have matching ANSWERABLE labels. Both members of every affected pair were excluded to reduce evaluation-set overlap while preserving class balance and complete pairs. This removed 207 pairs and 414 rows (4.14%). No raw MATE match was found in any SUM split.

The processed set contains 9,586 rows in 4,793 complete pairs, with 4,793 rows per class. Assertions verified one row per label per pair, consistent pair sources, required values present, zero exact or normalized duplicate texts or label conflicts, and zero exact and normalized overlap with SUM train, validation, test and UMWP test. The notebook preserves raw row order, original pair IDs, text, categorical labels, sources and category values, including empty categories for ANSWERABLE rows. After exclusion it assigns a unique sequential integer `id` from 1 to 9,586 as the first column in retained file order. The original `pair_id` continues to identify the pair. It adds `label_id` (ANSWERABLE=0, UNANSWERABLE=1) while keeping the standard string `label` column.

| Generated artifact | Contents | SHA-256 |
| --- | --- | --- |
| `data/processed/mate/test.csv` | `id,pair_id,text,label,source,category,label_id` | `377f7a28a6b2a5236ce24eeebc889c99ec38ff92c52c9b0a888d71f6542da5b8` |
| `data/processed/mate/excluded_pairs.csv` | `pair_id,exact_match_rows,normalized_only_match_rows,reason` | `478e7b5ef5bea2d105b07552d25131d000c69dc3660f34db1cbd5317aea86e19` |

The processed test SHA-256 after CRLF-to-LF normalization is `377f7a28a6b2a5236ce24eeebc889c99ec38ff92c52c9b0a888d71f6542da5b8`. It matches the file-byte hash because the notebook exports LF line endings. Both UTF-8 CSV exports were reloaded and verified field by field in saved row order. The raw hash was rechecked after export. Git allowlists the processed test CSV and exclusion ledger explicitly. Both are tracked. Text-level overlap checks cannot establish semantic independence or exclude pretraining exposure.

### Evaluation integration

Use `--dataset mate` with the encoder, local LLM and OpenAI API entry points. The shared loader in `src/25_evaluate_qwen.py` pins the LF-normalized hash and exact column order, and checks the row count, class balance, sequential `id`, complete `pair_id` groups and agreement between string and numeric labels. The three encoder scripts reuse that MATE validator, retain the MATE metadata in predictions, and preflight a separate `mate/` result directory beneath each selected run. The encoder `both` option still means SUM and UMWP.

Local LLM and hosted API outputs inherit the shared loader and preserve `id` as the question identifier and `pair_id` as metadata. The SQLite response key remains the zero-based row index. API resume binds that index to the fixed dataset hash, row order and run settings through the checkpoint manifest. Full and limited runs use separate paths. All seven local LLM Slurm wrappers accept MATE, including the existing standard Phi-4 prompt variants and smoke overrides. Question text remains the only model input.

Offline checks passed for all 12 Python CLI entry points, all dataset loaders, MATE metadata and encoder overwrite guards, API checkpoint isolation and resume, and Bash syntax and option dispatch for all seven wrappers. The 59 existing offline evaluator tests passed.

### Completed MATE encoder evaluations on Mercury

The three runs below completed on 2026-11-24. All 13 result files are available locally in commit `596029d`. Slurm job IDs and failure details come from the Mercury run history supplied by the user.

| Encoder and artifact directory | Slurm job | Accuracy | Precision | Recall | F1 UNANSWERABLE | Macro-F1 | Correct / incorrect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [BERT Run 02](results/bert/run_02/mate/) | 1883 | 0.7402461924 | 0.7305305305 | 0.7613185896 | 0.7456068655 | 0.7401307985 | 7,096 / 2,490 |
| [DeBERTa Run 05](results/deberta/run_05/mate/) | 1885 | 0.7599624452 | 0.7500000000 | 0.7798873357 | 0.7646517337 | 0.7598671122 | 7,285 / 2,301 |
| [ModernBERT Run 06](results/modernbert/run_06/mate/) | 1886 | 0.7351345712 | 0.7429926693 | 0.7189651575 | 0.7307814654 | 0.7350653041 | 7,047 / 2,539 |

Each prediction file contains 9,586 records in benchmark order with unique sequential question IDs and matching text, pair IDs, labels and metadata. Local verification normalized CRLF to LF in question text for the Windows checkout. The confusion matrices reconstructed from predictions matched both saved CSV and JSON. Accuracy, UNANSWERABLE precision/recall/F1, macro-F1, and correct/incorrect counts were recalculated. All runs used CUDA, maximum sequence length 256, evaluation batch size 32, and argmax. BERT loaded the Run 02 `best_model/` export. DeBERTa and ModernBERT loaded their validation-selected `checkpoint-11172/` directories directly.

The initial jobs 1879 through 1882 failed before inference because `sbatch --wrap` used `sh`, which did not support `source`. Explicit Bash resolved that launch issue. MathBERT job 1884 then failed before model loading because `evaluate_dataset()` still allowed only SUM and UMWP, although the CLI already accepted MATE. The local fix checks `DATASET_OUTPUTS` in both paths. The offline regression exercises the real CLI, configuration and dataset loading, and output checks up to the model-runtime import for all three datasets. It also verifies that the default and `both` still select SUM and UMWP. Three new regression tests and 59 existing evaluator tests passed. The fix was committed in `6377856`. The later MathBERT Run 04 MATE result is available under `results/mathbert/run_04/mate/` from `039fb6f`.


## 14. Zero-shot LLM protocol and completed runs

The original frozen prompt is defined in `src/25_evaluate_qwen.py` and stored in run metadata.

```text
Determine whether the following math problem can be answered using the information provided.

Respond with exactly one label:
ANSWERABLE or UNANSWERABLE.

Do not solve the problem or provide an explanation.

Math problem:
{question}

Label:
```

Each run uses the fixed SUM test or full official UMWP CSV in saved row order. The local evaluators use one user message serialized by the model's native chat template, zero-shot, BF16 on an A100 80 GB PCIe, no quantization, greedy decoding, one beam and at most 16 new tokens. Qwen disables thinking explicitly. Apertus uses its default thinking-off mode. Mistral uses native Tekken serialization, Gemma its processor template and standard Phi-4 native ChatML. Per-run JSON records model ID, prompt, dataset hash and runtime settings. Byte hashes can differ across checkouts with different line endings.

Surrounding whitespace is trimmed. Only the two exact uppercase labels are valid. INVALID counts as incorrect for full-dataset accuracy, occupies its own confusion-matrix column and contributes a missed positive when the true label is UNANSWERABLE. Precision, recall and F1 use UNANSWERABLE as positive, with zero for undefined denominators. The API shares the parser.

### Original frozen prompt, strict scoring

Scores are percentages. The archived metrics are in `results/llm/<slug>/<dataset>/run_01/metrics.json` and `results/api/openai/gpt-6.1-terra/<dataset>/run_01/metrics.json`. Here `sum` is the 10,944-row SUM Test and `umwp` is the full 5,200-row external benchmark.

| Model | Dataset | Accuracy | Precision | Recall | F1 | INVALID rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | SUM Test | 88.10 | 90.87 | 84.72 | 87.69 | 0 |
| Qwen3.6-27B | UMWP | 93.19 | 94.11 | 92.15 | 93.12 | 0 |
| Apertus v1.5-8B | SUM Test | 55.51 | 89.96 | 12.45 | 21.87 | 21 |
| Apertus v1.5-8B | UMWP | 55.69 | 94.63 | 12.19 | 21.60 | 20 |
| Mistral Small 3.2 24B | SUM Test | 74.57 | 86.13 | 58.57 | 69.73 | 0 |
| Mistral Small 3.2 24B | UMWP | 80.96 | 94.48 | 65.77 | 77.55 | 0 |
| Gemma 3 27B IT | SUM Test | 73.88 | 70.18 | 83.02 | 76.07 | 0 |
| Gemma 3 27B IT | UMWP | 86.85 | 93.59 | 79.12 | 85.74 | 0 |
| Standard Phi-4 14B | UMWP | 48.50 | 0.00 | 0.00 | 0.00 | 1943 |
| GPT-6.1 Terra (API) | SUM Test | 86.04 | 90.93 | 80.06 | 85.15 | 0 |
| GPT-6.1 Terra (API) | UMWP | 92.42 | 95.17 | 89.38 | 92.19 | 0 |

### Standard Phi-4 prompt sensitivity and offline postprocessing

Standard `microsoft/phi-4` (14B) is distinct from Phi-4-reasoning-plus. UMWP `run_01` used the original prompt. `src/32_evaluate_phi4_prompt_sensitivity.py` removes the period after `UNANSWERABLE` in the label instruction for `run_02`, retaining the strict parser. The standard evaluator and SLURM wrapper expose a failed-smoke override for documented sensitivity work. An original-prompt SUM result is absent.

| Dataset | Condition | Accuracy | Precision | Recall | F1 | INVALID rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SUM Test | Modified prompt, strict | 67.01 | 84.67 | 50.27 | 63.09 | 1683 |
| SUM Test | Modified prompt, offline extraction | 79.30 | 83.01 | 73.68 | 78.07 | 0 |
| UMWP | Modified prompt, strict | 81.69 | 96.52 | 66.12 | 78.48 | 115 |
| UMWP | Modified prompt, offline extraction | 83.73 | 96.26 | 70.19 | 81.18 | 0 |

Strict results are archived under `results/llm/phi-4/<dataset>/run_02/`. The paired `run_02_cleaned/predictions.csv` files apply researcher-initiated label extraction to the stored generations. Comparison by row confirms that previously invalid responses were replaced by the exact label on the first nonblank line (1,683 SUM and 115 UMWP). All true labels, texts, generated token IDs and cap flags are unchanged. The cleaned scores above are computed from those CSVs. Cite the preserved strict CSVs for verbatim model output. Treat the prompt change and the parser change as separate sensitivity conditions, outside the original frozen-prompt comparison.

### GPT-6.1 Terra API

`src/33_evaluate_openai.py` uses the original prompt with OpenAI Chat Completions, SDK 2.54.0, temperature 0, reasoning effort `none`, `max_completion_tokens=16`, `store=false` and `service_tier=default` (Standard). Tokenization and serving are provider-managed, and generated token IDs are unavailable. Both full runs record all rows received, with the returned ID `gpt-6.1-terra`, all finish reasons `stop`, and zero INVALID outputs.

| Saved run | Completed requests | Input tokens | Output tokens | Total tokens | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| SUM Test | 10,944 | 1,241,573 | 59,538 | 1,301,111 | 1.539263 |
| UMWP | 5,200 | 506,868 | 28,442 | 535,310 | 0.649078 |
| UMWP first-10 smoke | 10 | 857 | 50 | 907 | 0.001107 |

The locally untracked smoke artifacts are under `results/api/openai/gpt-6.1-terra/umwp/limit_10/run_01/`. All ten gold labels were ANSWERABLE, all ten predictions were correct, and positive-class precision/recall/F1 were 0 by the zero-denominator convention. This is an API pipeline check. Cost figures come from archived usage and pricing estimates. [API setup and resume guide](OPENAI_EVALUATION.md)

### Blocked or limited model attempts

- **Meta Llama 3.3 70B Instruct:** Mercury CUDA diagnostic job 1894 confirmed both NVIDIA A100 80 GB GPUs were individually accessible. All eight CPU-to-GPU transfers passed and all eight GPU-to-GPU transfers failed. Silent data corruption occurred in both directions with FP32 and BF16. The hardware, driver or system configuration cause remains unconfirmed. This makes the existing unquantized two-GPU 70B setup unreliable. The original evaluator and its forward, activation, boundary and transfer diagnostics are preserved in `src/29_evaluate_llama_70b.py` and `slurm/evaluate_llama_70b.sh`. The standalone CUDA transfer diagnostic and downloaded 70B checkpoint were untouched.
- **Phi-4-reasoning-plus:** `src/30_evaluate_phi4.py` targets `microsoft/Phi-4-reasoning-plus` with a native template that inserts a reasoning system message. Reasoning text can consume the 16-token budget before a final label, and a completed reasoning trace still fails strict whole-response matching. The diagnostic tests budgets 16/64/256/512 on two probes and can extract a standalone final label after a completed reasoning block. Those diagnostic conditions differ from the benchmark protocol. There is no full saved result, so use the standard Phi-4 artifacts for the completed Microsoft family evaluation.


### Meta Llama 3.1-8B-Instruct on Mercury

The replacement is `meta-llama/Llama-3.1-8B-Instruct`, reported downloaded on Mercury at `models/Llama-3.1-8B-Instruct` with four safetensors shards (about 15 GiB). The primary wrapper uses Conda `cds1092` and requests one A100. The evaluator checks for exactly one visible BF16-capable A100 80 GB and verifies all unquantized BF16 weights and model buffers are on `cuda:0`. CPU offloading is disabled.

The evaluator uses the original frozen user prompt, native chat template, greedy decoding (temperature 0, one beam), and a 16-token cap. It removes only the terminal EOS when decoding and uses the shared strict label parser, INVALID policy and metrics. The shared loader checks the fixed SUM, UMWP and MATE CSV hashes. Each `results/llm/llama-3.1-8b-instruct/<dataset>/run_01/` directory uses the existing SQLite checkpoint, model and source fingerprints, predictions, metrics and confusion-matrix conventions. Resume with the same dataset and run ID.

After the integration is available on Mercury, run the smoke test from the project root.

```bash
cd ~/projects/CDS1092_NLP_Project
sbatch --time=00:15:00 slurm/evaluate_llama.sh smoke
```

The smoke test loads the single-GPU checkpoint once, verifies placement, and runs two synthetic prompts with benchmark settings. It requires both expected strict labels without hitting the cap and logs decoded text, token IDs, cap flags and peak memory. The smoke test and benchmark runs are pending. Offline checks covered syntax and CLI routing, fixed dataset hashes, isolated result paths, mocked placement and decoding, strict smoke pass/fail behavior and shared checkpoint regressions.

## 15. Repository inventory

### Evaluation entry points

| Script | Purpose | Batch wrapper |
| --- | --- | --- |
| `src/25_evaluate_qwen.py` | Qwen3.6-27B and shared prompt/parser, metrics and result writing | `slurm/evaluate_qwen.sh` |
| `src/26_evaluate_apertus.py` | Apertus v1.5-8B | `slurm/evaluate_apertus.sh` |
| `src/27_evaluate_mistral.py` | Mistral Small 3.2 24B Instruct | `slurm/evaluate_mistral.sh` |
| `src/28_evaluate_gemma.py` | Gemma 3 27B IT | `slurm/evaluate_gemma.sh` |
| `src/29_evaluate_llama.py` | Llama 3.1-8B-Instruct single-GPU evaluation and smoke | `slurm/evaluate_llama.sh` |
| `src/29_evaluate_llama_70b.py` | Preserved Llama 3.3 70B evaluator and multi-GPU diagnostics | `slurm/evaluate_llama_70b.sh` |
| `src/30_evaluate_phi4.py` | Phi-4-reasoning-plus and reasoning diagnostic | `slurm/evaluate_phi4.sh` |
| `src/31_evaluate_phi4_standard.py` | Standard Phi-4 original prompt | `slurm/evaluate_phi4_standard.sh` |
| `src/32_evaluate_phi4_prompt_sensitivity.py` | Standard Phi-4 modified prompt | `slurm/evaluate_phi4_standard.sh` |
| `src/33_evaluate_openai.py` | OpenAI API evaluation and resumable SQLite ledger | Local Windows workflow |
| `src/diagnose_cuda_transfer.py` | Model-free GPU transfer diagnostic | `slurm/diagnose_cuda_transfer.sh` |

The earlier script inventory covers `src/01_*` through `src/24_*`. Tests live in `tests/` and cover the training workflow, each LLM evaluator and CUDA transfer diagnostics. Encoder training manifests are `configs/training_runs.json` and `configs/training_runs07_08.json`. API dependencies are in `requirements-api.txt`.

### LLM notebooks

- [07_qwen_evaluation_analysis.ipynb](notebooks/07_qwen_evaluation_analysis.ipynb)
- [08_apertus_evaluation_analysis.ipynb](notebooks/08_apertus_evaluation_analysis.ipynb)
- [09_mistral_evaluation_analysis.ipynb](notebooks/09_mistral_evaluation_analysis.ipynb)
- [10_gemma_evaluation_analysis.ipynb](notebooks/10_gemma_evaluation_analysis.ipynb)
- [11_phi4_evaluation_analysis.ipynb](notebooks/11_phi4_evaluation_analysis.ipynb)
- [12_gpt_terra_evaluation_analysis.ipynb](notebooks/12_gpt_terra_evaluation_analysis.ipynb)

All twelve notebooks parse as JSON and contain saved execution outputs with no recorded error cells. This audit did not execute them.

### Result paths

- `results/bert/run_01`?`run_06`, `results/mathbert/run_01`?`run_05`, and `results/{deberta,modernbert}/run_01`?`run_08` hold encoder training artifacts, including failed DeBERTa Run 01. Only the selected runs contain `sum_test/` and `umwp/`.
- `results/llm/{qwen3.6-27b,apertus-v1.5-8b,mistral-small-3.2-24b,gemma-3-27b-it}/{sum,umwp}/run_01/` contains the frozen strict prediction, metrics and confusion-matrix files.
- `results/llm/phi-4/{sum,umwp}/{run_02,run_02_cleaned}/` and `results/llm/phi-4/umwp/run_01/` hold the Phi-4 conditions. Cleaned directories contain predictions only.
- `results/api/openai/gpt-6.1-terra/{sum,umwp}/run_01/` and `results/api/openai/gpt-6.1-terra/umwp/limit_10/run_01/` hold API outputs and metadata. SQLite checkpoints are ignored.
- `results/encoder_training_comparison.md` and `results/final_test_comparison.md` are existing local reports.

## 16. Encoder run register

This table uses `best_checkpoint_validation_f1` and the recorded checkpoint step from each run's validation JSON. The planned epoch budget comes from the saved configuration. At 1,596 steps per epoch, checkpoint step identifies the selected epoch.

| Family | Run | LR | Planned epochs | Best epoch | Checkpoint step | Validation F1 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BERT | run_01 | 2e-05 | 5 | 5 | 7980 | 84.31 |
| BERT | run_02 (selected) | 3e-05 | 7 | 7 | 11172 | 85.17 |
| BERT | run_03 | 2e-05 | 7 | 7 | 11172 | 84.98 |
| BERT | run_04 | 4e-05 | 7 | 7 | 11172 | 85.13 |
| BERT | run_05 | 5e-05 | 5 | 5 | 7980 | 84.59 |
| BERT | run_06 | 5e-05 | 7 | 5 | 7980 | 85.10 |
| MathBERT | run_01 | 3e-05 | 7 | 7 | 11172 | 78.01 |
| MathBERT | run_02 | 2e-05 | 7 | 5 | 7980 | 77.37 |
| MathBERT | run_03 | 4e-05 | 5 | 4 | 6384 | 77.39 |
| MathBERT | run_04 (selected) | 4e-05 | 7 | 5 | 7980 | 78.16 |
| MathBERT | run_05 | 5e-05 | 7 | 2 | 3192 | 75.20 |
| DeBERTa-v3-base | run_01 (failed) | 3e-05 | 7 | 1 | 1596 | 0.00 |
| DeBERTa-v3-base | run_02 | 3e-05 | 7 | 7 | 11172 | 87.14 |
| DeBERTa-v3-base | run_03 | 2e-05 | 7 | 7 | 11172 | 86.52 |
| DeBERTa-v3-base | run_04 | 4e-05 | 7 | 7 | 11172 | 87.11 |
| DeBERTa-v3-base | run_05 (selected) | 5e-05 | 7 | 7 | 11172 | 87.24 |
| DeBERTa-v3-base | run_06 | 3e-05 | 5 | 5 | 7980 | 86.90 |
| DeBERTa-v3-base | run_07 | 6e-05 | 7 | 6 | 9576 | 86.51 |
| DeBERTa-v3-base | run_08 | 5e-05 | 10 | 5 | 7980 | 86.88 |
| ModernBERT-base | run_01 | 3e-05 | 5 | 5 | 7980 | 87.52 |
| ModernBERT-base | run_02 | 3e-05 | 7 | 7 | 11172 | 87.23 |
| ModernBERT-base | run_03 | 4e-05 | 5 | 5 | 7980 | 87.39 |
| ModernBERT-base | run_04 | 4e-05 | 7 | 7 | 11172 | 87.77 |
| ModernBERT-base | run_05 | 5e-05 | 5 | 4 | 6384 | 87.61 |
| ModernBERT-base | run_06 (selected) | 5e-05 | 7 | 7 | 11172 | 87.94 |
| ModernBERT-base | run_07 | 6e-05 | 7 | 7 | 11172 | 87.89 |
| ModernBERT-base | run_08 | 5e-05 | 10 | 5 | 7980 | 87.35 |

The four validation selections and frozen test results follow. Each test cell reports accuracy / UNANSWERABLE precision / recall / F1 as percentages. The full artifacts also include macro-F1, predictions and confusion matrices.

| Encoder | Selected run | Evaluator loaded under `models/` | SUM Test | UMWP |
| --- | --- | --- | --- | --- |
| BERT | run_02 | `bert/run_02/best_model/` | 84.80 / 85.46 / 83.86 / 84.65 | 67.62 / 73.02 / 55.88 / 63.31 |
| MathBERT | run_04 | `mathbert/run_04/best_model/` | 77.79 / 78.50 / 76.54 / 77.51 | 59.00 / 59.47 / 56.50 / 57.95 |
| DeBERTa-v3-base | run_05 | `deberta/run_05/checkpoints/checkpoint-11172/` | 86.71 / 86.39 / 87.13 / 86.76 | 71.25 / 75.24 / 63.35 / 68.78 |
| ModernBERT-base | run_06 | `modernbert/run_06/checkpoints/checkpoint-11172/` | 87.61 / 90.32 / 84.25 / 87.18 | 69.69 / 79.66 / 52.88 / 63.57 |

See [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) for the worktree state and prioritized handover.
