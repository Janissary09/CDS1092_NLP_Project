# CDS1092 NLP Project ? Progress and Handover

Last updated **2026-11-24** (Europe/Zurich).

Start here, then read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the methodology, model IDs, complete run register, data audit, metrics and file inventory. Use [NOTEBOOK_GUIDELINES.md](NOTEBOOK_GUIDELINES.md) when continuing the analysis. Repository scripts and saved JSON/CSV artifacts take precedence over draft prose.

## Current state

The project classifies the answerability of mathematical word problems using question text alone. ANSWERABLE = 0 and UNANSWERABLE = 1, the positive class. The four encoder families were fine-tuned and selected using fixed SUM training and validation data. SUM Test is held-out internal evaluation, and official UMWP is external evaluation. MATE is the next external benchmark.

| Encoder | Completed runs | Selected | Validation F1 | SUM Test F1 | UMWP F1 |
| --- | --- | --- | ---: | ---: | ---: |
| BERT | 01?06 | 02, epoch 7 | 85.17% | 84.65% | 63.31% |
| MathBERT | 01?05 | 04, epoch 5 | 78.16% | 77.51% | 57.95% |
| DeBERTa-v3-base | 02?08 | 05, epoch 7 | 87.24% | 86.76% | 68.78% |
| ModernBERT-base | 01?08 | 06, epoch 7 | 87.94% | 87.18% | 63.57% |

Twenty-six encoder training runs are complete. DeBERTa Run 01 failed with non-finite training loss. Selected checkpoints are absent locally and their retention on Mercury remains unverified. See [the saved encoder test comparison](results/final_test_comparison.md).

- **Complete, original frozen prompt:** Qwen3.6-27B, Apertus v1.5-8B, Mistral Small 3.2 24B, Gemma 3 27B IT and GPT-6.1 Terra each have full SUM Test and UMWP results. Qwen leads these LLM runs by positive-class F1 on SUM Test (87.69%) and UMWP (93.12%). GPT-6.1 Terra's full API F1 is 85.15% and 92.19% respectively, with zero INVALID responses on either dataset.
- **Complete, standard Phi-4:** Original-prompt strict UMWP F1 is 0.00%. Removing the period in the label instruction yields strict F1 of 63.09% on SUM Test and 78.48% on UMWP. Offline extraction from those same modified-prompt generations yields 78.07% and 81.18%. Keep these three conditions separate.
- **Blocked or limited:** Llama 3.3 70B has a reported multi-GPU transfer issue at the layer 39/40 boundary on Mercury. Phi-4-reasoning-plus is constrained by reasoning output under the 16-token exact-label protocol. The checkout contains diagnostic code for both attempts and no completed benchmark artifacts for either.
- **MATE prepared:** The notebook verifies the raw hash and removes both rows of every pair with an exact or normalized UMWP overlap. It excluded 207 pairs (414 rows) and retained 9,586 rows in 4,793 complete balanced pairs. Assertions confirmed required values, unique texts and labels per pair, and zero exact or normalized matches with all fixed SUM splits and UMWP. The regenerated test CSV has unique integer `id` values 1 through 9,586 in retained order as its first column. All previous field values, original pair IDs and row order match the preceding export. Both CSVs reload exactly and the exclusion ledger hash is unchanged. The processed file-byte and LF-normalized SHA-256 are both `377f7a28a6b2a5236ce24eeebc889c99ec38ff92c52c9b0a888d71f6542da5b8`. See [the MATE notebook](notebooks/13_mate_dataset_analysis.ipynb) and [technical context](PROJECT_CONTEXT.md#13-mate-dataset-audit) for artifact paths and hashes. Encoder, local LLM, API and Slurm integration is complete locally. MATE evaluations remain pending.
- **Analysis:** Thirteen notebooks exist with saved outputs. A final cross-model analysis, scientific report and presentation remain pending. Align the sibling LaTeX draft with the custom SUM split, current roster and recorded experiment conditions.

Gemma replaces Gemini. The remaining Meta evaluation completes the selected LLM roster. No additional LLM families are required.

## Next work in priority order

1. **Make the verified MATE benchmark and integration available on Mercury.** The two processed CSVs are now Git-eligible. After review and authorized publication, pull them and the integration changes on Mercury and verify the dataset hash.
2. **Evaluate the selected encoders and LLMs on MATE.** Reuse validation-selected checkpoints and the frozen zero-shot protocol. Report any Phi-4 sensitivity runs under separate conditions. Meta's MATE run depends on the Mercury model being operational.
3. **Complete the Meta Llama evaluation on Mercury.** Confirm the transfer issue and inspect existing diagnostic logs if available. Use a smaller model if 70B remains unreliable. Record the actual model ID and complete the benchmark evaluations for that model.
4. **Complete comparative analysis, notebooks, scientific report and presentation.** Cover dataset transfer, invalid outputs, pair/source/category results, and qualitative errors. Report the limitations of the single-seed encoder search, unequal tuning budgets, benchmark overlap and separate Phi-4 conditions.

## Local Git and uncommitted work

Snapshot on 2026-11-24. Branch `main` at `9862876` matched the locally cached `origin/main` (0 ahead, 0 behind). The index was clean at inspection. No fetch or Mercury login was performed.

| Status | Paths |
| --- | --- |
| Modified before this audit | `PROJECT_CONTEXT.md`, `notebooks/03_bert_training_analysis.ipynb`, `notebooks/04_mathbert_training_analysis.ipynb` |
| Deleted in the worktree before this audit | `data/raw/umwp/3. dataset/MATE Dataset` and `data/raw/umwp/3. dataset/MATE Dataset.csv` |
| Untracked documentation | `PROJECT_PROGRESS.md`, `NOTEBOOK_GUIDELINES.md`, `results/encoder_training_comparison.md`, `results/final_test_comparison.md` |
| Untracked notebooks | `notebooks/07_qwen_evaluation_analysis.ipynb` through `notebooks/12_gpt_terra_evaluation_analysis.ipynb` (all six; exact names in the context inventory) |
| Untracked API outputs | `results/api/openai/gpt-6.1-terra/umwp/limit_10/run_01/` contains `predictions.csv`, `metrics.json`, `confusion_matrix.csv` and `run_metadata.json` |
| Ignored local files | API SQLite checkpoints, local environment and credential files, and Python caches |
| MATE worktree changes | Evaluator and Slurm integration and scoped `.gitignore` exceptions are unstaged. The raw MATE CSV, both processed MATE CSVs and `notebooks/13_mate_dataset_analysis.ipynb` are untracked. |

The README modification predates MATE preparation. The raw and processed MATE artifacts are eligible for tracking under the local ignore exceptions. The new notebook can regenerate the two processed artifacts given the pinned raw data and fixed comparison files. The full Terra results are already committed in `fea226d`. The Phi-4 strict and cleaned artifacts are already committed in `fe21599`. Existing work was left unstaged.

## Continuing in a new conversation

Use repository-relative source paths and distinguish saved evidence, reported Mercury behavior and pending work. Recheck Git state and checkpoint availability before operating on Mercury. Keep fixed evaluation files and saved results intact. The user generally communicates in Turkish and prefers English technical documentation and notebook prose. Confirm the scope before any operational step that goes beyond the active request. The MATE preparation notebook was executed from the repository root and the notebooks directory with the existing fhgr kernel. The integration passed offline CLI, loader, metadata, output guard, API cache and wrapper checks, plus 59 existing evaluator tests. No training, inference, API calls, Slurm jobs, commits or pushes were performed.
