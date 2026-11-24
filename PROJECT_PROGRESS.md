# CDS1092 NLP Project ? Progress and Handover

Last updated **2026-11-24** (Europe/Zurich).

Start here, then read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the methodology, model IDs, complete run register, data audit, metrics and file inventory. Use [NOTEBOOK_GUIDELINES.md](NOTEBOOK_GUIDELINES.md) when continuing the analysis. Repository scripts and saved JSON/CSV artifacts take precedence over draft prose.

## Current state

The project classifies the answerability of mathematical word problems using question text alone. ANSWERABLE = 0 and UNANSWERABLE = 1, the positive class. The four encoder families were fine-tuned and selected using fixed SUM training and validation data. SUM Test is held-out internal evaluation, and official UMWP is external evaluation. MATE is the prepared second external benchmark.

| Encoder | Completed runs | Selected | Validation F1 | SUM Test F1 | UMWP F1 |
| --- | --- | --- | ---: | ---: | ---: |
| BERT | 01?06 | 02, epoch 7 | 85.17% | 84.65% | 63.31% |
| MathBERT | 01?05 | 04, epoch 5 | 78.16% | 77.51% | 57.95% |
| DeBERTa-v3-base | 02?08 | 05, epoch 7 | 87.24% | 86.76% | 68.78% |
| ModernBERT-base | 01?08 | 06, epoch 7 | 87.94% | 87.18% | 63.57% |

Twenty-six encoder training runs are complete. DeBERTa Run 01 failed with non-finite training loss. Selected checkpoints are absent locally and their retention on Mercury remains unverified. See [the saved encoder test comparison](results/final_test_comparison.md).

- **Complete, original frozen prompt:** Qwen3.6-27B, Apertus v1.5-8B, Mistral Small 3.2 24B, Gemma 3 27B IT and GPT-6.1 Terra each have full SUM Test and UMWP results. Qwen leads these LLM runs by positive-class F1 on SUM Test (87.69%) and UMWP (93.12%). GPT-6.1 Terra's full API F1 is 85.15% and 92.19% respectively, with zero INVALID responses on either dataset.
- **Complete, standard Phi-4:** Original-prompt strict UMWP F1 is 0.00%. Removing the period in the label instruction yields strict F1 of 63.09% on SUM Test and 78.48% on UMWP. Offline extraction from those same modified-prompt generations yields 78.07% and 81.18%. Keep these three conditions separate.
- **Standard Phi-4 on MATE:** Original-prompt strict `run_01` scored 0.13% UNANSWERABLE F1 with 3,074 INVALID outputs. Modified-prompt strict `run_02` scored 66.69% F1 with 205 INVALID outputs. The new offline `run_02_cleaned` result scored 69.30% F1 with zero INVALID outputs by extracting the first non-empty line of the 205 strict invalid responses. All 9,586 rows and raw responses were preserved, and metrics and confusion matrices were independently verified. [MATE conditions and artifact paths](PROJECT_CONTEXT.md#standard-phi-4-mate-conditions)
- **Meta Llama:** Mercury job 1894 passed all eight CPU-to-GPU checks and failed all eight GPU-to-GPU checks. Both transfer directions silently corrupted FP32 and BF16 values. The root cause is unconfirmed. Llama 3.1-8B-Instruct replaces the unquantized two-GPU 70B setup. The 8B model is reported downloaded on Mercury. The single-A100 BF16 integration passed offline checks. The 70B model and diagnostics were retained separately. Mercury job 1895 loaded successfully in BF16 on NVIDIA A100 80 GB PCIe at cuda:0 (14.99 GiB peak allocated). Both outputs were valid strict labels. The ANSWERABLE probe was correct and the UNANSWERABLE probe returned ANSWERABLE, so the smoke test failed on classification. The explicit benchmark override is implemented and offline-tested. SUM, UMWP and MATE completed under the original strict prompt with the override recorded. All nine artifacts in `1070238` were independently checked against dataset rows and hashes, parsing, metrics and confusion matrices. UNANSWERABLE F1 was 14.07%, 20.63% and 17.03% respectively. [Technical context and results](PROJECT_CONTEXT.md#meta-llama-31-8b-instruct-on-mercury)
- **Phi-4-reasoning-plus:** Reasoning output is constrained by the 16-token exact-label protocol. No completed benchmark artifacts are available for this variant.
- **MATE prepared:** The notebook verifies the raw hash and removes both rows of every pair with an exact or normalized UMWP overlap. It excluded 207 pairs (414 rows) and retained 9,586 rows in 4,793 complete balanced pairs. Assertions confirmed required values, unique texts and labels per pair, and zero exact or normalized matches with all fixed SUM splits and UMWP. The regenerated test CSV has unique integer `id` values 1 through 9,586 in retained order as its first column. All previous field values, original pair IDs and row order match the preceding export. Both CSVs reload exactly and the exclusion ledger hash is unchanged. The processed file-byte and LF-normalized SHA-256 are both `377f7a28a6b2a5236ce24eeebc889c99ec38ff92c52c9b0a888d71f6542da5b8`. See [the MATE notebook](notebooks/13_mate_dataset_analysis.ipynb) and [technical context](PROJECT_CONTEXT.md#13-mate-dataset-audit) for artifact paths and hashes. Encoder, local LLM, API and Slurm integration is complete locally. All four encoder MATE evaluations are saved. Apertus, Gemma, Mistral, Qwen and standard Phi-4 have committed MATE outputs in `4c155fe`.
- **Three-dataset notebooks complete:** Encoder notebooks `03` through `06` retain their original training and SUM validation-based checkpoint selection. Their frozen test analyses now cover SUM Test, UMWP and MATE. Local LLM notebooks `07` through `11` and the new [Llama notebook](notebooks/14_llama_evaluation_analysis.ipynb) cover the same datasets in that order. Phi-4 presents original-prompt strict, modified-prompt strict and offline cleaned conditions separately. Original-prompt SUM is N/A, and historical cleaned SUM/UMWP metrics are reconstructed from the saved prediction CSVs. All ten notebooks executed from fresh `fhgr` kernels in both launch directories. The score and available confusion-matrix checks independently covered 35 saved prediction artifacts.

### MATE encoder results on Mercury, 2026-11-24

| Model | Run | Slurm job | Accuracy | F1 UNANSWERABLE | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| BERT | run_02 | 1883 | 0.7402461924 | 0.7456068655 | 0.7401307985 |
| DeBERTa | run_05 | 1885 | 0.7599624452 | 0.7646517337 | 0.7598671122 |
| ModernBERT | run_06 | 1886 | 0.7351345712 | 0.7307814654 | 0.7350653041 |

All 13 artifacts are present in `596029d`. Each run's 9,586 predictions align with the fixed dataset and sequential IDs. Local checks confirmed matching texts and metadata, confusion matrices, recalculated metrics and counts, selected checkpoints, and evaluation parameters. See [technical context](PROJECT_CONTEXT.md#completed-mate-encoder-evaluations-on-mercury) for details.

The first four jobs (1879 through 1882) failed before inference because `sbatch --wrap` used `sh` and could not execute `source`. Explicit Bash resolved the shell issue. MathBERT job 1884 failed on a stale SUM/UMWP guard inside `evaluate_dataset()`. The local fix uses the same dataset registry as the CLI. Three MathBERT regression tests passed without loading models, including the MATE path and unchanged SUM, UMWP and `both` behavior. The 59 existing evaluator tests also passed. MathBERT Run 04 later completed MATE evaluation. Its five artifacts are in `039fb6f`.

Gemma replaces Gemini. The remaining Meta evaluation completes the selected LLM roster. No additional LLM families are required.

## Next work in priority order

1. **Update the GPT-6.1 Terra notebook after MATE finishes.** The user reports that the API run is still in progress. Verify its saved artifacts before extending the notebook.
2. **Complete final cross-model analysis, scientific report and presentation.** Cover dataset transfer, invalid outputs, pair/source/category results and qualitative errors. Keep original strict, modified strict and offline Phi-4 conditions distinct.

## Local Git and uncommitted work

Snapshot on 2026-11-24. Local HEAD is `b5a9027`. Nine notebook updates and this progress update are unstaged. The new Llama notebook is untracked. The running Terra MATE process and its files were left untouched.

## Continuing in a new conversation

Use repository-relative source paths and distinguish saved evidence, reported Mercury behavior and pending work. Recheck Git state and checkpoint availability before operating on Mercury. Keep fixed evaluation files and saved results intact. The user generally communicates in Turkish and prefers English technical documentation and notebook prose. Confirm the scope before any operational step that goes beyond the active request. The MATE preparation notebook was executed from the repository root and the notebooks directory with the existing fhgr kernel. The integration passed offline CLI, loader, metadata, output guard, API cache and wrapper checks, plus 59 existing evaluator tests. No training, inference, API calls, Slurm jobs, commits or pushes were performed.

The override passed 67 offline evaluator tests, plus Python and Bash syntax and Slurm argument-routing checks. The completed Llama benchmark artifacts were later pulled and validated offline.
