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
- **Blocked or limited:** Llama 3.3 70B has a reported multi-GPU transfer issue at the layer 39/40 boundary on Mercury. Phi-4-reasoning-plus is constrained by reasoning output under the 16-token exact-label protocol. The checkout contains diagnostic code for both attempts and no completed benchmark artifacts for either.
- **MATE prepared:** The notebook verifies the raw hash and removes both rows of every pair with an exact or normalized UMWP overlap. It excluded 207 pairs (414 rows) and retained 9,586 rows in 4,793 complete balanced pairs. Assertions confirmed required values, unique texts and labels per pair, and zero exact or normalized matches with all fixed SUM splits and UMWP. The regenerated test CSV has unique integer `id` values 1 through 9,586 in retained order as its first column. All previous field values, original pair IDs and row order match the preceding export. Both CSVs reload exactly and the exclusion ledger hash is unchanged. The processed file-byte and LF-normalized SHA-256 are both `377f7a28a6b2a5236ce24eeebc889c99ec38ff92c52c9b0a888d71f6542da5b8`. See [the MATE notebook](notebooks/13_mate_dataset_analysis.ipynb) and [technical context](PROJECT_CONTEXT.md#13-mate-dataset-audit) for artifact paths and hashes. Encoder, local LLM, API and Slurm integration is complete locally. MathBERT and LLM MATE evaluations remain pending.
- **Analysis:** Thirteen notebooks exist with saved outputs. A final cross-model analysis, scientific report and presentation remain pending. Align the sibling LaTeX draft with the custom SUM split, current roster and recorded experiment conditions.

### MATE encoder results on Mercury, 2026-11-24

| Model | Run | Slurm job | Accuracy | F1 UNANSWERABLE | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| BERT | run_02 | 1883 | 0.7402461924 | 0.7456068655 | 0.7401307985 |
| DeBERTa | run_05 | 1885 | 0.7599624452 | 0.7646517337 | 0.7598671122 |
| ModernBERT | run_06 | 1886 | 0.7351345712 | 0.7307814654 | 0.7350653041 |

All 13 artifacts are present in `596029d`. Each run's 9,586 predictions align with the fixed dataset and sequential IDs. Local checks confirmed matching texts and metadata, confusion matrices, recalculated metrics and counts, selected checkpoints, and evaluation parameters. See [technical context](PROJECT_CONTEXT.md#completed-mate-encoder-evaluations-on-mercury) for details.

The first four jobs (1879 through 1882) failed before inference because `sbatch --wrap` used `sh` and could not execute `source`. Explicit Bash resolved the shell issue. MathBERT job 1884 failed on a stale SUM/UMWP guard inside `evaluate_dataset()`. The local fix uses the same dataset registry as the CLI. Three MathBERT regression tests passed without loading models, including the MATE path and unchanged SUM, UMWP and `both` behavior. The 59 existing evaluator tests also passed. MathBERT MATE remains pending on Mercury.

Gemma replaces Gemini. The remaining Meta evaluation completes the selected LLM roster. No additional LLM families are required.

## Next work in priority order

1. **Complete MathBERT MATE on Mercury.** After authorized publication and pull of the local guard fix, evaluate only the selected MathBERT Run 04 on MATE and verify its artifacts.
2. **Evaluate the selected LLMs on MATE.** Use the frozen zero-shot protocol. Report any Phi-4 sensitivity runs under separate conditions. Meta's MATE run depends on the Mercury model being operational.
3. **Complete the Meta Llama evaluation on Mercury.** Confirm the transfer issue and inspect existing diagnostic logs if available. Use a smaller model if 70B remains unreliable. Record the actual model ID and complete the benchmark evaluations for that model.
4. **Complete comparative analysis, notebooks, scientific report and presentation.** Cover dataset transfer, invalid outputs, pair/source/category results, and qualitative errors. Report the limitations of the single-seed encoder search, unequal tuning budgets, benchmark overlap and separate Phi-4 conditions.

## Local Git and uncommitted work

Snapshot on 2026-11-24. Branch `main` at `596029d` matched the locally cached `origin/main` (0 ahead, 0 behind). The index was clean. The MathBERT guard and handover changes are unstaged, and `tests/test_mathbert_evaluation.py` is new. The pre-existing `results/api/openai/gpt-6.1-terra/umwp/limit_10/` artifacts remain untracked. No fetch or Mercury login was performed.

## Continuing in a new conversation

Use repository-relative source paths and distinguish saved evidence, reported Mercury behavior and pending work. Recheck Git state and checkpoint availability before operating on Mercury. Keep fixed evaluation files and saved results intact. The user generally communicates in Turkish and prefers English technical documentation and notebook prose. Confirm the scope before any operational step that goes beyond the active request. The MATE preparation notebook was executed from the repository root and the notebooks directory with the existing fhgr kernel. The integration passed offline CLI, loader, metadata, output guard, API cache and wrapper checks, plus 59 existing evaluator tests. No training, inference, API calls, Slurm jobs, commits or pushes were performed.
