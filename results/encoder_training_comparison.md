# Encoder Training Comparison: DeBERTa and ModernBERT

Date: **2026-11-18**. Scope: DeBERTa **run_02-run_08** and ModernBERT **run_01-run_08** (15 completed runs). Failed DeBERTa run_01 is excluded. This report analyzes saved SUM validation artifacts only; no training, inference or test-set evaluation was run.

## 1. Sources and interpretation rules

For every listed run, the sources are `results/<family>/<run>/config.json`, `training_history.csv` and `validation_metrics.json`. Links below are relative to this report's `results/` directory. Configuration provides learning rate and maximum epochs; the report provides `best_checkpoint` and `best_checkpoint_validation_f1`; history identifies the corresponding original epoch and its metrics. All 45 requested source files are present.

The training summary event (`train_runtime`) separates original epoch evaluations from the final restored-model reevaluation. The latter inherits the final training epoch and is **not** an additional epoch or evidence of the best checkpoint's epoch. Best epoch is determined by matching the saved checkpoint step to the original history event. For all 15 runs, checkpoint F1 agrees with the history maximum, restored-model metrics agree with the original best event, and overlapping configuration/report fields agree.

Selection is strictly by binary validation F1 for **UNANSWERABLE (label 1)**; ANSWERABLE is label 0. Precision and recall also refer to UNANSWERABLE. Metrics below are fractions rounded to six decimals; comparison deltas are **percentage points (pp)** computed from unrounded records. No test metrics enter selection.

Shared recorded settings: FP32 parameters; FP16/BF16 disabled; train/evaluation batch 32; maximum length 256; seed/data seed 42; AdamW, weight decay 0.01; linear scheduler; 5% warmup; epoch validation/checkpointing; best-checkpoint restoration; early-stopping patience 2. Reports record 51,072 training and 10,944 validation examples. Where dataset hashes are present (all included runs except DeBERTa run_02), the recorded train/validation hashes agree across runs; absence for run_02 limits independent byte-level provenance verification for that run.

## 2. All-run validation comparison

| Model | Run / validation source | LR | Max epochs | Best epoch | Completed epochs | F1 (U) | Precision (U) | Recall (U) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeBERTa-v3-base | [run_02](deberta/run_02/validation_metrics.json) | 3e-05 | 7 | 7 | 7 | 0.871413 | 0.866065 | 0.876827 | 0.870614 |
| DeBERTa-v3-base | [run_03](deberta/run_03/validation_metrics.json) | 2e-05 | 7 | 7 | 7 | 0.865223 | 0.858299 | 0.872259 | 0.864126 |
| DeBERTa-v3-base | [run_04](deberta/run_04/validation_metrics.json) | 4e-05 | 7 | 7 | 7 | 0.871127 | 0.873851 | 0.868421 | 0.871528 |
| DeBERTa-v3-base | [run_05](deberta/run_05/validation_metrics.json) | 5e-05 | 7 | 7 | 7 | 0.872356 | 0.870451 | 0.874269 | 0.872076 |
| DeBERTa-v3-base | [run_06](deberta/run_06/validation_metrics.json) | 3e-05 | 5 | 5 | 5 | 0.869045 | 0.874676 | 0.863487 | 0.869883 |
| DeBERTa-v3-base | [run_07](deberta/run_07/validation_metrics.json) | 6e-05 | 7 | 6 | 7 | 0.865066 | 0.881305 | 0.849415 | 0.867507 |
| DeBERTa-v3-base | [run_08](deberta/run_08/validation_metrics.json) | 5e-05 | 10 | 5 | 7 | 0.868753 | 0.888592 | 0.849781 | 0.871619 |
| ModernBERT-base | [run_01](modernbert/run_01/validation_metrics.json) | 3e-05 | 5 | 5 | 5 | 0.875210 | 0.893714 | 0.857456 | 0.877741 |
| ModernBERT-base | [run_02](modernbert/run_02/validation_metrics.json) | 3e-05 | 7 | 7 | 7 | 0.872323 | 0.893466 | 0.852156 | 0.875274 |
| ModernBERT-base | [run_03](modernbert/run_03/validation_metrics.json) | 4e-05 | 5 | 5 | 5 | 0.873851 | 0.887946 | 0.860197 | 0.875822 |
| ModernBERT-base | [run_04](modernbert/run_04/validation_metrics.json) | 4e-05 | 7 | 7 | 7 | 0.877738 | 0.899808 | 0.856725 | 0.880665 |
| ModernBERT-base | [run_05](modernbert/run_05/validation_metrics.json) | 5e-05 | 5 | 4 | 5 | 0.876060 | 0.884214 | 0.868056 | 0.877193 |
| ModernBERT-base | [run_06](modernbert/run_06/validation_metrics.json) | 5e-05 | 7 | 7 | 7 | 0.879413 | 0.906814 | 0.853618 | 0.882950 |
| ModernBERT-base | [run_07](modernbert/run_07/validation_metrics.json) | 6e-05 | 7 | 7 | 7 | 0.878925 | 0.904955 | 0.854349 | 0.882310 |
| ModernBERT-base | [run_08](modernbert/run_08/validation_metrics.json) | 5e-05 | 10 | 5 | 7 | 0.873512 | 0.889394 | 0.858187 | 0.875731 |

## 3. Selected run in each family

- **DeBERTa-v3-base: run_05**, LR `5e-05`, maximum 7 epochs; best epoch **7**, `checkpoint-11172`. Validation F1 **0.872355944566**, precision 0.870451, recall 0.874269, accuracy 0.872076. [Configuration](deberta/run_05/config.json), [history](deberta/run_05/training_history.csv), [validation report](deberta/run_05/validation_metrics.json).
- **ModernBERT-base: run_06**, LR `5e-05`, maximum 7 epochs; best epoch **7**, `checkpoint-11172`. Validation F1 **0.879412595312**, precision 0.906814, recall 0.853618, accuracy 0.882950. [Configuration](modernbert/run_06/config.json), [history](modernbert/run_06/training_history.csv), [validation report](modernbert/run_06/validation_metrics.json).

**Observation:** neither follow-up setting changes either family's selected run. These are the best observed checkpoints in the recorded search, not evidence of a global optimum. Single-seed differences do not establish statistical significance or future test-set ranking.

## 4. Follow-up comparisons

The previous 5e-5 / 7-epoch reference is **DeBERTa run_05** and **ModernBERT run_06**. ModernBERT run_05 has a five-epoch budget and is not the seven-epoch reference.

### 4.1 Higher learning rate: 6e-5 / 7 epochs

| Model | Reference | New run | Reference F1 | New F1 | Delta F1 (pp) | Delta precision (pp) | Delta recall (pp) | Delta accuracy (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeBERTa-v3-base | run_05 | run_07 | 0.872356 | 0.865066 | -0.7290 | +1.0853 | -2.4854 | -0.4569 |
| ModernBERT-base | run_06 | run_07 | 0.879413 | 0.878925 | -0.0488 | -0.1859 | +0.0731 | -0.0640 |

**Observations:** increasing LR from 5e-5 to 6e-5 reduces the best validation F1 in both families. DeBERTa gains precision but loses substantially more recall. ModernBERT changes only slightly, with lower precision and slightly higher recall; the approximately 0.049 pp F1 decrease should not be interpreted as statistically established superiority of either learning rate. The planned epoch budget and warmup horizon remain unchanged.

### 4.2 Longer planned training: 5e-5 / 10 epochs

| Model | Reference | New run | Reference F1 | New F1 | Delta F1 (pp) | Delta precision (pp) | Delta recall (pp) | Delta accuracy (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeBERTa-v3-base | run_05 | run_08 | 0.872356 | 0.868753 | -0.3603 | +1.8140 | -2.4488 | -0.0457 |
| ModernBERT-base | run_06 | run_08 | 0.879413 | 0.873512 | -0.5901 | -1.7420 | +0.4569 | -0.7219 |

**Observations:** the longer-budget experiments have lower best F1 in both families. DeBERTa again exchanges recall for precision, while ModernBERT has lower precision and slightly higher recall. Neither 10-epoch experiment reaches epoch 10 (section 6).

**Scheduler interpretation:** these are fresh pretrained-model runs, not continuations of the seven-epoch checkpoints. At 1,596 optimizer steps per epoch, seven epochs plan 11,172 steps with 559 warmup steps; ten epochs plan 15,960 steps with 798 warmup steps. The entire warmup/decay trajectory differs from the first epoch onward. Therefore this comparison cannot isolate the effect of simply appending epochs 8-10; those epochs were not even executed. Early stopping does not rescale the original planned schedule.

## 5. Original per-epoch trajectories: run_05-run_08

Training loss is the epoch training-log value. Validation loss/F1 comes from the original epoch evaluation, **excluding** the post-training restored-model event. "Best" marks the F1-selected checkpoint, not necessarily the minimum-loss epoch. Values are rounded to six decimals; no epoch or metric is interpolated.

### DeBERTa-v3-base

Sources: [run_05/training_history.csv](deberta/run_05/training_history.csv), [run_06/training_history.csv](deberta/run_06/training_history.csv), [run_07/training_history.csv](deberta/run_07/training_history.csv), [run_08/training_history.csv](deberta/run_08/training_history.csv).

| Run | Epoch | Training loss | Validation loss | Validation F1 (U) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| run_05 | 1 | 0.496116 | 0.439374 | 0.804843 |  |
| run_05 | 2 | 0.377885 | 0.379494 | 0.835143 |  |
| run_05 | 3 | 0.305688 | 0.357111 | 0.847308 |  |
| run_05 | 4 | 0.244715 | 0.355763 | 0.862385 |  |
| run_05 | 5 | 0.187112 | 0.412169 | 0.867717 |  |
| run_05 | 6 | 0.140949 | 0.456238 | 0.871610 |  |
| run_05 | 7 | 0.103830 | 0.533745 | 0.872356 | Best |
| run_06 | 1 | 0.485798 | 0.425938 | 0.819846 |  |
| run_06 | 2 | 0.358344 | 0.341463 | 0.846427 |  |
| run_06 | 3 | 0.285678 | 0.339516 | 0.859125 |  |
| run_06 | 4 | 0.224840 | 0.351277 | 0.866403 |  |
| run_06 | 5 | 0.170232 | 0.404408 | 0.869045 | Best |
| run_07 | 1 | 0.503542 | 0.420749 | 0.804235 |  |
| run_07 | 2 | 0.385110 | 0.402214 | 0.829128 |  |
| run_07 | 3 | 0.317130 | 0.394666 | 0.838614 |  |
| run_07 | 4 | 0.252423 | 0.354839 | 0.851772 |  |
| run_07 | 5 | 0.194039 | 0.425273 | 0.860254 |  |
| run_07 | 6 | 0.143912 | 0.449162 | 0.865066 | Best |
| run_07 | 7 | 0.106510 | 0.551639 | 0.863520 |  |
| run_08 | 1 | 0.507366 | 0.411305 | 0.801850 |  |
| run_08 | 2 | 0.386320 | 0.384910 | 0.831520 |  |
| run_08 | 3 | 0.315388 | 0.403835 | 0.836127 |  |
| run_08 | 4 | 0.260162 | 0.353160 | 0.858453 |  |
| run_08 | 5 | 0.207711 | 0.365407 | 0.868753 | Best |
| run_08 | 6 | 0.167032 | 0.372199 | 0.867784 |  |
| run_08 | 7 | 0.128586 | 0.486703 | 0.867281 |  |

### ModernBERT-base

Sources: [run_05/training_history.csv](modernbert/run_05/training_history.csv), [run_06/training_history.csv](modernbert/run_06/training_history.csv), [run_07/training_history.csv](modernbert/run_07/training_history.csv), [run_08/training_history.csv](modernbert/run_08/training_history.csv).

| Run | Epoch | Training loss | Validation loss | Validation F1 (U) | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| run_05 | 1 | 0.476719 | 0.360452 | 0.827532 |  |
| run_05 | 2 | 0.326260 | 0.322330 | 0.854621 |  |
| run_05 | 3 | 0.213924 | 0.333744 | 0.867821 |  |
| run_05 | 4 | 0.114074 | 0.421062 | 0.876060 | Best |
| run_05 | 5 | 0.049338 | 0.677532 | 0.874332 |  |
| run_06 | 1 | 0.480879 | 0.363411 | 0.821665 |  |
| run_06 | 2 | 0.326013 | 0.325888 | 0.857458 |  |
| run_06 | 3 | 0.218736 | 0.319324 | 0.869308 |  |
| run_06 | 4 | 0.129783 | 0.403993 | 0.873945 |  |
| run_06 | 5 | 0.071740 | 0.577405 | 0.876245 |  |
| run_06 | 6 | 0.037812 | 0.843183 | 0.879082 |  |
| run_06 | 7 | 0.014988 | 1.022859 | 0.879413 | Best |
| run_07 | 1 | 0.485874 | 0.378166 | 0.809587 |  |
| run_07 | 2 | 0.335746 | 0.328502 | 0.854631 |  |
| run_07 | 3 | 0.235282 | 0.320011 | 0.867589 |  |
| run_07 | 4 | 0.143673 | 0.342750 | 0.876264 |  |
| run_07 | 5 | 0.080411 | 0.478063 | 0.874127 |  |
| run_07 | 6 | 0.042504 | 0.723171 | 0.878482 |  |
| run_07 | 7 | 0.017912 | 0.968041 | 0.878925 | Best |
| run_08 | 1 | 0.499968 | 0.395589 | 0.804947 |  |
| run_08 | 2 | 0.346528 | 0.347533 | 0.850188 |  |
| run_08 | 3 | 0.246235 | 0.328249 | 0.864310 |  |
| run_08 | 4 | 0.161417 | 0.340573 | 0.868665 |  |
| run_08 | 5 | 0.101099 | 0.407399 | 0.873512 | Best |
| run_08 | 6 | 0.065052 | 0.647838 | 0.872622 |  |
| run_08 | 7 | 0.042644 | 0.816112 | 0.872995 |  |

## 6. Did the 10-epoch runs reach epoch 10?

| Model / run_08 | Planned maximum | Last training epoch | Best epoch | Best F1 | Original final-epoch F1 | Reached epoch 10? |
| --- | --- | --- | --- | --- | --- | --- |
| DeBERTa-v3-base | 10 | 7 | 5 | 0.868753 | 0.867281 | No |
| ModernBERT-base | 10 | 7 | 5 | 0.873512 | 0.872995 | No |

**Observed:** both runs finish after epoch 7 with a training-summary event followed by reevaluation of the epoch-5 checkpoint. In both histories, epochs 6 and 7 do not improve on epoch 5. The final JSON metrics therefore describe the restored epoch-5 model, not the epoch-7 model.

**Interpretation:** this termination pattern is consistent with configured early-stopping patience 2. The compact artifacts do not contain an explicit stop-reason field or SLURM accounting; the exact termination cause cannot be independently proven from these three file types alone. There is no recorded training at epochs 8-10.

## 7. Overfitting-related observations and limitations

| Model | Run | Minimum-loss epoch | Minimum validation loss | Final original validation loss | First training loss | Final training loss |
| --- | --- | --- | --- | --- | --- | --- |
| DeBERTa-v3-base | run_05 | 4 | 0.355763 | 0.533745 | 0.496116 | 0.103830 |
| DeBERTa-v3-base | run_06 | 3 | 0.339516 | 0.404408 | 0.485798 | 0.170232 |
| DeBERTa-v3-base | run_07 | 4 | 0.354839 | 0.551639 | 0.503542 | 0.106510 |
| DeBERTa-v3-base | run_08 | 4 | 0.353160 | 0.486703 | 0.507366 | 0.128586 |
| ModernBERT-base | run_05 | 2 | 0.322330 | 0.677532 | 0.476719 | 0.049338 |
| ModernBERT-base | run_06 | 3 | 0.319324 | 1.022859 | 0.480879 | 0.014988 |
| ModernBERT-base | run_07 | 3 | 0.320011 | 0.968041 | 0.485874 | 0.017912 |
| ModernBERT-base | run_08 | 3 | 0.328249 | 0.816112 | 0.499968 | 0.042644 |

**Recorded observations:**

- In all eight detailed runs, training loss decreases from the first to the last recorded epoch, while validation loss ends above its earlier minimum.
- DeBERTa run_05 still improves F1 through epoch 7 despite increasing validation loss after epoch 4. Run_07 peaks in F1 at epoch 6 and declines at epoch 7. Run_08 peaks at epoch 5 and declines in both subsequent F1 evaluations.
- ModernBERT run_06 reaches its highest F1 at epoch 7, although validation loss rises from 0.319324 at epoch 3 to 1.022859 at epoch 7 and training loss falls to 0.014988. Run_07 shows a similar loss divergence while its highest F1 remains at epoch 7. Run_05 peaks in F1 at epoch 4; run_08 peaks at epoch 5 and does not recover in epochs 6-7.

**Interpretation, not a proven cause:** falling training loss with worsening validation loss is consistent with overfitting and/or increasingly confident validation errors. The logs do not isolate those mechanisms or directly measure calibration. Loss and argmax F1 measure different aspects of prediction quality, so rising loss does not by itself invalidate an F1-selected checkpoint. For run_08 in both families, the post-peak F1 decline additionally supports stopping rather than assuming further epochs would help.

**Limits:** all experiments use one recorded seed; no uncertainty estimates or significance tests are available here. Repeated validation-based search can favor validation-specific choices. The best observed validation model is not guaranteed to generalize best, and this report neither accesses nor uses SUM test or UMWP results to resolve that question. Higher LR and a longer scheduled horizon did not improve the observed best F1; that is a conclusion about these recorded runs, not a universal optimization rule.
