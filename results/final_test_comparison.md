# Final Test Evaluation Comparison

This report compares the four validation-selected checkpoints using existing evaluation artifacts only. The positive class is **UNANSWERABLE (label 1)**; ANSWERABLE is label 0. Precision, recall and F1 below refer to UNANSWERABLE. All metric values are percentages, rounded to two decimal places; differences are percentage points (pp), calculated before rounding.

SUM Test contains **10,944 examples** (5,472 per class); external UMWP contains **5,200 examples** (2,600 per class). All saved evaluations use maximum sequence length 256, evaluation batch size 32 and argmax predictions. No training, inference, threshold adjustment or test-based checkpoint selection was performed to prepare this report.

## Selected checkpoints and validation F1

Validation F1 is taken from `best_checkpoint_validation_f1` in each selected run's `validation_metrics.json`.

| Model | Selected run | Recorded best checkpoint | Validation F1 (%) |
| --- | --- | --- | ---: |
| BERT | `run_02` | `models/bert/run_02/checkpoints/checkpoint-11172` | 85.17 |
| MathBERT | `run_04` | `models/mathbert/run_04/checkpoints/checkpoint-7980` | 78.16 |
| DeBERTa-v3-base | `run_05` | `models/deberta/run_05/checkpoints/checkpoint-11172` | 87.24 |
| ModernBERT-base | `run_06` | `models/modernbert/run_06/checkpoints/checkpoint-11172` | 87.94 |

The evaluation artifacts record loading `best_model/` exports for BERT and MathBERT, and the recorded `checkpoint-11172/` directories for DeBERTa and ModernBERT. These are the fixed selected runs; the test results are not used to choose another checkpoint.

## SUM Test results

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| BERT | 84.80 | 85.46 | 83.86 | 84.65 |
| MathBERT | 77.79 | 78.50 | 76.54 | 77.51 |
| DeBERTa-v3-base | 86.71 | 86.39 | 87.13 | 86.76 |
| ModernBERT-base | 87.61 | 90.32 | 84.25 | 87.18 |

## UMWP results

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| BERT | 67.62 | 73.02 | 55.88 | 63.31 |
| MathBERT | 59.00 | 59.47 | 56.50 | 57.95 |
| DeBERTa-v3-base | 71.25 | 75.24 | 63.35 | 68.78 |
| ModernBERT-base | 69.69 | 79.66 | 52.88 | 63.57 |

## F1 difference between datasets

The gap is defined as **SUM Test F1 minus UMWP F1**. A positive value indicates lower F1 on UMWP.

| Model | SUM Test F1 (%) | UMWP F1 (%) | Gap (pp) |
| --- | ---: | ---: | ---: |
| BERT | 84.65 | 63.31 | 21.34 |
| MathBERT | 77.51 | 57.95 | 19.56 |
| DeBERTa-v3-base | 86.76 | 68.78 | 17.98 |
| ModernBERT-base | 87.18 | 63.57 | 23.61 |

## Confusion matrices

Each model is represented by its two matrix rows: **rows are true labels; columns are predicted labels**. ANSWERABLE = 0; UNANSWERABLE = 1. Thus the ANSWERABLE row contains TN/FP and the UNANSWERABLE row contains FN/TP. Counts are copied from the saved confusion matrices.

### SUM Test

| Model | True label | Predicted ANSWERABLE | Predicted UNANSWERABLE |
| --- | --- | ---: | ---: |
| BERT | ANSWERABLE | 4,691 | 781 |
| BERT | UNANSWERABLE | 883 | 4,589 |
| MathBERT | ANSWERABLE | 4,325 | 1,147 |
| MathBERT | UNANSWERABLE | 1,284 | 4,188 |
| DeBERTa-v3-base | ANSWERABLE | 4,721 | 751 |
| DeBERTa-v3-base | UNANSWERABLE | 704 | 4,768 |
| ModernBERT-base | ANSWERABLE | 4,978 | 494 |
| ModernBERT-base | UNANSWERABLE | 862 | 4,610 |

### UMWP

| Model | True label | Predicted ANSWERABLE | Predicted UNANSWERABLE |
| --- | --- | ---: | ---: |
| BERT | ANSWERABLE | 2,063 | 537 |
| BERT | UNANSWERABLE | 1,147 | 1,453 |
| MathBERT | ANSWERABLE | 1,599 | 1,001 |
| MathBERT | UNANSWERABLE | 1,131 | 1,469 |
| DeBERTa-v3-base | ANSWERABLE | 2,058 | 542 |
| DeBERTa-v3-base | UNANSWERABLE | 953 | 1,647 |
| ModernBERT-base | ANSWERABLE | 2,249 | 351 |
| ModernBERT-base | UNANSWERABLE | 1,225 | 1,375 |

## Observations and interpretation

- **SUM Test ranking by positive-class F1:** ModernBERT (87.18%) > DeBERTa (86.76%) > BERT (84.65%) > MathBERT (77.51%). This is also the validation F1 ranking. ModernBERT has the highest SUM Test accuracy and precision; DeBERTa has the highest recall.
- **UMWP ranking by positive-class F1:** DeBERTa (68.78%) > ModernBERT (63.57%) > BERT (63.31%) > MathBERT (57.95%). DeBERTa also has the highest accuracy and recall; ModernBERT has the highest precision. ModernBERT exceeds BERT by only 0.26 pp in UMWP F1.
- **Recorded cross-dataset gap:** all four models have lower F1 on UMWP. DeBERTa has the smallest gap (17.98 pp); ModernBERT has the largest (23.61 pp). ModernBERT's leading SUM Test F1 therefore does not translate into the leading UMWP F1.
- **Error pattern:** on UMWP, every model has more false negatives than false positives. ModernBERT has 1,225 false negatives and 351 false positives, consistent with its high precision (79.66%) and lower recall (52.88%). DeBERTa detects more UNANSWERABLE examples: 1,647 true positives versus ModernBERT's 1,375.
- **Interpretation boundary:** these results document weaker performance on the external dataset under the saved evaluation protocol. They do not establish the cause of the gap, statistical significance of differences, or performance across repeated seeds. The smaller cross-dataset gap alone is not a sufficient ranking criterion; absolute UMWP F1 also matters.
- **Dataset limitation recorded by all four evaluations:** UMWP contains 10 duplicated question texts, including 5 with conflicting ground-truth labels. All four saved evaluations report identical predictions for identical text. These rows remain included in the reported full-dataset metrics.

## Source files and consistency checks

All required metrics and matrices were available. Each CSV confusion matrix matched its JSON counterpart; matrix totals, class supports, accuracy, positive-class precision, recall and F1 were checked against the saved metrics. No existing artifacts were modified.

Paths below are relative to the project root; links are relative to this report. Each run provides its validation metadata, two evaluation metric files and two confusion-matrix files.

- **BERT:** `results/bert/run_02/`: [validation_metrics.json](bert/run_02/validation_metrics.json); [sum_test/metrics.json](bert/run_02/sum_test/metrics.json); [sum_test/confusion_matrix.csv](bert/run_02/sum_test/confusion_matrix.csv); [umwp/metrics.json](bert/run_02/umwp/metrics.json); [umwp/confusion_matrix.csv](bert/run_02/umwp/confusion_matrix.csv).

- **MathBERT:** `results/mathbert/run_04/`: [validation_metrics.json](mathbert/run_04/validation_metrics.json); [sum_test/metrics.json](mathbert/run_04/sum_test/metrics.json); [sum_test/confusion_matrix.csv](mathbert/run_04/sum_test/confusion_matrix.csv); [umwp/metrics.json](mathbert/run_04/umwp/metrics.json); [umwp/confusion_matrix.csv](mathbert/run_04/umwp/confusion_matrix.csv).

- **DeBERTa-v3-base:** `results/deberta/run_05/`: [validation_metrics.json](deberta/run_05/validation_metrics.json); [sum_test/metrics.json](deberta/run_05/sum_test/metrics.json); [sum_test/confusion_matrix.csv](deberta/run_05/sum_test/confusion_matrix.csv); [umwp/metrics.json](deberta/run_05/umwp/metrics.json); [umwp/confusion_matrix.csv](deberta/run_05/umwp/confusion_matrix.csv).

- **ModernBERT-base:** `results/modernbert/run_06/`: [validation_metrics.json](modernbert/run_06/validation_metrics.json); [sum_test/metrics.json](modernbert/run_06/sum_test/metrics.json); [sum_test/confusion_matrix.csv](modernbert/run_06/sum_test/confusion_matrix.csv); [umwp/metrics.json](modernbert/run_06/umwp/metrics.json); [umwp/confusion_matrix.csv](modernbert/run_06/umwp/confusion_matrix.csv).
