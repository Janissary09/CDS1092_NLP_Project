# Erkennung unbeantwortbarer mathematischer Textaufgaben

FHGR NLP research project comparing task-specific fine-tuned encoder models with general-purpose LLMs in a zero-shot setting.

- Main dataset: SUM
- External evaluation dataset: UMWP

## SUM preprocessing

The 36,480 original SUM pairs are split into 70% train, 15% validation,
and 15% test using random seed 42, before expanding pairs into classification
examples. Answerable and unanswerable versions of each problem always remain
in the same split. Pairs sharing any exact duplicate question text are grouped
transitively into the same split to prevent cross-split text leakage.

The final dataset in `data/processed/sum/` contains 72,960 examples, with each
split exactly balanced between `ANSWERABLE` and `UNANSWERABLE`. Final validation
confirmed zero pair leakage and zero exact-text leakage across splits.
Preparation script: `python src/prepare_sum.py`.
