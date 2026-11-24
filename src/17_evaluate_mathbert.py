"""Evaluate the frozen, validation-selected MathBERT Run 04; never train or tune.

Usage: python src/17_evaluate_mathbert.py --dataset sum
       python src/17_evaluate_mathbert.py --dataset umwp
       python src/17_evaluate_mathbert.py --dataset mate
Default (no arguments): evaluate both benchmarks separately.
Only requested test data are read. Outputs must not already exist.
The model/tokenizer are loaded locally; no checkpoint selection or training occurs.
"""

import argparse
import json
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/mathbert/run_04/best_model"
CONFIG_PATH = ROOT / "results/mathbert/run_04/config.json"
VALIDATION_PATH = CONFIG_PATH.with_name("validation_metrics.json")
LABEL_TO_ID = {"ANSWERABLE": 0, "UNANSWERABLE": 1}
LABELS = list(LABEL_TO_ID)
DATASET_OUTPUTS = {"sum": "sum_test", "umwp": "umwp", "mate": "mate"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_dataset(name):
    path = ROOT / f"data/processed/{name}/test.csv"
    if name == "mate":
        shared = import_module("25_evaluate_qwen")
        rows, _ = shared.load_dataset(ROOT, name)
        return path, pd.DataFrame(rows).astype({"id": int, "pair_id": int, "label_id": int})
    # Preserve literal text and metadata such as strings resembling NA tokens.
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"text", "label", "pair_id"} if name == "sum" else {
        "id", "text", "label", "category", "source"}
    require(required.issubset(frame.columns), f"Missing columns: {sorted(required-set(frame.columns))}")
    require(len(frame) == (10944 if name == "sum" else 5200), "Unexpected test dataset size.")
    require(frame["text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
            "Missing, empty, or invalid test text.")
    require(frame["label"].isin(LABELS).all(), "Invalid ground-truth labels.")
    if name == "umwp":
        counts = frame["label"].value_counts()
        require(all(counts.get(label, 0) == 2600 for label in LABELS), "UMWP class balance differs.")
        require(frame["id"].is_unique and frame["id"].ne("").all(), "UMWP IDs must be unique and nonempty.")
    return path, frame


def calculate_metrics(true_ids, predicted_ids):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_ids, predicted_ids, average="binary", pos_label=1, zero_division=0)
    correct = int(np.sum(true_ids == predicted_ids))
    return {
        "accuracy": float(accuracy_score(true_ids, predicted_ids)),
        "precision_unanswerable": float(precision),
        "recall_unanswerable": float(recall),
        "f1_unanswerable": float(f1),
        "macro_f1": float(f1_score(true_ids, predicted_ids, labels=[0, 1], average="macro", zero_division=0)),
        "total_examples": len(true_ids), "correct_predictions": correct,
        "incorrect_predictions": len(true_ids) - correct,
    }


def evaluate_dataset(dataset):
    """Evaluate one explicit benchmark with the already selected best model."""
    require(dataset in {"sum", "umwp"}, "Expected SUM test or UMWP.")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Selected model directory missing: {MODEL_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(config["run_name"] == "run_04", "Expected MathBERT Run 04 training configuration.")
    require(config["label_to_id"] == LABEL_TO_ID, "Training label mapping differs.")
    require(config["model_name"] == "tbs17/MathBERT-custom"
            and config["model_family"] == "mathbert", "Unexpected pretrained model/family.")
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    require(validation["configuration"]["label_to_id"] == LABEL_TO_ID,
            "Validation report label mapping differs.")
    require(validation["configuration"]["model_name"] == config["model_name"],
            "Validation report pretrained model differs.")
    checkpoint = validation["best_checkpoint"]
    require("/models/mathbert/run_04/checkpoints/" in checkpoint.replace("\\", "/"),
            "Validation checkpoint is not from MathBERT Run 04.")
    max_length = int(config["max_length"])
    require(max_length == 256 and validation["configuration"]["max_length"] == max_length,
            "Expected the unchanged training maximum sequence length of 256.")
    batch_size = int(config.get("eval_batch_size", config["batch_size"]))
    require(max_length > 0 and batch_size > 0, "Invalid saved tokenization/batch settings.")
    dataset_path, frame = load_dataset(dataset)
    output = ROOT / "results/mathbert/run_04" / DATASET_OUTPUTS[dataset]
    # Protect previously recorded evaluation and training artifacts.
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists; refusing to overwrite: {output}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    set_seed(int(config.get("seed", config["random_seed"])))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation only; no training or tuning will occur.")
    print("Selected model:", MODEL_PATH)
    print("Dataset:", dataset, dataset_path)
    print("Examples:", len(frame))
    print("Device:", device)
    print("Evaluation batch size:", batch_size)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    require(model.config.num_labels == 2 and model.config.label2id == LABEL_TO_ID,
            "Saved model class mapping does not match the selected training setup.")
    require(max_length <= model.config.max_position_embeddings,
            "Input length exceeds saved model positional capacity.")
    require(len(tokenizer) == model.get_input_embeddings().num_embeddings
            and tokenizer.pad_token_id == model.config.pad_token_id,
            "Saved tokenizer/model vocabulary size or padding ID differs.")
    model.to(device)
    model.eval()
    probabilities = []
    predictions = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            # Only question text reaches the tokenizer/model. Order is unchanged.
            texts = frame["text"].iloc[start:start + batch_size].tolist()
            inputs = tokenizer(texts, truncation=True, max_length=max_length,
                               padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    predicted_ids = np.concatenate(predictions)
    probability = np.concatenate(probabilities)
    require(probability.shape == (len(frame), 2) and np.isfinite(probability).all(),
            "Invalid inference probability output.")
    true_ids = frame["label"].map(LABEL_TO_ID).to_numpy()
    metrics = calculate_metrics(true_ids, predicted_ids)
    matrix = confusion_matrix(true_ids, predicted_ids, labels=[0, 1])
    metrics.update({"dataset": dataset, "dataset_path": str(dataset_path),
                    "model_path": str(MODEL_PATH), "selected_run": "run_04",
                    "positive_class": "UNANSWERABLE", "max_length": max_length,
                    "eval_batch_size": batch_size, "device": str(device),
                    "confusion_matrix": matrix.tolist(),
                    "confusion_matrix_label_order": LABELS,
                    "confusion_matrix_axes": "rows=true, columns=predicted",
                    "prediction_rule": "argmax", "evaluation_only": True})
    software_versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "numpy", "pandas", "scikit-learn",
                    "tokenizers", "huggingface-hub"):
        try:
            software_versions[package] = version(package)
        except PackageNotFoundError:
            software_versions[package] = None
    metrics.update({
        "model_family": "mathbert", "pretrained_model_id": config["model_name"],
        "pretrained_revision_resolved": config.get("pretrained_revision_resolved"),
        "checkpoint_path": checkpoint,
        "checkpoint_path_note": "Training provenance; weights loaded only from model_path.",
        "best_checkpoint_validation_f1": validation["best_checkpoint_validation_f1"],
        "selection_basis": "SUM validation UNANSWERABLE F1; Run 04 fixed before test evaluation",
        "training_config_path": str(CONFIG_PATH),
        "validation_metrics_path": str(VALIDATION_PATH),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "seed": int(config.get("seed", config["random_seed"])),
        "label_to_id": LABEL_TO_ID, "padding": "dynamic per batch",
        "truncation": True, "local_files_only": True,
        "deterministic_algorithms": True, "software_versions": software_versions,
        "tokenizer_class": type(tokenizer).__name__,
    })
    report = classification_report(true_ids, predicted_ids, labels=[0, 1],
                                   target_names=LABELS, zero_division=0, output_dict=True)
    report_text = classification_report(true_ids, predicted_ids, labels=[0, 1],
                                        target_names=LABELS, zero_division=0, digits=6)
    # Preserve useful metadata for analysis; it never reaches model inputs.
    result = frame.drop(columns=["label"]).copy()
    result["true_label"] = frame["label"]
    result["predicted_label"] = [LABELS[index] for index in predicted_ids]
    result["correct"] = true_ids == predicted_ids
    result["probability_answerable"] = probability[:, 0]
    result["probability_unanswerable"] = probability[:, 1]
    common = ["text", "true_label", "predicted_label", "correct",
              "probability_answerable", "probability_unanswerable"]
    if dataset == "umwp":
        result = result[["id", *common, "category", "source"]]
        duplicates = result.loc[result["text"].duplicated(keep=False)].copy()
        groups = duplicates.groupby("text", sort=False)
        inconsistent = int((groups["predicted_label"].nunique() > 1).sum())
        metrics["duplicate_consistency"] = {
            "duplicated_question_texts": groups.ngroups,
            "conflicting_ground_truth_texts": int((groups["true_label"].nunique() > 1).sum()),
            "texts_with_inconsistent_predictions": inconsistent,
            "identical_text_received_identical_predictions": inconsistent == 0,
        }
        print("Duplicate consistency:", metrics["duplicate_consistency"])
    else:
        metadata = [column for column in result if column not in common]
        result = result[[*metadata, *common]]
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(matrix, index=pd.Index(LABELS, name="true_label"),
                 columns=LABELS).to_csv(output / "confusion_matrix.csv", encoding="utf-8")
    if dataset == "umwp":
        duplicates.to_csv(output / "duplicate_analysis.csv", index=False, encoding="utf-8")
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "classification_report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "classification_report.txt").open("x", encoding="utf-8") as handle:
        handle.write(report_text)
    print("\nEvaluation metrics:")
    for key in ["accuracy", "precision_unanswerable", "recall_unanswerable", "f1_unanswerable",
                "macro_f1", "total_examples", "correct_predictions", "incorrect_predictions"]:
        print(f"{key}: {metrics[key]}")
    print("Confusion matrix (rows=true, columns=predicted; ANSWERABLE, UNANSWERABLE):")
    print(matrix)
    for path in sorted(output.iterdir()):
        print("Saved:", path)
    print("Evaluation completed. No training occurred; no model or hyperparameters were changed.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation only: frozen MathBERT Run 04; no training or test-based selection.")
    parser.add_argument("--dataset", choices=[*DATASET_OUTPUTS, "both"], default="both",
                        help="Default: both benchmarks; each is saved separately.")
    args = parser.parse_args()
    datasets = ["sum", "umwp"] if args.dataset == "both" else [args.dataset]
    # Preflight all requested destinations so an existing second benchmark
    # cannot cause an avoidable partial rerun of the first benchmark.
    for dataset in datasets:
        output = CONFIG_PATH.parent / DATASET_OUTPUTS[dataset]
        if output.exists():
            raise FileExistsError(f"Evaluation output already exists; refusing to overwrite: {output}")
    for dataset in datasets:
        evaluate_dataset(dataset)


if __name__ == "__main__":
    # Required by deterministic CUDA matrix operations; set before importing torch.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
