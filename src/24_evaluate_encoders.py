"""Final evaluation only: validation-selected DeBERTa Run 05 / ModernBERT Run 06.

Usage: python src/24_evaluate_encoders.py --model deberta
       python src/24_evaluate_encoders.py --model modernbert
       python src/24_evaluate_encoders.py --model both
Each command defaults to both test benchmarks; --dataset sum|umwp|mate is optional.
Only saved local checkpoints/tokenizers are loaded. No training, model search,
threshold tuning, downloads or checkpoint writes. Existing outputs are refused.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
LABEL_TO_ID = {"ANSWERABLE": 0, "UNANSWERABLE": 1}
LABELS = list(LABEL_TO_ID)
DATASET_OUTPUTS = {"sum": "sum_test", "umwp": "umwp", "mate": "mate"}
SELECTED = {
    "deberta": ("run_05", "microsoft/deberta-v3-base", "deberta-v2"),
    "modernbert": ("run_06", "answerdotai/ModernBERT-base", "modernbert"),
}


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


def weight_files(directory):
    """Check availability only; never deserialize weights during preflight."""
    require((directory / "config.json").is_file(), f"Missing model config: {directory}")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = directory / filename
        if path.is_file() and path.stat().st_size > 0:
            return [path]
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / filename
        if not index.is_file():
            continue
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        require(bool(mapping), f"Empty model shard map: {index}")
        shards = []
        for name in sorted(set(mapping.values())):
            shard = (directory / name).resolve()
            require(shard.is_relative_to(directory.resolve()), "Shard path escapes model directory.")
            require(shard.is_file() and shard.stat().st_size > 0, f"Missing/empty shard: {shard}")
            shards.append(shard)
        return shards
    raise FileNotFoundError(f"No saved model weights found in {directory}")


def has_tokenizer(directory):
    return ((directory / "tokenizer_config.json").is_file()
            and any((directory / name).is_file() for name in
                    ("tokenizer.json", "spm.model", "vocab.txt", "vocab.json")))


def resolve_selection(family):
    """Resolve only the preselected run's recorded checkpoint or best-model export."""
    run, model_id, model_type = SELECTED[family]
    result_root = ROOT / "results" / family / run
    run_root = ROOT / "models" / family / run
    config_path = result_root / "config.json"
    validation_path = result_root / "validation_metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    metadata = validation["configuration"]
    require(config["run_name"] == run and config["model_family"] == family,
            "Training configuration does not match the fixed selected run.")
    require(config["model_name"] == model_id and metadata["model_name"] == model_id,
            "Unexpected pretrained model identity.")
    for key in config.keys() & metadata.keys():
        require(config[key] == metadata[key], f"Config/validation metadata disagree: {key}")
    require(config["label_to_id"] == metadata["label_to_id"] == LABEL_TO_ID,
            "Training label mapping differs.")
    require(config["max_length"] == metadata["max_length"] == 256,
            "Expected the unchanged maximum sequence length 256.")
    batch_size = int(config.get("eval_batch_size", config["batch_size"]))
    require(batch_size == 32, "Expected the recorded evaluation batch size 32.")
    require(config["metric_for_best_model"] == "f1" and config["greater_is_better"] is True,
            "Expected validation-F1 checkpoint selection.")
    checkpoint_f1 = validation["best_checkpoint_validation_f1"]
    require(isinstance(checkpoint_f1, (int, float)) and math.isfinite(checkpoint_f1),
            "Invalid recorded validation checkpoint F1.")
    recorded = validation["best_checkpoint"].replace("\\", "/")
    prefix = f"/models/{family}/{run}/checkpoints/"
    require(prefix in recorded, "Recorded checkpoint belongs to a different run.")
    checkpoint_name = recorded.rsplit(prefix, 1)[1]
    require(re.fullmatch(r"checkpoint-[0-9]+", checkpoint_name) is not None,
            "Invalid recorded checkpoint directory name.")
    checkpoint = run_root / "checkpoints" / checkpoint_name
    export = run_root / "best_model"
    if checkpoint.is_dir():
        model_path = checkpoint
        resolution = "validation_metadata_checkpoint"
    elif export.is_dir():
        model_path = export
        resolution = "saved_best_model_export"
    else:
        raise FileNotFoundError(
            f"Selected {family}/{run} weights unavailable. Expected {checkpoint} "
            f"or the saved best-model export {export}. No other checkpoint will be used."
        )
    weights = weight_files(model_path)
    # An incomplete recorded checkpoint fails above rather than silently trying
    # another checkpoint. Only matching run-local tokenizer files are allowed.
    tokenizer_path = model_path if has_tokenizer(model_path) else export
    require(has_tokenizer(tokenizer_path), f"Saved tokenizer missing: {tokenizer_path}")
    return {"family": family, "run": run, "config": config, "validation": validation,
            "model_id": model_id, "model_type": model_type, "model_path": model_path,
            "tokenizer_path": tokenizer_path, "checkpoint_path": checkpoint,
            "resolution": resolution, "weight_files": weights,
            "result_root": result_root, "config_path": config_path,
            "validation_path": validation_path, "batch_size": batch_size}


def preflight(families, datasets):
    """Resolve every requested model/output before loading weights or test rows."""
    selections = []
    for family in families:
        run = SELECTED[family][0]
        for dataset in datasets:
            output = ROOT / "results" / family / run / DATASET_OUTPUTS[dataset]
            if output.exists():
                raise FileExistsError(f"Evaluation output exists; refusing to overwrite: {output}")
        selections.append(resolve_selection(family))
    return selections


def evaluate_dataset(dataset, selection, model, tokenizer, torch, device):
    """Unchanged text-only, argmax inference; write a fresh dataset-specific output."""
    config = selection["config"]
    dataset_path, frame = load_dataset(dataset)
    batch_size, max_length = selection["batch_size"], int(config["max_length"])
    output = selection["result_root"] / DATASET_OUTPUTS[dataset]
    if output.exists():
        raise FileExistsError(f"Evaluation output exists; refusing to overwrite: {output}")
    model.eval()
    probabilities, predictions = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(frame), batch_size):
            texts = frame["text"].iloc[start:start + batch_size].tolist()
            inputs = tokenizer(texts, truncation=True, max_length=max_length,
                               padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            require(bool(torch.isfinite(logits).all()),
                    f"Non-finite logits at {selection['family']} / {dataset}, row {start}.")
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    predicted_ids = np.concatenate(predictions)
    probability = np.concatenate(probabilities)
    require(predicted_ids.shape == (len(frame),) and np.isin(predicted_ids, [0, 1]).all(),
            "Invalid predicted-label output.")
    require(probability.shape == (len(frame), 2) and np.isfinite(probability).all()
            and np.allclose(probability.sum(axis=1), 1.0, atol=1e-6),
            "Invalid inference probability output.")
    true_ids = frame["label"].map(LABEL_TO_ID).to_numpy()
    metrics = calculate_metrics(true_ids, predicted_ids)
    matrix = confusion_matrix(true_ids, predicted_ids, labels=[0, 1])
    report = classification_report(true_ids, predicted_ids, labels=[0, 1],
                                   target_names=LABELS, zero_division=0, output_dict=True)
    report_text = classification_report(true_ids, predicted_ids, labels=[0, 1],
                                        target_names=LABELS, zero_division=0, digits=6)
    require(int(matrix.sum()) == len(frame) and int(np.trace(matrix)) == metrics["correct_predictions"],
            "Confusion matrix and metric counts disagree.")
    require(math.isclose(report["UNANSWERABLE"]["f1-score"], metrics["f1_unanswerable"], abs_tol=1e-12),
            "Classification report and positive-class F1 disagree.")
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "numpy", "pandas", "scikit-learn",
                    "tokenizers", "sentencepiece", "protobuf", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    metrics.update({
        "dataset": dataset, "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "model_path": str(selection["model_path"]),
        "tokenizer_path": str(selection["tokenizer_path"]),
        "checkpoint_path": str(selection["checkpoint_path"]),
        "recorded_training_checkpoint_path": selection["validation"]["best_checkpoint"],
        "model_resolution": selection["resolution"],
        "model_family": selection["family"], "selected_run": selection["run"],
        "pretrained_model_id": selection["model_id"],
        "pretrained_revision_resolved": config.get("pretrained_revision_resolved"),
        "best_checkpoint_validation_f1": selection["validation"]["best_checkpoint_validation_f1"],
        "selection_basis": "Fixed run and validation-selected checkpoint; no test-based selection",
        "training_config_path": str(selection["config_path"]),
        "validation_metrics_path": str(selection["validation_path"]),
        "training_config_sha256": hashlib.sha256(selection["config_path"].read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256((selection["model_path"] / "config.json").read_bytes()).hexdigest(),
        "positive_class": "UNANSWERABLE", "label_to_id": LABEL_TO_ID,
        "max_length": max_length, "eval_batch_size": batch_size,
        "seed": int(config.get("seed", config["random_seed"])),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "fp16": False, "bf16": False, "padding": "dynamic per batch", "truncation": True,
        "attention_implementation": model.config._attn_implementation,
        "tokenizer_class": type(tokenizer).__name__, "software_versions": versions,
        "confusion_matrix": matrix.tolist(), "confusion_matrix_label_order": LABELS,
        "confusion_matrix_axes": "rows=true, columns=predicted",
        "prediction_rule": "argmax", "evaluation_only": True,
        "local_files_only": True, "deterministic_algorithms": True,
    })
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
    else:
        metadata = [column for column in result if column not in common]
        result = result[[*metadata, *common]]
    require(len(result) == metrics["total_examples"], "Prediction-row count differs.")
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(matrix, index=pd.Index(LABELS, name="true_label"),
                 columns=LABELS).to_csv(output / "confusion_matrix.csv", encoding="utf-8")
    if dataset == "umwp":
        duplicates.to_csv(output / "duplicate_analysis.csv", index=False, encoding="utf-8")
    for name, content in [("metrics.json", metrics), ("classification_report.json", report)]:
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write("\n")
    with (output / "classification_report.txt").open("x", encoding="utf-8") as handle:
        handle.write(report_text)
    print(f"Saved {selection['family']}/{selection['run']} / {dataset}: {output}", flush=True)
    print(json.dumps({key: metrics[key] for key in
                      ("accuracy", "precision_unanswerable", "recall_unanswerable",
                       "f1_unanswerable", "macro_f1", "total_examples")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["deberta", "modernbert", "both"])
    parser.add_argument("--dataset", choices=[*DATASET_OUTPUTS, "both"], default="both")
    args = parser.parse_args()
    families = list(SELECTED) if args.model == "both" else [args.model]
    datasets = ["sum", "umwp"] if args.dataset == "both" else [args.dataset]
    selections = preflight(families, datasets)
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Use one process, not distributed evaluation.")
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Final evaluation only; no training or test-based selection. Device:", device, flush=True)
    for selection in selections:
        config = selection["config"]
        set_seed(int(config.get("seed", config["random_seed"])))
        model_config = AutoConfig.from_pretrained(selection["model_path"], local_files_only=True)
        require(model_config.model_type == selection["model_type"], "Saved architecture mismatch.")
        require(model_config.num_labels == 2 and model_config.label2id == LABEL_TO_ID
                and model_config.id2label == {value: key for key, value in LABEL_TO_ID.items()},
                "Saved model label mapping differs.")
        if selection["family"] == "modernbert" and hasattr(model_config, "reference_compile"):
            model_config.reference_compile = False
        attention = config.get("attention_implementation")
        require(attention in {"eager", "sdpa"}, "Missing/unsupported recorded attention backend.")
        tokenizer = AutoTokenizer.from_pretrained(selection["tokenizer_path"],
                                                local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            selection["model_path"], config=model_config, local_files_only=True,
            dtype=torch.float32, attn_implementation=attention,
        )
        rows = model.get_input_embeddings().num_embeddings
        vocab = tokenizer.get_vocab()
        require(bool(vocab) and min(vocab.values()) >= 0 and max(vocab.values()) < rows,
                "Saved tokenizer IDs exceed model embedding capacity.")
        require(tokenizer.pad_token_id is not None and tokenizer.pad_token_id == model.config.pad_token_id,
                "Saved tokenizer/model padding mismatch.")
        require(config["max_length"] <= model.config.max_position_embeddings,
                "Input length exceeds saved positional capacity.")
        require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in model.parameters()),
                "Expected FP32 parameters.")
        model.to(device)
        model.eval()
        print("Resolved weights:", selection["model_path"], "tokenizer:", selection["tokenizer_path"], flush=True)
        for dataset in datasets:
            set_seed(int(config.get("seed", config["random_seed"])))
            evaluate_dataset(dataset, selection, model, tokenizer, torch, device)
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
