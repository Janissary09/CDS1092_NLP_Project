"""Train one fixed overnight matrix entry; SUM train/validation only.

Usage: python src/22_train_encoder.py --model deberta --run run_03
Configuration: configs/overnight_runs.json. No existing experiment is resumed or
replaced. ModernBERT retains the tested FP32 SDPA backend (no Flash Attention).
"""

import argparse
import hashlib
import json
import math
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("CDS1092_PROJECT_ROOT", str(CODE_ROOT))).resolve()
DATA_DIR = PROJECT_ROOT / "data/processed/sum"
MANIFEST_PATH = CODE_ROOT / "configs/overnight_runs.json"
LABEL_TO_ID = {"ANSWERABLE": 0, "UNANSWERABLE": 1}



def load_split(name):
    """Read a permitted split and reject missing text or unknown labels."""
    if name not in {"train", "validation"}:
        raise ValueError("This training script permits only train and validation.")
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    required = {"pair_id", "text", "label"}
    if not required <= set(frame.columns):
        raise ValueError(f"{name}: required columns are {sorted(required)}.")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f"{name}: empty data or missing required values.")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{name}: questions must be strings.")
    if not frame["label"].isin(LABEL_TO_ID).all():
        raise ValueError(f"{name}: unexpected labels.")
    # pair_id is metadata, and ground_truth is never used as model input.
    return pd.DataFrame({"text": frame["text"], "labels": frame["label"].map(LABEL_TO_ID)})


def compute_metrics(prediction):
    """Binary precision, recall and F1 all refer to UNANSWERABLE (label 1)."""
    predicted_labels = np.argmax(prediction.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        prediction.label_ids, predicted_labels,
        average="binary", pos_label=1, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(prediction.label_ids, predicted_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["deberta", "modernbert"])
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = [r for r in manifest["runs"]
               if r["model_family"] == args.model and r["run_name"] == args.run]
    if len(entries) != 1:
        raise ValueError("Run is not a unique entry in the overnight matrix.")
    entry = entries[0]
    CONFIG = {**manifest["shared"], **manifest["models"][args.model], **entry}
    run_model_dir = PROJECT_ROOT / "models" / args.model / args.run
    MODEL_DIR = run_model_dir / "best_model"
    CHECKPOINT_DIR = run_model_dir / "checkpoints"
    RESULTS_DIR = PROJECT_ROOT / "results" / args.model / args.run
    # Use ordinary `python`, not torchrun. Respect a server-assigned GPU mask;
    # otherwise choose GPU 0. Trainer must not start automatic multi-GPU work.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Launch with python on one GPU, without distributed training.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    # Check for existing runs before loading data or downloading anything.
    # Refuse partial runs too: restarting requires an explicitly chosen new run.
    run_model_dir = MODEL_DIR.parent
    for directory in (run_model_dir, RESULTS_DIR):
        if directory.exists():
            raise FileExistsError(
                f"Experiment directory already exists; refusing to overwrite: {directory}"
            )
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated, visible CUDA GPU is required.")
    set_seed(CONFIG["random_seed"])
    print("Training device:", "CUDA" if torch.cuda.is_available() else "CPU")

    train_frame = load_split("train")
    validation_frame = load_split("validation")
    if len(train_frame) != 51072 or len(validation_frame) != 10944:
        raise ValueError("Expected the fixed SUM train/validation example counts.")
    # Read-only isolation check: never load any held-out dataset.
    import csv
    pair_sets, text_sets = [], []
    for split in ("train", "validation"):
        with (DATA_DIR / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        pair_sets.append({row["pair_id"] for row in rows})
        text_sets.append({row["text"] for row in rows})
    if pair_sets[0] & pair_sets[1] or text_sets[0] & text_sets[1]:
        raise ValueError("SUM train/validation pair or exact-text leakage detected.")
    # Resolve metadata first, then load tokenizer and weights from the same
    # immutable commit when the Hub supplies one. Each model uses its own tokenizer.
    pretrained_config = AutoConfig.from_pretrained(
        CONFIG["model_name"], revision=CONFIG["model_revision"],
    )
    resolved_revision = getattr(pretrained_config, "_commit_hash", None)
    revision = resolved_revision or CONFIG["model_revision"]
    if pretrained_config.model_type != CONFIG["model_type"]:
        raise ValueError("Pretrained architecture differs from the configured model family.")
    if CONFIG["max_length"] > pretrained_config.max_position_embeddings:
        raise ValueError("Maximum sequence length exceeds pretrained position embeddings.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"], revision=revision, use_fast=True,
        )
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "Tokenizer loading failed. Check Transformers/tokenizers and, for DeBERTa, "
            "SentencePiece/protobuf dependencies. No packages were installed."
        ) from exc

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=CONFIG["max_length"])

    # Tokenize in memory, remove raw text, and dynamically pad each batch.
    train_data = Dataset.from_pandas(train_frame, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False,
    )
    validation_data = Dataset.from_pandas(validation_frame, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False,
    )
    model_kwargs = {}
    if args.model == "modernbert":
        # Match the successful technical test: FP32 SDPA, no optional compile path.
        if hasattr(pretrained_config, "reference_compile"):
            pretrained_config.reference_compile = False
        pretrained_config.num_labels = 2
        pretrained_config.label2id = LABEL_TO_ID
        pretrained_config.id2label = {value: key for key, value in LABEL_TO_ID.items()}
        model_kwargs = {"config": pretrained_config, "attn_implementation": "sdpa"}
    else:
        model_kwargs = {"num_labels": 2, "label2id": LABEL_TO_ID,
                        "id2label": {value: key for key, value in LABEL_TO_ID.items()}}
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], revision=revision, dtype=torch.float32, **model_kwargs,
    )

    # Initialize the downstream classification head/pooler as needed.
    # Do not freeze parameters: train the full encoder and the new head.
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Expected all encoder and classification parameters to be trainable.")
    # Checkpoints may reserve unused embedding rows beyond the tokenizer
    # vocabulary. Keep these rows intact; require every tokenizer ID to fit.
    embedding_rows = model.get_input_embeddings().num_embeddings
    if not tokenizer.get_vocab() or max(tokenizer.get_vocab().values()) >= embedding_rows:
        raise ValueError("Tokenizer vocabulary IDs exceed model input embeddings.")
    if min(tokenizer.get_vocab().values()) < 0:
        raise ValueError("Tokenizer vocabulary contains a negative ID.")
    if tokenizer.pad_token_id != model.config.pad_token_id:
        raise ValueError("Tokenizer and model padding token IDs differ.")
    if any(token_id < 0 or token_id >= model.config.vocab_size
           for token_id in tokenizer.all_special_ids):
        raise ValueError("Tokenizer special token ID is outside the model vocabulary.")

    software_versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "numpy", "pandas",
                    "scikit-learn", "accelerate", "tokenizers", "huggingface-hub",
                    "sentencepiece", "protobuf"):
        try:
            software_versions[package] = version(package)
        except PackageNotFoundError:
            software_versions[package] = None
    provenance = {
        "pretrained_model_id": CONFIG["model_name"],
        "pretrained_revision_requested": CONFIG["model_revision"],
        "pretrained_revision_resolved": resolved_revision,
        "revision_source": "AutoConfig._commit_hash" if resolved_revision else "unavailable",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "model_class": type(model).__name__,
        "model_type": model.config.model_type,
        "model_embedding_rows": embedding_rows,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "fp16": False,
        "bf16": False,
        "software_versions": software_versions,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "reference_compile": getattr(model.config, "reference_compile", None),
        "batch_id": manifest["batch_id"],
        "matrix_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_sha256": {name: hashlib.sha256((DATA_DIR / f"{name}.csv").read_bytes()).hexdigest()
                           for name in ("train", "validation")},
    }

    # One GPU, no gradient accumulation: each batch is one optimizer step.
    # Integer warmup_steps avoids the deprecated warmup_ratio argument in v5.3.
    total_steps = math.ceil(len(train_data) / CONFIG["batch_size"]) * CONFIG["max_epochs"]
    warmup_steps = math.ceil(total_steps * CONFIG["warmup_ratio"])
    arguments = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=CONFIG["max_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        gradient_accumulation_steps=1,
        fp16=False,
        bf16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=CONFIG["random_seed"],
        data_seed=CONFIG["random_seed"],
        report_to="none",
    )
    # Initialization and TrainingArguments dependency/device checks succeeded.
    # Recheck before reserving outputs: another process may have created a run
    # during initialization. Exclusive creation still refuses existing paths.
    # Reserve before Trainer, which may itself create the checkpoint directory.
    for directory in (run_model_dir, RESULTS_DIR):
        if directory.exists():
            raise FileExistsError(
                f"Experiment directory already exists; refusing to overwrite: {directory}"
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    run_model_dir.mkdir(parents=True, exist_ok=False)
    for directory in (MODEL_DIR, CHECKPOINT_DIR):
        directory.mkdir(exist_ok=False)

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=CONFIG["early_stopping_patience"],
            early_stopping_threshold=0.0,
        )],
    )

    # Fail rather than silently run the experiment with reduced-precision weights.
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise RuntimeError("Overnight runs require FP32 parameters throughout training setup.")

    # Validation runs each epoch. Stop after two evaluations without an F1
    # improvement; Trainer reloads the best checkpoint when training finishes.
    # Record the exact matrix entry before starting training.
    run_config = {
        **CONFIG,
        "run_name": args.run,
        "model_family": args.model,
        **provenance,
        "num_train_epochs": arguments.num_train_epochs,
        "train_batch_size": arguments.per_device_train_batch_size,
        "eval_batch_size": arguments.per_device_eval_batch_size,
        "seed": arguments.seed,
        "data_seed": arguments.data_seed,
        "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
        "metric_for_best_model": arguments.metric_for_best_model,
        "greater_is_better": arguments.greater_is_better,
        "train_dataset_path": str(DATA_DIR / "train.csv"),
        "validation_dataset_path": str(DATA_DIR / "validation.csv"),
        "label_to_id": LABEL_TO_ID,
        "warmup_steps": warmup_steps,
    }
    with (RESULTS_DIR / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")

    trainer.train()
    best_metrics = trainer.evaluate(eval_dataset=validation_data)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(MODEL_DIR)

    # Preserve epoch losses and validation metrics in a small, readable table.
    pd.DataFrame(trainer.state.log_history).to_csv(
        RESULTS_DIR / "training_history.csv", index=False,
    )
    report = {
        "best_validation_metrics": best_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_checkpoint_validation_f1": (float(trainer.state.best_metric)
                                          if trainer.state.best_metric is not None else None),
        "configuration": {
            **CONFIG,
            **provenance,
            "data_seed": arguments.data_seed,
            "gradient_accumulation_steps": arguments.gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "planned_optimizer_steps": total_steps,
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
            "label_to_id": LABEL_TO_ID,
            "evaluation_strategy": "epoch",
            "scheduler": "linear",
            "optimizer": "adamw_torch",
            "device": str(arguments.device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
    }
    with (RESULTS_DIR / "validation_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Best model and tokenizer: {MODEL_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    if trainer.state.best_metric is not None:
        print(f"Best validation F1: {trainer.state.best_metric:.6f}")


if __name__ == "__main__":
    main()
