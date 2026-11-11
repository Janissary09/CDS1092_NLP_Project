"""Fine-tune BERT on SUM training data, selecting by validation F1.

Run from the project root: python src/train_bert.py
Requires the project libraries and Accelerate (used by Hugging Face Trainer).
Only train.csv and validation.csv are read. Run on one allocated CUDA GPU;
CUDA_VISIBLE_DEVICES can select it on the FHGR Linux server.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/sum"
MODEL_DIR = PROJECT_ROOT / "models/bert/best_model"
CHECKPOINT_DIR = PROJECT_ROOT / "models/bert/checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results/bert"
LABEL_TO_ID = {"ANSWERABLE": 0, "UNANSWERABLE": 1}
CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 256,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "max_epochs": 5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "random_seed": 42,
    "positive_class": "UNANSWERABLE",
    "positive_label_id": 1,
    "early_stopping_patience": 2,
    "best_model_metric": "validation_f1",
}


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
    # Use ordinary `python`, not torchrun. Respect a server-assigned GPU mask;
    # otherwise choose GPU 0. Trainer must not start automatic multi-GPU work.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Launch with python on one GPU, without distributed training.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    import torch
    import transformers
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        BertTokenizerFast,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if torch.cuda.device_count() > 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU.")
    set_seed(CONFIG["random_seed"])
    print("Training device:", "CUDA" if torch.cuda.is_available() else "CPU")

    train_frame = load_split("train")
    validation_frame = load_split("validation")
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=CONFIG["max_length"])

    # Tokenize in memory, remove raw text, and dynamically pad each batch.
    train_data = Dataset.from_pandas(train_frame, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False,
    )
    validation_data = Dataset.from_pandas(validation_frame, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=2,
        label2id=LABEL_TO_ID, id2label={value: key for key, value in LABEL_TO_ID.items()},
    )

    for directory in (MODEL_DIR, CHECKPOINT_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

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

    # Validation runs each epoch. Stop after two evaluations without an F1
    # improvement; Trainer reloads the best checkpoint when training finishes.
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
        "best_checkpoint_validation_f1": float(trainer.state.best_metric),
        "configuration": {
            **CONFIG,
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
    with (RESULTS_DIR / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Best model and tokenizer: {MODEL_DIR}")
    print(f"Training history and validation metrics: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
