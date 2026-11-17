"""Five FP32 GPU optimizer steps for ModernBERT compatibility, not an experiment.

Usage inside a single-GPU SLURM allocation: python -u src/21_test_modernbert.py
Only SUM train is read; no model checkpoints or experiment results are saved.
Ordinary Hugging Face downloads/cache may be used. Trainer has a disposable
temporary directory only to obtain the installed version's AdamW grouping and
defaults. The five updates use constant LR 3e-5, without warmup: a zero-LR first
step would not test parameter updates. No Trainer.train/evaluate call is made.
SDPA attention is explicit for FP32; Flash Attention 2 is not used. This tests
this backend/configuration, not every ModernBERT acceleration path.
"""

import json
import os
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "answerdotai/ModernBERT-base"
LABEL_TO_ID = {"ANSWERABLE": 0, "UNANSWERABLE": 1}
BATCH_SIZE = 32
MAX_LENGTH = 256
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
SEED = 42
STEPS = 5


def load_train_sample():
    """Run-style validation; select 160 reproducible rows from SUM train only."""
    import pandas as pd

    frame = pd.read_csv(ROOT / "data/processed/sum/train.csv")
    required = {"pair_id", "text", "label"}
    if not required <= set(frame.columns):
        raise ValueError(f"train: required columns are {sorted(required)}.")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("train: empty data or missing required values.")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("train: questions must be strings.")
    if not frame["label"].isin(LABEL_TO_ID).all():
        raise ValueError("train: unexpected labels.")
    if len(frame) < BATCH_SIZE * STEPS:
        raise ValueError("SUM train has too few rows for five full batches.")
    sample = frame.sample(n=BATCH_SIZE * STEPS, random_state=SEED)
    print("SUM training row indices:", sample.index.tolist(), flush=True)
    return pd.DataFrame({"text": sample["text"],
                         "labels": sample["label"].map(LABEL_TO_ID)})


def main():
    state = {"step": 0, "operation": "environment initialization"}
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Use ordinary python with one GPU, not torchrun.")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        import torch
        from torch.utils.data import DataLoader
        from datasets import Dataset
        from transformers import (
            AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
            DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one allocated, visible CUDA GPU is required.")
        set_seed(SEED)
        versions = {"python": platform.python_version(), "cuda": torch.version.cuda}
        for package in ("torch", "transformers", "datasets", "numpy", "pandas",
                        "tokenizers", "accelerate", "huggingface-hub"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("GPU memory bytes:", torch.cuda.get_device_properties(0).total_memory, flush=True)
        print("Software:", json.dumps(versions), flush=True)
        print("Configuration: FP32; FP16=False; BF16=False; attention=sdpa; "
              "batch=32; max_length=256; AdamW LR=3e-5; weight_decay=0.01; seed=42", flush=True)

        def finite(tensor, name):
            if tensor.is_floating_point() or tensor.is_complex():
                if not bool(torch.isfinite(tensor.detach()).all()):
                    raise FloatingPointError(f"NaN/Inf in {name}; shape={tuple(tensor.shape)}, "
                                             f"dtype={tensor.dtype}")

        def parameters_finite():
            for name, parameter in model.named_parameters():
                if parameter.is_floating_point() and parameter.dtype != torch.float32:
                    raise RuntimeError(f"Expected FP32 parameter: {name} is {parameter.dtype}")
                finite(parameter, f"parameter:{name}")
            for name, buffer in model.named_buffers():
                finite(buffer, f"buffer:{name}")

        def gradients_finite():
            count = 0
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    count += 1
                    finite(parameter.grad, f"gradient:{name}")
            if not count:
                raise RuntimeError("No parameter gradients were produced.")

        state["operation"] = "pretrained configuration/tokenizer loading"
        config = AutoConfig.from_pretrained(MODEL_ID)
        revision = getattr(config, "_commit_hash", None) or "main"
        if config.model_type != "modernbert" or config.max_position_embeddings < MAX_LENGTH:
            raise ValueError("Unexpected ModernBERT architecture or position capacity.")
        # Older ModernBERT implementations offer an optional compilation path.
        # Disable it when present to isolate the ordinary FP32 SDPA path.
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        config.num_labels = 2
        config.label2id = LABEL_TO_ID
        config.id2label = {value: key for key, value in LABEL_TO_ID.items()}
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
        state["operation"] = "FP32 model loading"
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=revision, config=config, dtype=torch.float32,
            attn_implementation="sdpa",
        )
        parameters_finite()
        vocab = tokenizer.get_vocab()
        embedding_rows = model.get_input_embeddings().num_embeddings
        if not vocab or min(vocab.values()) < 0 or max(vocab.values()) >= embedding_rows:
            raise ValueError("Tokenizer IDs exceed model embedding capacity.")
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id != model.config.pad_token_id:
            raise ValueError("Missing or mismatched padding token ID.")
        if any(index < 0 or index >= embedding_rows for index in tokenizer.all_special_ids):
            raise ValueError("Special token ID outside model embeddings.")
        if model.config.num_labels != 2 or model.config.label2id != LABEL_TO_ID:
            raise ValueError("Incorrect classification head or label mapping.")
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Full encoder/head fine-tuning is required.")
        model.to("cuda")
        parameters_finite()
        print("Model/tokenizer:", type(model).__name__, type(tokenizer).__name__,
              "revision:", revision, "parameter dtype:", next(model.parameters()).dtype,
              "attention:", model.config._attn_implementation, flush=True)

        state["operation"] = "SUM train preprocessing"
        data = Dataset.from_pandas(load_train_sample(), preserve_index=False).map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
            batched=True, remove_columns=["text"], load_from_cache_file=False,
        )
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
        with tempfile.TemporaryDirectory(prefix="cds1092-modernbert-test-") as temporary:
            state["operation"] = "AdamW initialization"
            arguments = TrainingArguments(
                output_dir=temporary, per_device_train_batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                optim="adamw_torch", gradient_accumulation_steps=1,
                fp16=False, bf16=False, seed=SEED, data_seed=SEED,
                save_strategy="no", eval_strategy="no", logging_strategy="no", report_to="none",
            )
            trainer = Trainer(model=model, args=arguments, train_dataset=data,
                              processing_class=tokenizer, data_collator=collator)
            optimizer = trainer.create_optimizer()
            parameters_finite()
            print("Optimizer:", type(optimizer).__name__, "defaults:", optimizer.defaults,
                  "gradient clipping:", arguments.max_grad_norm, flush=True)
            iterator = iter(loader)
            batch = {key: value.to("cuda") for key, value in next(iterator).items()}
            state["operation"] = "initial logits/loss"
            model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                output = model(**batch)
            finite(output.logits, "initial logits")
            finite(output.loss, "initial loss")
            print("Initial loss:", output.loss.item(), "logits range:", output.logits.min().item(),
                  output.logits.max().item(), flush=True)
            del output
            model.train()
            for step in range(1, STEPS + 1):
                state.update(step=step, operation="batch preparation")
                if step > 1:
                    batch = {key: value.to("cuda") for key, value in next(iterator).items()}
                optimizer.zero_grad(set_to_none=True)
                state["operation"] = "forward/logits/loss"
                with torch.autocast(device_type="cuda", enabled=False):
                    output = model(**batch)
                    finite(output.logits, "training logits")
                    finite(output.loss, "training loss")
                    loss = output.loss.detach().item()
                    state["operation"] = "backward"
                    output.loss.backward()
                gradients_finite()
                state["operation"] = "gradient norm/clipping"
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.max_grad_norm)
                finite(norm, "gradient norm before clipping")
                gradients_finite()
                state["operation"] = "AdamW update"
                optimizer.step()
                parameters_finite()
                for name, parameter in model.named_parameters():
                    for key, value in optimizer.state.get(parameter, {}).items():
                        if isinstance(value, torch.Tensor):
                            finite(value, f"optimizer:{name}:{key}")
                print(f"Step {step}/{STEPS}: loss={loss:.9g}; grad_norm={norm.item():.9g}; "
                      f"lr={optimizer.param_groups[0]['lr']:.9g}; FP32 parameters finite", flush=True)
                del output
        print("PASS: five actual GPU AdamW updates completed with finite logits, losses, "
              "gradients and FP32 parameters. No checkpoint or result saved. "
              "This does not establish full-run stability.", flush=True)
    except Exception as exc:
        print(f"FAIL: step={state['step']} operation={state['operation']}; "
              f"{type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
