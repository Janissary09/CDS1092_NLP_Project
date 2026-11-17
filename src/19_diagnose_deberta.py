"""Bounded, in-memory numerical diagnosis of DeBERTa Run 01 on SUM train only.

Run inside a fresh, single-GPU SLURM allocation:
    python -u src/19_diagnose_deberta.py --steps 5
Add --anomaly for PyTorch backward-operation tracebacks (slower).
No checkpoint, experiment result, tokenizer or dataset is saved. Trainer uses
an automatically cleaned temporary output directory, never a run directory.
This is a diagnostic replay, not an exact reproduction of the original RNG
trajectory: the initial eval probe and instrumentation differ from Trainer.train.
Forward hooks identify the first observed module output, not every functional op.
Model/tokenizer files may be downloaded into the ordinary Hugging Face cache.
"""

import argparse
import ast
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/18_train_deberta_run01.py"
SAVED_CONFIG = ROOT / "results/deberta/run_01/config.json"


class NonFiniteError(RuntimeError):
    """Stop at the first observed non-finite tensor, without further updates."""


def read_reference():
    """Read only literal settings from source; never import/execute training."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    settings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CONFIG", "LABEL_TO_ID"}:
                    settings[target.id] = ast.literal_eval(node.value)
    config, labels = settings["CONFIG"], settings["LABEL_TO_ID"]
    saved = None
    if SAVED_CONFIG.is_file():
        saved = json.loads(SAVED_CONFIG.read_text(encoding="utf-8"))
        for key, value in config.items():
            if saved[key] != value:
                raise ValueError(f"Saved/source Run 01 configuration differs: {key}")
        if saved["label_to_id"] != labels:
            raise ValueError("Saved/source label mapping differs.")
    return config, labels, saved


def load_train(labels):
    """Same required-field, missing-value, text and label checks as Run 01."""
    import pandas as pd

    frame = pd.read_csv(ROOT / "data/processed/sum/train.csv")
    required = {"pair_id", "text", "label"}
    if not required <= set(frame.columns):
        raise ValueError(f"train: required columns are {sorted(required)}.")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("train: empty data or missing required values.")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("train: questions must be strings.")
    if not frame["label"].isin(labels).all():
        raise ValueError("train: unexpected labels.")
    return pd.DataFrame({"text": frame["text"], "labels": frame["label"].map(labels)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5, help="1–20 optimizer steps (default 5).")
    parser.add_argument("--anomaly", action="store_true", help="Trace NaN-producing backward ops.")
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error("--steps must be between 1 and 20.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use ordinary python on one allocated GPU, not torchrun.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    import torch
    from datasets import Dataset
    from transformers import (
        AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated, visible CUDA GPU is required.")
    config, labels, saved = read_reference()
    set_seed(config["random_seed"])
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "datasets", "accelerate", "tokenizers",
                    "numpy", "pandas", "sentencepiece", "protobuf", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    print("Environment:", json.dumps(versions), flush=True)
    print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda, flush=True)
    print("Configuration:", json.dumps(config), flush=True)
    if saved:
        print("Original environment:", json.dumps(saved.get("software_versions")), flush=True)
    revision = ((saved or {}).get("pretrained_revision_resolved")
                or config["model_revision"])
    pretrained = AutoConfig.from_pretrained(config["model_name"], revision=revision)
    revision = getattr(pretrained, "_commit_hash", None) or revision
    print("Loading pretrained revision:", revision,
          "(main is not an immutable reproduction when original hash is unavailable)", flush=True)
    if pretrained.model_type != "deberta-v2" or config["max_length"] > pretrained.max_position_embeddings:
        raise ValueError("Unexpected pretrained architecture or positional capacity.")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=revision, use_fast=True)
    frame = load_train(labels)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config["max_length"])

    data = Dataset.from_pandas(frame, preserve_index=False).map(
        tokenize, batched=True, remove_columns=["text"], load_from_cache_file=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], revision=revision, num_labels=2, label2id=labels,
        id2label={value: key for key, value in labels.items()},
    )
    rows = model.get_input_embeddings().num_embeddings
    vocab_ids = list(tokenizer.get_vocab().values())
    if not vocab_ids or min(vocab_ids) < 0 or max(vocab_ids) >= rows:
        raise ValueError("Tokenizer IDs outside model embeddings.")
    if tokenizer.pad_token_id != model.config.pad_token_id:
        raise ValueError("Tokenizer/model padding mismatch.")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Expected full encoder and head fine-tuning.")

    state = {"step": 0, "operation": "pretrained parameter inspection"}

    def check_tensor(tensor, name):
        if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            return
        finite = torch.isfinite(tensor.detach())
        if not bool(finite.all()):
            bad = int((~finite).sum().item())
            message = (f"FIRST NON-FINITE: step={state['step']} operation={state['operation']} "
                       f"tensor={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                       f"nonfinite={bad}/{tensor.numel()}")
            print(message, flush=True)
            raise NonFiniteError(message)

    def check_nested(value, name):
        if isinstance(value, torch.Tensor):
            check_tensor(value, name)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                check_nested(item, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                check_nested(item, f"{name}[{index}]")

    def check_model():
        for name, parameter in model.named_parameters():
            check_tensor(parameter, f"parameter:{name}")
        for name, buffer in model.named_buffers():
            check_tensor(buffer, f"buffer:{name}")

    def check_gradients():
        count = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                count += 1
                check_tensor(parameter.grad, f"gradient:{name}")
        if not count:
            raise RuntimeError("Backward produced no parameter gradients.")

    check_model()
    print("Initial pretrained parameters and buffers are finite.", flush=True)
    handles = []

    def forward_hook(name):
        def hook(module, inputs, output):
            check_nested(output, f"module_output:{name}")
        return hook

    def gradient_hook(name):
        def hook(gradient):
            check_tensor(gradient, f"backward_parameter_gradient:{name}")
            return gradient
        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(forward_hook(name or "<model>")))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            handles.append(parameter.register_hook(gradient_hook(name)))

    try:
        # Trainer supplies the installed version's AdamW parameter groups,
        # beta/epsilon defaults and sampler. Never call Trainer.train/evaluate.
        with tempfile.TemporaryDirectory(prefix="cds1092-deberta-diagnostic-") as temporary:
            total_steps = math.ceil(len(data) / config["batch_size"]) * config["max_epochs"]
            warmup_steps = math.ceil(total_steps * config["warmup_ratio"])
            training_args = TrainingArguments(
                output_dir=temporary, num_train_epochs=config["max_epochs"],
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["batch_size"],
                learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
                warmup_steps=warmup_steps, lr_scheduler_type="linear", optim="adamw_torch",
                gradient_accumulation_steps=1, seed=config["random_seed"],
                data_seed=config["random_seed"], report_to="none",
                save_strategy="no", eval_strategy="no", logging_strategy="no",
            )
            if training_args.fp16 or training_args.bf16:
                raise RuntimeError("This manual diagnostic does not support mixed-precision scaling.")
            trainer = Trainer(model=model, args=training_args, train_dataset=data,
                              processing_class=tokenizer,
                              data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
            trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
            optimizer, scheduler = trainer.optimizer, trainer.lr_scheduler
            print("Optimizer:", type(optimizer).__name__, "defaults:", optimizer.defaults, flush=True)
            print("Full-run schedule:", total_steps, "steps; warmup:", warmup_steps,
                  "clip max norm:", training_args.max_grad_norm,
                  "parameter dtypes:", sorted({str(p.dtype) for p in model.parameters()}), flush=True)
            print("Diagnostic disables only saving/evaluation/log filtering; raw loss is checked.", flush=True)
            iterator = iter(trainer.get_train_dataloader())
            batch = trainer._prepare_inputs(next(iterator))
            check_model()  # Device transfer must not hide an initial failure.
            state["operation"] = "initial eval forward"
            model.eval()
            with torch.no_grad():
                output = model(**batch)
            check_tensor(output.logits, "initial logits")
            check_tensor(output.loss, "initial loss")
            print("Initial logits range:", output.logits.min().item(), output.logits.max().item(),
                  "initial loss:", output.loss.item(), flush=True)
            del output
            model.train()
            for step in range(1, args.steps + 1):
                state.update(step=step, operation="batch preparation")
                if step > 1:
                    batch = trainer._prepare_inputs(next(iterator))
                check_nested(batch, "batch")
                optimizer.zero_grad(set_to_none=True)
                state["operation"] = "training forward / loss"
                context = torch.autograd.detect_anomaly(check_nan=True) if args.anomaly else nullcontext()
                with context:
                    output = model(**batch)
                    check_tensor(output.logits, "logits")
                    check_tensor(output.loss, "raw loss")
                    loss = float(output.loss.detach().item())
                    state["operation"] = "backward"
                    output.loss.backward()
                check_gradients()
                state["operation"] = "gradient norm / clipping"
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                check_tensor(norm, "pre-clip total gradient norm")
                check_gradients()
                lr = optimizer.param_groups[0]["lr"]
                state["operation"] = "AdamW optimizer update"
                optimizer.step()
                check_model()
                for name, parameter in model.named_parameters():
                    check_nested(optimizer.state.get(parameter, {}), f"optimizer_state:{name}")
                state["operation"] = "linear scheduler update"
                scheduler.step()
                if not all(math.isfinite(group["lr"]) for group in optimizer.param_groups):
                    raise NonFiniteError(f"step={step}: non-finite scheduler learning rate")
                print(f"PASS step={step} raw_loss={loss:.9g} grad_norm={norm.item():.9g} "
                      f"lr_used={lr:.9g} parameters/gradients/AdamW-state=finite", flush=True)
                del output
            print(f"No non-finite values observed in {args.steps} diagnostic steps. "
                  "This does not establish stability later in training.", flush=True)
    except Exception:
        print(f"Diagnostic stopped: step={state['step']} operation={state['operation']}. "
              "See the exception/first non-finite tensor above. No experiment files saved.", flush=True)
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
