"""Run Apertus v1.5-8B zero-shot classification on one complete benchmark."""

import argparse
from contextlib import closing
import hashlib
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import signal


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
MODEL_PATH = ROOT / "models/Apertus-v1.5-8B"
SWISS_AI_TRANSFORMERS_REVISION = "3797303dda74844e3d1f8977ff5518bb91f818b4"

# Reuse the benchmark protocol without changing the frozen Qwen entry point.
SHARED_PATH = Path(__file__).with_name("25_evaluate_qwen.py")
spec = spec_from_file_location("cds1092_zero_shot_protocol", SHARED_PATH)
shared = module_from_spec(spec)
spec.loader.exec_module(shared)


def output_path(root, dataset, run_id):
    return root / "results/llm/apertus-v1.5-8b" / dataset / run_id


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def transformers_distribution_url():
    metadata = distribution("transformers").read_text("direct_url.json")
    if metadata is None:
        return None
    source = json.loads(metadata)
    return {"vcs_info": source.get("vcs_info")}


def load_config(transformers):
    config_path = MODEL_PATH / "config.json"
    model_type = json.loads(config_path.read_text(encoding="utf-8"))["model_type"]
    if model_type != "apertus1p5":
        raise ValueError(f"Expected apertus1p5 in {config_path}, found {model_type}")
    try:
        return transformers.AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    except ValueError as error:
        raise RuntimeError(
            "This Transformers build cannot load Apertus 1.5. "
            f"Use the Swiss AI Transformers integration at commit {SWISS_AI_TRANSFORMERS_REVISION}."
        ) from error


def load_generator(torch, transformers, config):
    processor = transformers.AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, config=config, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, use_safetensors=True,
    ).eval()
    device = torch.device("cuda:0")
    # Apertus keeps its inactive audio/vision tokenizers in FP32 by design.
    if (getattr(model, "is_quantized", False)
            or model.get_input_embeddings().weight.dtype != torch.bfloat16
            or any(p.device != device or p.dtype not in (torch.bfloat16, torch.float32)
                   for p in model.parameters() if p.is_floating_point())):
        raise RuntimeError("Expected unquantized BF16 text inference on cuda:0")

    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if not eos_ids:
        raise ValueError("Model generation configuration has no EOS token")
    generation = transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=shared.MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id, use_cache=True,
    )

    @torch.inference_mode()
    def generate(prompt):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        input_length = inputs["input_ids"].shape[-1]
        ids = model.generate(**inputs, generation_config=generation)[0, input_length:].tolist()
        terminated = bool(ids and ids[-1] in eos_ids)
        content = ids[:-1] if terminated else ids
        raw = processor.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return raw, ids, len(ids) == shared.MAX_NEW_TOKENS and not terminated

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=shared.DATASETS)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must start with an alphanumeric character and use only letters, digits, _ or -")

    rows, dataset_hash = shared.load_dataset(ROOT, args.dataset)
    import torch
    import transformers

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if ("A100" not in gpu.name
            or gpu.total_memory < 75 * 1024**3 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"Expected BF16 on an A100 80 GB, found {gpu.name}")

    config = load_config(transformers)
    output = output_path(ROOT, args.dataset, args.run_id)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset, "dataset_sha256": dataset_hash, "row_count": len(rows),
        "run_id": args.run_id, "model_id": MODEL_ID, "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_type": config.model_type, "model_architectures": config.architectures,
        "model_sha256": shared.model_fingerprint(MODEL_PATH, output / "model_fingerprint_cache.json"),
        "evaluator_sha256": source_digest(Path(__file__)),
        "shared_evaluator_sha256": source_digest(SHARED_PATH),
        "prompt": shared.PROMPT, "chat_template": "Apertus default, user message only, default thinking off",
        "model_api": "AutoModelForMultimodalLM", "processor_api": "AutoProcessor",
        "dtype": "bfloat16", "quantization": None, "gpu": gpu.name,
        "requested_temperature": 0, "do_sample": False, "num_beams": 1,
        "max_new_tokens": shared.MAX_NEW_TOKENS, "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "transformers_distribution_url": transformers_distribution_url(),
        "invalid_output_policy": shared.INVALID_POLICY,
    }
    received_signal = 0

    def stop(signum, _frame):
        nonlocal received_signal
        received_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with closing(shared.open_checkpoint(output / "checkpoint.sqlite3", manifest)) as connection:
        if not shared.export_results(connection, rows, output, manifest):
            if not received_signal:
                generate = load_generator(torch, transformers, config)
                shared.evaluate(connection, rows, generate, lambda: bool(received_signal))
            if not shared.export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
