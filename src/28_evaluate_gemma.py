"""Run Gemma 3 27B IT zero-shot classification on one complete benchmark."""

import argparse
from contextlib import closing
import hashlib
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import signal


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/gemma-3-27b-it"
MODEL_PATH = ROOT / "models/gemma-3-27b-it"
ARCHITECTURE = "Gemma3ForConditionalGeneration"

# Reuse the benchmark protocol without changing the frozen Qwen entry point.
SHARED_PATH = Path(__file__).with_name("25_evaluate_qwen.py")
spec = spec_from_file_location("cds1092_zero_shot_protocol", SHARED_PATH)
shared = module_from_spec(spec)
spec.loader.exec_module(shared)


def output_path(root, dataset, run_id):
    return root / "results/llm/gemma-3-27b-it" / dataset / run_id


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def transformers_distribution_url():
    metadata = distribution("transformers").read_text("direct_url.json")
    if metadata is None:
        return None
    source = json.loads(metadata)
    return {"vcs_info": source.get("vcs_info")}


def load_config():
    config_path = MODEL_PATH / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "gemma3" or ARCHITECTURE not in config.get("architectures", []):
        raise ValueError(f"Expected the Hugging Face Gemma 3 configuration in {config_path}")
    index_path = MODEL_PATH / "model.safetensors.index.json"
    shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
    if not shards or any(not (MODEL_PATH / shard).is_file() for shard in shards):
        raise FileNotFoundError(f"Incomplete Hugging Face safetensors download under {MODEL_PATH}")
    return config


def load_generator(torch, transformers):
    processor = transformers.AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, use_safetensors=True,
    ).eval()
    device = torch.device("cuda:0")
    if (getattr(model, "is_quantized", False)
            or any(p.device != device or p.dtype != torch.bfloat16
                   for p in model.parameters() if p.is_floating_point())):
        raise RuntimeError("Expected unquantized BF16 inference on cuda:0")
    # Preserve the checkpoint's EOS IDs, including Gemma's end-of-turn token.
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if not eos_ids:
        raise ValueError("Gemma generation configuration has no EOS tokens")
    generation = transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=shared.MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id, use_cache=True,
    )

    @torch.inference_mode()
    def generate(prompt):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        input_length = inputs["input_ids"].shape[-1]
        ids = model.generate(**inputs, generation_config=generation)[0, input_length:].tolist()
        terminated = bool(ids and ids[-1] in eos_ids)
        raw = processor.decode(ids[:-1] if terminated else ids,
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return raw, ids, len(ids) == shared.MAX_NEW_TOKENS and not terminated

    return generate


def smoke_test(torch, transformers, gpu):
    torch.cuda.reset_peak_memory_stats()
    generate = load_generator(torch, transformers)
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    question = "A basket contains 4 apples. Two apples are added. How many apples are in the basket?"
    raw, ids, hit_limit = generate(shared.PROMPT.format(question=question))
    torch.cuda.synchronize()
    print(json.dumps({
        "gpu": gpu.name,
        "load_peak_allocated_gib": load_peak / 1024**3,
        "combined_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "combined_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "prediction": shared.parse_label(raw), "raw_response": raw,
        "generated_token_ids": ids, "hit_token_limit": hit_limit,
    }, indent=2))
    if not ids:
        raise RuntimeError("Smoke test produced no tokens")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", choices=["run_01"])
    args = parser.parse_args()
    if args.smoke_test == (args.run_id is not None):
        parser.error("Use --smoke-test alone or --dataset with --run-id run_01")

    import torch
    import transformers

    if not hasattr(transformers, ARCHITECTURE):
        raise RuntimeError(f"Transformers must provide {ARCHITECTURE}")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if ("A100" not in gpu.name
            or gpu.total_memory < 75 * 1024**3 or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"Expected BF16 on an A100 80 GB, found {gpu.name}")

    config = load_config()
    if args.smoke_test:
        return smoke_test(torch, transformers, gpu)

    rows, dataset_hash = shared.load_dataset(ROOT, args.dataset)
    output = output_path(ROOT, args.dataset, args.run_id)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset, "dataset_sha256": dataset_hash, "row_count": len(rows),
        "run_id": args.run_id, "model_id": MODEL_ID, "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_type": config["model_type"], "model_architectures": config["architectures"],
        "model_sha256": shared.model_fingerprint(MODEL_PATH, output / "model_fingerprint_cache.json"),
        "evaluator_sha256": source_digest(Path(__file__)),
        "shared_evaluator_sha256": source_digest(SHARED_PATH),
        "prompt": shared.PROMPT, "chat_template": "Gemma processor native chat template, one text-only user message",
        "model_api": ARCHITECTURE, "processor_api": "AutoProcessor",
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
                generate = load_generator(torch, transformers)
                shared.evaluate(connection, rows, generate, lambda: bool(received_signal))
            if not shared.export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
