"""Run standard Microsoft Phi-4 (14B) zero-shot classification on one complete benchmark."""

import argparse
from contextlib import closing
import hashlib
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/phi-4"
MODEL_PATH = ROOT / "models/phi-4"
ARCHITECTURE = "Phi3ForCausalLM"
SMOKE_PROBES = (
    ("A basket contains 4 apples. Two apples are added. How many apples are in the basket?",
     "ANSWERABLE"),
    ("A basket contains some apples. Two apples are added. How many apples are in the basket?",
     "UNANSWERABLE"),
)

# Reuse the benchmark protocol without changing the frozen Qwen entry point.
SHARED_PATH = Path(__file__).with_name("25_evaluate_qwen.py")
spec = spec_from_file_location("cds1092_zero_shot_protocol", SHARED_PATH)
shared = module_from_spec(spec)
spec.loader.exec_module(shared)


def output_path(root, dataset, run_id):
    return root / "results/llm/phi-4" / dataset / run_id


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
    if config.get("model_type") != "phi3" or ARCHITECTURE not in config.get("architectures", []):
        raise ValueError(f"Expected the Hugging Face Phi-4 configuration in {config_path}")
    if config.get("max_position_embeddings") != 16384:
        raise ValueError(f"Expected standard Phi-4's 16K checkpoint in {config_path}")
    if config.get("quantization_config") is not None:
        raise ValueError("Expected an unquantized Phi-4 checkpoint")
    index_path = MODEL_PATH / "model.safetensors.index.json"
    shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
    if not shards or any(not (MODEL_PATH / shard).is_file() for shard in shards):
        raise FileNotFoundError(f"Incomplete Hugging Face safetensors download under {MODEL_PATH}")
    return config


def load_generator(torch, transformers):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("The local Phi-4 tokenizer is missing its native chat template")
    # Standard Phi-4 renders only the supplied roles. Reasoning Plus inserts a system message.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "template check"}], add_generation_prompt=True, tokenize=False,
    )
    if rendered != "<|im_start|>user<|im_sep|>template check<|im_end|><|im_start|>assistant<|im_sep|>":
        raise ValueError("Expected the standard Phi-4 native ChatML template")

    model, loading_info = transformers.Phi3ForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": 0}, use_safetensors=True, output_loading_info=True,
    )
    problems = {key: details for key, details in loading_info.items() if details}
    if problems:
        raise RuntimeError(f"Phi-4 checkpoint load was incomplete or incompatible: {problems}")
    model.eval()
    device = torch.device("cuda:0")
    if (getattr(model, "is_quantized", False)
            or any(p.device != device or p.dtype != torch.bfloat16 for p in model.parameters())):
        raise RuntimeError("Expected unquantized BF16 inference on cuda:0")

    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if not eos_ids or tokenizer.eos_token_id not in eos_ids:
        raise ValueError("Phi-4 generation configuration is missing the tokenizer's end-of-turn EOS")
    generation = transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=shared.MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )

    @torch.inference_mode()
    def generate(prompt):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        input_length = inputs["input_ids"].shape[-1]
        ids = model.generate(**inputs, generation_config=generation)[0, input_length:].tolist()
        terminated = bool(ids and ids[-1] in eos_ids)
        raw = tokenizer.decode(ids[:-1] if terminated else ids,
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return raw, ids, len(ids) == shared.MAX_NEW_TOKENS and not terminated

    return generate


def smoke_test(torch, generate, gpu, load_peak):
    responses = []
    for question, expected in SMOKE_PROBES:
        raw, ids, hit_limit = generate(shared.PROMPT.format(question=question))
        prediction = shared.parse_label(raw)
        passed = bool(ids and prediction == expected and not hit_limit)
        responses.append({
            "expected": expected, "prediction": prediction, "raw_response": raw,
            "generated_token_ids": ids, "hit_token_limit": hit_limit,
            "passed": passed, "status": "PASS" if passed else "FAIL",
        })

    torch.cuda.synchronize()
    passed = all(response["passed"] for response in responses)
    print(json.dumps({
        "model_id": MODEL_ID, "gpu": gpu.name, "max_new_tokens": shared.MAX_NEW_TOKENS,
        "load_peak_allocated_gib": load_peak / 1024**3,
        "combined_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "combined_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "responses": responses, "passed": passed,
    }, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", choices=["run_01"])
    parser.add_argument("--allow-failed-smoke", action="store_true",
                        help="Continue a dataset evaluation if either smoke probe fails")
    args = parser.parse_args()
    if (args.dataset is None) == (args.run_id is not None):
        parser.error("Use --smoke-test alone or --dataset with --run-id run_01")
    if args.smoke_test and args.allow_failed_smoke:
        parser.error("--allow-failed-smoke requires --dataset")

    import torch
    import transformers

    if not hasattr(transformers, ARCHITECTURE):
        raise RuntimeError(f"Transformers must provide {ARCHITECTURE}")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if ("A100" not in gpu.name or gpu.total_memory < 75 * 1024**3
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"Expected BF16 on an A100 80 GB, found {gpu.name}")

    config = load_config()
    torch.cuda.reset_peak_memory_stats()
    generate = load_generator(torch, transformers)
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    smoke_status = smoke_test(torch, generate, gpu, load_peak)
    if args.smoke_test or (smoke_status and not args.allow_failed_smoke):
        return smoke_status
    if smoke_status:
        print("Smoke test failed. Continuing benchmark evaluation with --allow-failed-smoke.",
              file=sys.stderr, flush=True)

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
        "prompt": shared.PROMPT,
        "chat_template": "Standard Phi-4 native ChatML template, one unchanged benchmark user message",
        "model_api": ARCHITECTURE, "tokenizer_api": "AutoTokenizer",
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
                shared.evaluate(connection, rows, generate, lambda: bool(received_signal))
            if not shared.export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
