"""Evaluate Meta Llama 3.1 8B Instruct on one fixed zero-shot benchmark."""

import argparse
from contextlib import closing
import hashlib
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = ROOT / "models/Llama-3.1-8B-Instruct"
ARCHITECTURE = "LlamaForCausalLM"
DEVICE_MAP = {"": 0}
END_OF_TURN_ID = 128009
SMOKE_PROBES = (
    ("A basket contains 4 apples. Two apples are added. How many apples are in the basket?",
     "ANSWERABLE"),
    ("A basket contains some apples. Two apples are added. How many apples are in the basket?",
     "UNANSWERABLE"),
)

SHARED_PATH = Path(__file__).with_name("25_evaluate_qwen.py")
spec = spec_from_file_location("cds1092_zero_shot_protocol", SHARED_PATH)
shared = module_from_spec(spec)
spec.loader.exec_module(shared)


def output_path(root, dataset, run_id):
    return root / "results/llm/llama-3.1-8b-instruct" / dataset / run_id


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_config():
    config_path = MODEL_PATH / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("model_type") != "llama"
            or ARCHITECTURE not in config.get("architectures", [])
            or config.get("num_hidden_layers") != 32
            or config.get("hidden_size") != 4096):
        raise ValueError(f"Expected the Llama 3.1 8B configuration in {config_path}")
    if config.get("quantization_config") is not None:
        raise ValueError("Expected an unquantized Llama checkpoint")

    index = MODEL_PATH / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    if len(shards) != 4 or any(
        Path(shard).name != shard or not shard.endswith(".safetensors")
        or not (MODEL_PATH / shard).is_file() for shard in shards
    ):
        raise FileNotFoundError(f"Expected four indexed safetensors shards under {MODEL_PATH}")
    return config


def require_gpu(torch):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    with torch.cuda.device(0):
        if ("A100" not in gpu.name or gpu.total_memory < 75 * 1024**3
                or not torch.cuda.is_bf16_supported()):
            raise RuntimeError(f"Expected BF16 support on an A100 80 GB, found {gpu.name}")
    return gpu


def load_generator(torch, transformers):
    device = torch.device("cuda:0")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("The local Llama tokenizer is missing its native chat template")
    if tokenizer.convert_tokens_to_ids("<|eot_id|>") != END_OF_TURN_ID:
        raise ValueError("Unexpected Llama end-of-turn token ID")

    model, loading_info = transformers.LlamaForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, device_map=DEVICE_MAP,
        use_safetensors=True, output_loading_info=True,
    )
    problems = {key: loading_info[key]
                for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")
                if loading_info.get(key)}
    if problems:
        raise RuntimeError(f"Llama checkpoint load was incomplete or incompatible: {problems}")
    parameters = list(model.parameters())
    if (getattr(model, "is_quantized", False) or not parameters
            or any(p.device != device or p.dtype != torch.bfloat16 for p in parameters)
            or any(buffer.device != device for buffer in model.buffers())):
        raise RuntimeError("Expected all unquantized BF16 weights and model buffers on cuda:0")
    model.eval()

    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if END_OF_TURN_ID not in eos_ids:
        raise ValueError("Llama generation configuration is missing the native end-of-turn EOS token")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    vocab_ids = set(tokenizer.get_vocab().values())
    head_size = model.get_output_embeddings().weight.shape[0]
    generation = transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=shared.MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=pad, use_cache=True,
    )

    @torch.inference_mode()
    def generate(prompt):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        input_length = inputs["input_ids"].shape[-1]
        sequences = model.generate(**inputs, generation_config=generation)
        ids = sequences[0, input_length:].tolist()
        if any(type(token) is not int or token not in vocab_ids or not 0 <= token < head_size
               for token in ids):
            raise RuntimeError(f"Llama generated token IDs outside the tokenizer or LM head: {ids}")
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
        responses.append({
            "expected": expected, "prediction": prediction, "raw_response": raw,
            "generated_token_ids": ids, "hit_token_limit": hit_limit,
            "passed": bool(ids and prediction == expected and not hit_limit),
        })
    torch.cuda.synchronize(0)
    passed = all(response["passed"] for response in responses)
    print(json.dumps({
        "model_id": MODEL_ID, "gpu": gpu.name, "device": "cuda:0", "dtype": "bfloat16",
        "load_peak_allocated_gib": load_peak / 1024**3,
        "combined_peak_allocated_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
        "max_new_tokens": shared.MAX_NEW_TOKENS, "responses": responses, "passed": passed,
    }, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", choices=["run_01"])
    parser.add_argument("--allow-failed-smoke", action="store_true",
                        help="Continue a dataset evaluation after a completed smoke test fails")
    args = parser.parse_args()
    if (args.dataset is None) == (args.run_id is not None):
        parser.error("Use --smoke-test alone, or --dataset with --run-id run_01")
    if args.smoke_test and args.allow_failed_smoke:
        parser.error("--allow-failed-smoke requires --dataset")

    import torch
    import transformers

    if not hasattr(transformers, ARCHITECTURE):
        raise RuntimeError(f"Transformers must provide {ARCHITECTURE}")
    gpu = require_gpu(torch)
    config = load_config()
    torch.cuda.reset_peak_memory_stats(0)
    generate = load_generator(torch, transformers)
    torch.cuda.synchronize(0)
    load_peak = torch.cuda.max_memory_allocated(0)
    smoke_status = smoke_test(torch, generate, gpu, load_peak)
    if args.smoke_test or (smoke_status != 0 and not args.allow_failed_smoke):
        return smoke_status
    if smoke_status != 0:
        print("Smoke test failed. Continuing benchmark evaluation with --allow-failed-smoke.",
              file=sys.stderr, flush=True)

    rows, dataset_hash = shared.load_dataset(ROOT, args.dataset)
    output = output_path(ROOT, args.dataset, args.run_id)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset, "dataset_sha256": dataset_hash, "row_count": len(rows),
        "run_id": args.run_id, "model_id": MODEL_ID, "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_type": config["model_type"], "model_architectures": config["architectures"],
        "model_sha256": shared.model_fingerprint(MODEL_PATH, output / "model_fingerprint_cache.json"),
        "evaluator_sha256": source_digest(Path(__file__)), "shared_evaluator_sha256": source_digest(SHARED_PATH),
        "prompt": shared.PROMPT, "chat_template": "Llama tokenizer native chat template, one user message",
        "model_api": ARCHITECTURE, "tokenizer_api": "AutoTokenizer", "dtype": "bfloat16",
        "quantization": None, "device_map": DEVICE_MAP, "cpu_offload": False, "gpu": gpu.name,
        "accelerate_version": version("accelerate"), "tokenizers_version": version("tokenizers"),
        "requested_temperature": 0, "do_sample": False, "num_beams": 1,
        "max_new_tokens": shared.MAX_NEW_TOKENS, "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "invalid_output_policy": shared.INVALID_POLICY,
        # Keep variable GPU memory measurements in the smoke log so resumes can match the manifest.
        "smoke_test_passed": smoke_status == 0,
        "allow_failed_smoke": args.allow_failed_smoke,
        "failed_smoke_override_applied": smoke_status != 0 and args.allow_failed_smoke,
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
