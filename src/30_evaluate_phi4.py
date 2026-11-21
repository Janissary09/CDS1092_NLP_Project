"""Run Phi-4-reasoning-plus (14B) zero-shot classification on one complete benchmark."""

import argparse
from contextlib import closing
from copy import copy
import hashlib
from importlib.metadata import distribution
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import signal


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "microsoft/Phi-4-reasoning-plus"
MODEL_PATH = ROOT / "models/phi-4-reasoning-plus"
ARCHITECTURE = "Phi3ForCausalLM"
SMOKE_PROBES = (
    ("A basket contains 4 apples. Two apples are added. How many apples are in the basket?",
     "ANSWERABLE"),
    ("A basket contains some apples. Two apples are added. How many apples are in the basket?",
     "UNANSWERABLE"),
)
DIAGNOSTIC_BUDGETS = (16, 64, 256, 512)

# Reuse the benchmark protocol without changing the frozen Qwen entry point.
SHARED_PATH = Path(__file__).with_name("25_evaluate_qwen.py")
spec = spec_from_file_location("cds1092_zero_shot_protocol", SHARED_PATH)
shared = module_from_spec(spec)
spec.loader.exec_module(shared)


def output_path(root, dataset, run_id):
    return root / "results/llm/phi-4-reasoning-plus" / dataset / run_id


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
    def generate(prompt, *, max_new_tokens=shared.MAX_NEW_TOKENS):
        config = generation
        if max_new_tokens != shared.MAX_NEW_TOKENS:
            config = copy(generation)
            config.max_new_tokens = max_new_tokens
        # The native template adds Microsoft's reasoning instruction.
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(device)
        input_length = inputs["input_ids"].shape[-1]
        ids = model.generate(**inputs, generation_config=config)[0, input_length:].tolist()
        terminated = bool(ids and ids[-1] in eos_ids)
        raw = tokenizer.decode(ids[:-1] if terminated else ids,
                               skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return raw, ids, len(ids) == max_new_tokens and not terminated

    return generate


def smoke_test(torch, transformers, gpu):
    torch.cuda.reset_peak_memory_stats()
    generate = load_generator(torch, transformers)
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    responses = []
    for question, expected in SMOKE_PROBES:
        raw, ids, hit_limit = generate(shared.PROMPT.format(question=question))
        prediction = shared.parse_label(raw)
        responses.append({
            "expected": expected, "prediction": prediction, "raw_response": raw,
            "generated_token_ids": ids, "hit_token_limit": hit_limit,
            "passed": bool(ids and prediction == expected and not hit_limit),
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


def extract_diagnostic_label(raw):
    """Read a standalone final label outside a completed reasoning block."""
    before, closing_tag, final = raw.rpartition("</think>")
    if "<think>" in final:
        return None
    if closing_tag and before.count("<think>") != before.count("</think>") + 1:
        return None
    lines = final.strip().splitlines()
    match = re.fullmatch(r"(?:Label:[ \t]*)?(ANSWERABLE|UNANSWERABLE)", lines[-1].strip()) if lines else None
    return match.group(1) if match else None


def reasoning_diagnostic(torch, transformers, gpu):
    generate = load_generator(torch, transformers)
    responses = []
    for question, expected in SMOKE_PROBES:
        prompt = shared.PROMPT.format(question=question)
        for budget in DIAGNOSTIC_BUDGETS:
            raw, ids, hit_limit = generate(prompt, max_new_tokens=budget)
            responses.append({
                "question": question, "expected": expected, "max_new_tokens": budget,
                "raw_response": raw, "extracted_final_label": extract_diagnostic_label(raw),
                "strict_prediction": shared.parse_label(raw), "generated_token_count": len(ids),
                "hit_token_limit": hit_limit,
            })
    print(json.dumps({
        "model_id": MODEL_ID, "gpu": gpu.name, "mode": "reasoning_diagnostic",
        "responses": responses,
    }, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--diagnose-reasoning", action="store_true")
    parser.add_argument("--run-id", choices=["run_01"])
    args = parser.parse_args()
    if (args.dataset is None) == (args.run_id is not None):
        parser.error("Use a smoke or diagnostic mode alone, or --dataset with --run-id run_01")

    import torch
    import transformers

    if not hasattr(transformers, ARCHITECTURE):
        raise RuntimeError(f"Transformers must provide {ARCHITECTURE}")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if ("A100" not in gpu.name or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"Expected BF16 on an A100, found {gpu.name}")

    config = load_config()
    if args.diagnose_reasoning:
        return reasoning_diagnostic(torch, transformers, gpu)
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
        "prompt": shared.PROMPT,
        "chat_template": ("Phi-4 native ChatML template with built-in reasoning system message, "
                          "one unchanged benchmark user message"),
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
                generate = load_generator(torch, transformers)
                shared.evaluate(connection, rows, generate, lambda: bool(received_signal))
            if not shared.export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
