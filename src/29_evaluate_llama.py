"""Run Llama 3.3 70B Instruct zero-shot classification on one complete benchmark."""

import argparse
from contextlib import ExitStack, closing
import hashlib
from heapq import nlargest
from importlib.metadata import distribution, version
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_PATH = ROOT / "models/llama-3.3-70b-instruct"
ARCHITECTURE = "LlamaForCausalLM"
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
    return root / "results/llm/llama-3.3-70b-instruct" / dataset / run_id


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
    if config.get("model_type") != "llama" or ARCHITECTURE not in config.get("architectures", []):
        raise ValueError(f"Expected the Hugging Face Llama 3.3 configuration in {config_path}")
    if config.get("quantization_config") is not None:
        raise ValueError("Expected an unquantized Llama checkpoint")
    if config.get("num_hidden_layers") != 80 or config.get("tie_word_embeddings"):
        raise ValueError("Expected the 80-layer Llama 3.3 70B layout with untied embeddings")
    index_path = MODEL_PATH / "model.safetensors.index.json"
    shards = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
    if not shards or any(not (MODEL_PATH / shard).is_file() for shard in shards):
        raise FileNotFoundError(f"Incomplete Hugging Face safetensors download under {MODEL_PATH}")
    return config


def gpu_device_map(layer_count):
    # Keep contiguous decoder blocks on each GPU with embeddings and head on opposite ends.
    return {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        **{f"model.layers.{index}": int(index >= layer_count // 2) for index in range(layer_count)},
        "model.norm": 1,
        "lm_head": 1,
    }


def log_generation(event, details):
    print(json.dumps({"event": event, **details}), file=sys.stderr, flush=True)


class InvalidGeneratedTokens(RuntimeError):
    def __init__(self, ids, hit_limit):
        super().__init__("Llama generated token IDs outside the tokenizer vocabulary or LM head range")
        self.ids = ids
        self.hit_limit = hit_limit


def synchronize_both_gpus(torch):
    for index in (0, 1):
        torch.cuda.synchronize(index)


def inspect_forward_logits(torch, model, inputs):
    # Match generate's final-position projection without entering its decoding loop.
    logits = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True).logits
    synchronize_both_gpus(torch)
    final = logits[0, -1].float()
    cuda_argmax = final.argmax()
    synchronize_both_gpus(torch)
    cpu_logits = final.cpu()
    values = cpu_logits.tolist()
    finite = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    top = nlargest(10, finite, key=lambda item: item[1])
    return {
        "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype),
        "logits_device": str(logits.device), "final_position_shape": list(final.shape),
        "attention_implementation": model.config._attn_implementation,
        "nan_count": sum(math.isnan(value) for value in values),
        "inf_count": sum(math.isinf(value) for value in values),
        "finite_min": min((value for _, value in finite), default=None),
        "finite_max": top[0][1] if top else None,
        "top_10_finite": [{"token_id": index, "score": score} for index, score in top],
        "cpu_argmax_id": cpu_logits.argmax().item(), "cuda_argmax_id": cuda_argmax.item(),
    }


def activation_values(tensor):
    return tensor.detach().cpu().float().flatten().tolist()


def activation_statistics(tensor, *, values=None):
    if values is None:
        values = activation_values(tensor)
    finite = [value for value in values if math.isfinite(value)]
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
        "nan_count": sum(math.isnan(value) for value in values),
        "inf_count": sum(math.isinf(value) for value in values),
        "zero_count": values.count(0.0),
        "finite_min": min(finite, default=None), "finite_max": max(finite, default=None),
        "finite_abs_mean": sum(map(abs, finite)) / len(finite) if finite else None,
    }


def inspect_layer_activations(torch, model, inputs):
    """Job 1811 diagnostic only. Retain selected outputs for post-forward inspection."""
    decoder = model.model
    checkpoints = (
        ("model.embed_tokens", decoder.embed_tokens),
        *((f"model.layers.{index}", decoder.layers[index]) for index in (0, 39, 40, 79)),
        ("model.norm", decoder.norm),
        ("lm_head", model.lm_head),
    )
    activations = {}

    def capture(name):
        def hook(_module, _args, output):
            activations[name] = output[0] if isinstance(output, tuple) else output
        return hook

    with ExitStack() as hooks:
        for name, module in checkpoints:
            hooks.callback(module.register_forward_hook(capture(name)).remove)
        # Keep the same final-position head projection as the Job 1811 forward probe.
        model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)

    synchronize_both_gpus(torch)
    for name, _ in checkpoints:
        if name not in activations:
            raise RuntimeError(f"Missing activation from {name}")
        log_generation("llama_layer_activation", {"module": name, **activation_statistics(activations[name])})

    # Scan the entire parameter tensors on their resident devices; transfer counts only.
    for name, module in checkpoints:
        for suffix, weight in module.named_parameters():
            size = weight.numel()
            nonzero = torch.count_nonzero(weight).item()
            log_generation("llama_layer_parameter", {
                "parameter": f"{name}.{suffix}", "shape": list(weight.shape),
                "dtype": str(weight.dtype), "device": str(weight.device),
                "zero_count": size - nonzero, "all_zero": size > 0 and nonzero == 0,
            })


def inspect_layer_boundary(torch, model, inputs):
    """Job 1812 diagnostic. Observe the dispatched block without changing its hooks."""
    layer39, layer40 = model.model.layers[39], model.model.layers[40]
    captured = {}

    def record(name):
        def hook(_module, _args, value):
            captured[name] = value[0] if isinstance(value, tuple) else value
        return hook

    def record_input(name):
        def hook(_module, args):
            captured[name] = args[0]
        return hook

    with ExitStack() as hooks:
        # A pre-hook on the block itself would run before Accelerate's forward wrapper.
        # Its first RMSNorm sees the same hidden states as the aligned input residual.
        for module, name in (
            (layer40.input_layernorm, "layer40_input"),
            (layer40.post_attention_layernorm, "layer40_post_attention_residual"),
        ):
            hooks.callback(module.register_forward_pre_hook(record_input(name)).remove)
        for module, name in (
            (layer39, "layer39_output"), (layer40, "layer40_output"),
            (layer40.self_attn, "layer40_attention_output"), (layer40.mlp, "layer40_mlp_output"),
        ):
            hooks.callback(module.register_forward_hook(record(name)).remove)
        model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)

    synchronize_both_gpus(torch)
    names = ("layer39_output", "layer40_input", "layer40_output")
    snapshots = {}
    for name in names:
        values = snapshots[name] = activation_values(captured[name])
        log_generation("llama_boundary_activation", {
            "checkpoint": name, **activation_statistics(captured[name], values=values),
        })

    source, destination = (snapshots[name] for name in names[:2])
    source_tensor, destination_tensor = (captured[name] for name in names[:2])
    same_shape = source_tensor.shape == destination_tensor.shape
    log_generation("llama_boundary_comparison", {
        "expected_devices": str(source_tensor.device) == "cuda:0" and str(destination_tensor.device) == "cuda:1",
        "same_shape": same_shape,
        "exactly_equal": same_shape and source == destination,
        "max_abs_difference_finite_pairs": max(
            (abs(a - b) for a, b in zip(source, destination) if math.isfinite(a) and math.isfinite(b)),
            default=None,
        ) if same_shape else None,
    })
    output = snapshots["layer40_output"]
    if any(value != 0.0 and math.isfinite(value) for value in destination) and output and not any(output):
        for name in ("layer40_attention_output", "layer40_post_attention_residual", "layer40_mlp_output"):
            log_generation("llama_boundary_activation", {
                "checkpoint": name, **activation_statistics(captured[name]),
            })


def load_generator(torch, transformers, device_map, *, forward_diagnostic=False,
                   activation_diagnostic=False, boundary_diagnostic=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("The local Llama tokenizer is missing its native chat template")
    special_tokens = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007,
        "<|eom_id|>": 128008,
        "<|eot_id|>": 128009,
    }
    if tokenizer.convert_tokens_to_ids(list(special_tokens)) != list(special_tokens.values()):
        raise ValueError("Unexpected Llama 3 special-token IDs in the local tokenizer")
    model, loading_info = transformers.LlamaForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.bfloat16,
        device_map=device_map, use_safetensors=True, output_loading_info=True,
    )
    problems = {
        key: loading_info[key]
        for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")
        if loading_info.get(key)
    }
    if problems:
        raise RuntimeError(f"Llama checkpoint load was incomplete or incompatible: {problems}")
    model.eval()
    devices = {torch.device("cuda:0"), torch.device("cuda:1")}
    parameters = list(model.parameters())
    if (getattr(model, "is_quantized", False)
            or {p.device for p in parameters} != devices
            or any(p.dtype != torch.bfloat16 for p in parameters)):
        raise RuntimeError("Expected all unquantized BF16 weights on exactly two CUDA GPUs")
    # Preserve the checkpoint's EOS IDs, including the native end-of-turn token.
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if special_tokens["<|eot_id|>"] not in eos_ids:
        raise ValueError("Llama generation configuration is missing the native end-of-turn EOS token")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    input_device = model.get_input_embeddings().weight.device
    output_weights = model.get_output_embeddings().weight
    vocab_ids = set(tokenizer.get_vocab().values())
    vocab_bounds = {
        "tokenizer_vocab_base_size": tokenizer.vocab_size,
        "tokenizer_vocab_id_count": len(vocab_ids),
        "tokenizer_min_id": min(vocab_ids),
        "tokenizer_max_id": max(vocab_ids),
        "model_config_vocab_size": model.config.vocab_size,
        "lm_head_vocab_size": output_weights.shape[0],
    }
    generation = transformers.GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=shared.MAX_NEW_TOKENS,
        eos_token_id=eos, pad_token_id=pad, use_cache=True,
    )

    @torch.inference_mode()
    def generate(prompt):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(input_device)
        if boundary_diagnostic:
            inspect_layer_boundary(torch, model, inputs)
            return
        if activation_diagnostic:
            inspect_layer_activations(torch, model, inputs)
            return
        input_length = inputs["input_ids"].shape[-1]
        if forward_diagnostic:
            forward = inspect_forward_logits(torch, model, inputs)
            log_generation("llama_direct_forward", forward)
        sequences = model.generate(**inputs, generation_config=generation)
        if forward_diagnostic:
            synchronize_both_gpus(torch)
        ids = sequences[0, input_length:].tolist()
        if forward_diagnostic:
            log_generation("llama_plain_generation_comparison", {
                "generated_token_ids": ids, "first_generated_id": ids[0] if ids else None,
                "matches_forward_cpu_argmax": bool(ids and ids[0] == forward["cpu_argmax_id"]),
                "matches_forward_cuda_argmax": bool(ids and ids[0] == forward["cuda_argmax_id"]),
                "max_new_tokens": shared.MAX_NEW_TOKENS,
            })
        outside_uint32 = [
            index for index, token_id in enumerate(ids)
            if type(token_id) is int and not 0 <= token_id < 2**32
        ]
        invalid_positions = [
            index for index, token_id in enumerate(ids)
            if type(token_id) is not int or token_id not in vocab_ids
            or not 0 <= token_id < min(output_weights.shape[0], 2**32)
        ]
        terminated = bool(ids and type(ids[-1]) is int and ids[-1] in eos_ids)
        hit_limit = len(ids) == shared.MAX_NEW_TOKENS and not terminated
        details = {
            **vocab_bounds, "generated_token_ids": ids, "invalid_token_positions": invalid_positions,
            "outside_uint32_positions": outside_uint32,
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_is_fast": tokenizer.is_fast,
            "input_token_count": input_length, "input_device": str(input_device),
            "lm_head_device": str(output_weights.device),
            "sequence_dtype": str(sequences.dtype), "sequence_device": str(sequences.device),
            "attention_implementation": model.config._attn_implementation,
            "do_sample": False, "num_beams": 1, "use_cache": True,
            "max_new_tokens": shared.MAX_NEW_TOKENS, "hit_token_limit": hit_limit,
            "eos_token_ids": sorted(eos_ids), "pad_token_id": pad, "ended_with_eos": terminated,
        }
        if invalid_positions:
            log_generation("llama_before_decode", details)
            raise InvalidGeneratedTokens(ids, hit_limit)
        try:
            raw = tokenizer.decode(ids[:-1] if terminated else ids,
                                   skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except OverflowError:
            log_generation("llama_decode_overflow", details)
            raise
        return raw, ids, hit_limit

    return generate


def forward_diagnostic_test(torch, transformers, device_map):
    generate = load_generator(torch, transformers, device_map, forward_diagnostic=True)
    question, expected = SMOKE_PROBES[0]
    raw, ids, hit_limit = generate(shared.PROMPT.format(question=question))
    prediction = shared.parse_label(raw)
    log_generation("llama_forward_diagnostic_prediction", {
        "raw_response": raw, "prediction": prediction, "expected": expected,
        "hit_token_limit": hit_limit,
    })
    return int(not ids or prediction != expected or hit_limit)


def smoke_test(torch, transformers, gpus, device_map):
    for index in range(len(gpus)):
        torch.cuda.reset_peak_memory_stats(index)
    generate = load_generator(torch, transformers, device_map)
    for index in range(len(gpus)):
        torch.cuda.synchronize(index)
    load_peaks = [torch.cuda.max_memory_allocated(index) for index in range(len(gpus))]
    responses = []
    for question, expected in SMOKE_PROBES:
        try:
            raw, ids, hit_limit = generate(shared.PROMPT.format(question=question))
            prediction = shared.parse_label(raw)
            error = None
        except InvalidGeneratedTokens as exc:
            raw, ids, hit_limit = None, exc.ids, exc.hit_limit
            prediction, error = shared.INVALID, str(exc)
        responses.append({
            "expected": expected, "prediction": prediction, "raw_response": raw,
            "generated_token_ids": ids, "hit_token_limit": hit_limit,
            "passed": bool(ids and prediction == expected and not hit_limit),
            "error": error,
        })

    passed = all(response["passed"] for response in responses)
    for index in range(len(gpus)):
        torch.cuda.synchronize(index)
    print(json.dumps({
        "gpus": [
            {"device": index, "name": gpu.name,
             "load_peak_allocated_gib": load_peaks[index] / 1024**3,
             "combined_peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 1024**3,
             "combined_peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 1024**3}
            for index, gpu in enumerate(gpus)
        ],
        "generation_mode": "plain_tensor", "max_new_tokens": shared.MAX_NEW_TOKENS,
        "responses": responses, "passed": passed,
    }, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--diagnose-forward", action="store_true")
    mode.add_argument("--diagnose-activations", action="store_true")
    mode.add_argument("--diagnose-boundary", action="store_true")
    parser.add_argument("--run-id", choices=["run_01"])
    args = parser.parse_args()
    if (args.dataset is None) == (args.run_id is not None):
        parser.error("Use a smoke or diagnostic mode alone, or --dataset with --run-id run_01")

    import torch
    import transformers

    if not hasattr(transformers, ARCHITECTURE):
        raise RuntimeError(f"Transformers must provide {ARCHITECTURE}")

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Expected exactly two visible CUDA GPUs")
    gpus = [torch.cuda.get_device_properties(index) for index in range(2)]
    for index, gpu in enumerate(gpus):
        with torch.cuda.device(index):
            if ("A100" not in gpu.name or gpu.total_memory < 75 * 1024**3
                    or not torch.cuda.is_bf16_supported()):
                raise RuntimeError(f"Expected BF16 on an A100 80 GB at cuda:{index}, found {gpu.name}")

    accelerate_version = version("accelerate")
    config = load_config()
    device_map = gpu_device_map(config["num_hidden_layers"])
    if args.dataset is None:
        print(json.dumps({
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "transformers_distribution_url": transformers_distribution_url(),
            "accelerate_version": accelerate_version,
            "tokenizers_version": version("tokenizers"), "safetensors_version": version("safetensors"),
            "device_map": device_map,
        }, indent=2), file=sys.stderr)
        if args.diagnose_activations or args.diagnose_boundary:
            inspect_activations = load_generator(
                torch, transformers, device_map, activation_diagnostic=args.diagnose_activations,
                boundary_diagnostic=args.diagnose_boundary,
            )
            inspect_activations(shared.PROMPT.format(question=SMOKE_PROBES[0][0]))
            return 0
        if args.diagnose_forward:
            return forward_diagnostic_test(torch, transformers, device_map)
        return smoke_test(torch, transformers, gpus, device_map)

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
        "prompt": shared.PROMPT, "chat_template": "Llama tokenizer native chat template, one user message",
        "model_api": ARCHITECTURE, "tokenizer_api": "AutoTokenizer",
        "dtype": "bfloat16", "quantization": None, "gpus": [gpu.name for gpu in gpus],
        "device_map": device_map, "accelerate_version": accelerate_version,
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
                generate = load_generator(torch, transformers, device_map)
                shared.evaluate(connection, rows, generate, lambda: bool(received_signal))
            if not shared.export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
