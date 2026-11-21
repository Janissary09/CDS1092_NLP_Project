"""Run Llama 3.3 70B Instruct zero-shot classification on one complete benchmark."""

import argparse
from contextlib import closing, contextmanager, nullcontext
from copy import copy
import hashlib
from importlib.metadata import distribution, version
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getsourcelines
import json
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_PATH = ROOT / "models/llama-3.3-70b-instruct"
ARCHITECTURE = "LlamaForCausalLM"
SMOKE_QUESTION = "A basket contains 4 apples. Two apples are added. How many apples are in the basket?"

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


# Job 1808 diagnostic only. Observe the installed 5.22.2 loop without patching it.
@contextmanager
def trace_greedy_steps(torch, sample, input_length, steps, limit=4):
    source, first_line = getsourcelines(sample)
    anchors = (
        "input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)",
        "unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)",
    )
    lines = [[first_line + i for i, line in enumerate(source) if line.strip() == anchor]
             for anchor in anchors]
    if any(len(matches) != 1 for matches in lines) or lines[1][0] != lines[0][0] + 1:
        raise RuntimeError("Unsupported installed Transformers _sample source for the Job 1808 trace")
    before, after = (matches[0] for matches in lines)
    code = sample.__code__

    def capture(frame, event, _arg):
        if event != "line" or frame.f_lineno not in (before, after):
            return capture
        state = frame.f_locals
        ids = state["input_ids"]
        torch.cuda.synchronize(ids.device)
        if frame.f_lineno == before:
            scores = state["next_token_scores"][0].cpu()
            steps.append({
                "step": len(steps) + 1,
                "scores_device": str(state["next_token_scores"].device),
                "post_processor_cpu_argmax": scores.argmax().item(),
                "unfinished": bool(state["unfinished_sequences"][0].item()),
                "cuda_selected_after_padding": state["next_tokens"][0].item(),
                "suffix_before_cat": ids[0, input_length:].tolist(),
            })
        else:
            steps[-1]["suffix_after_cat"] = ids[0, input_length:].tolist()
            if len(steps) == limit:
                frame.f_trace = None
                return None
        return capture

    def trace(frame, event, _arg):
        return capture if event == "call" and frame.f_code is code else None

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        yield
    finally:
        sys.settrace(previous)


def first_trace_divergence(steps, final_ids, pad):
    previous = []
    for step in steps:
        number = step["step"]
        selected = step["cuda_selected_after_padding"]
        expected = step["post_processor_cpu_argmax"] if step["unfinished"] else pad
        if step["suffix_before_cat"] != previous:
            return {"step": number, "stage": "before_cat_changed_prior_suffix"}
        if selected != expected:
            return {"step": number, "stage": "cuda_selection_or_padding"}
        appended = step["suffix_after_cat"]
        if appended != previous + [selected]:
            return {"step": number, "stage": "cat"}
        previous = appended
    if final_ids[:len(previous)] != previous:
        index = next(i for i, value in enumerate(previous) if final_ids[i:i + 1] != [value])
        return {"step": index + 1, "stage": "after_last_traced_cat_or_readback"}
    return None


def diagnose_generation(torch, model, tokenizer, inputs, generation, eos_ids, pad, vocab_ids, vocab_width):
    input_length = inputs["input_ids"].shape[-1]

    def run(structured, *, tracing=False):
        config = copy(generation)
        config.return_dict_in_generate = structured
        config.output_logits = structured
        steps = []
        context = (trace_greedy_steps(torch, model._sample, input_length, steps)
                   if tracing else nullcontext())
        try:
            with context:
                output = model.generate(**{key: value.clone() for key, value in inputs.items()},
                                        generation_config=config)
        except Exception:
            if tracing:
                log_generation("llama_job1808_trace_interrupted", {
                    "mode": "structured" if structured else "plain", "steps": steps,
                })
            raise
        sequences = output.sequences if structured else output
        if tracing:
            torch.cuda.synchronize(sequences.device)
        # Match the original smoke-test readback for both untraced baselines.
        ids = sequences[0, input_length:].tolist()
        invalid = [index for index, token_id in enumerate(ids)
                   if type(token_id) is not int or not 0 <= token_id < vocab_width
                   or token_id not in vocab_ids]
        report = {
            "mode": "structured" if structured else "plain", "traced": tracing,
            "final_generated_token_ids": ids, "invalid_token_positions": invalid,
            "sequence_device": str(sequences.device), "sequence_dtype": str(sequences.dtype),
        }
        if structured:
            report["cache_class"] = type(output.past_key_values).__name__
        if tracing:
            if not steps:
                raise RuntimeError("The installed Transformers greedy loop was not traced")
            report["steps"] = steps
            report["final_sequence"] = sequences[0].tolist()
            report["first_divergence"] = first_trace_divergence(steps, ids, pad)
            log_generation("llama_job1808_trace", report)
        else:
            log_generation("llama_job1808_baseline", report)
        return ids, invalid

    structured = run(True)
    plain = run(False)
    log_generation("llama_job1808_comparison", {"sequences_match": structured[0] == plain[0]})
    for mode, result in ((True, structured), (False, plain)):
        if result[1]:
            run(mode, tracing=True)
    if structured[1] or plain[1] or structured[0] != plain[0]:
        return 1
    ids = structured[0]
    ended = bool(ids and ids[-1] in eos_ids)
    raw = tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False,
                           clean_up_tokenization_spaces=False)
    prediction = shared.parse_label(raw)
    hit_limit = len(ids) == shared.MAX_NEW_TOKENS and not ended
    log_generation("llama_job1808_prediction", {
        "prediction": prediction, "raw_response": raw, "hit_token_limit": hit_limit,
    })
    return int(prediction != "ANSWERABLE" or hit_limit)


def load_generator(torch, transformers, device_map, *, diagnostics=False, trace_corruption=False):
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
        if trace_corruption:
            if model.config._attn_implementation != "sdpa":
                raise RuntimeError("The Job 1808 diagnostic requires SDPA")
            diagnostic_config = copy(generation)
            diagnostic_config.cache_implementation = "dynamic"
            log_generation("llama_job1808_settings", {
                "attention": "sdpa", "cache_implementation": "dynamic",
                "input_device": str(input_device), "lm_head_device": str(output_weights.device),
                "lm_head_vocab_size": output_weights.shape[0], "pad_token_id": pad,
                "eos_token_ids": sorted(eos_ids), "max_new_tokens": shared.MAX_NEW_TOKENS,
            })
            return diagnose_generation(torch, model, tokenizer, inputs, diagnostic_config, eos_ids,
                                       pad, vocab_ids, output_weights.shape[0])
        input_length = inputs["input_ids"].shape[-1]
        options = {"return_dict_in_generate": True, "output_logits": True} if diagnostics else {}
        output = model.generate(**inputs, generation_config=generation, **options)
        sequences = output.sequences if diagnostics else output
        ids = sequences[0, input_length:].tolist()
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
        if diagnostics or invalid_positions:
            log_generation("llama_before_decode", details)
        nonfinite = []
        if diagnostics:
            argmax_ids = []
            for step, logits in enumerate(output.logits):
                values = logits[0].float().cpu()
                if torch.isfinite(values).all().item():
                    argmax_ids.append(values.argmax().item())
                else:
                    nonfinite.append(step)
                    argmax_ids.append(None)
            log_generation("llama_generation_logits", {
                "raw_logits_cpu_argmax_ids": argmax_ids, "nonfinite_logit_steps": nonfinite,
                "cache_class": type(output.past_key_values).__name__,
            })
        if invalid_positions:
            raise RuntimeError("Llama generated token IDs outside the tokenizer vocabulary or LM head range")
        if nonfinite:
            raise RuntimeError("Llama generated non-finite logits; see the per-step diagnostics")
        try:
            raw = tokenizer.decode(ids[:-1] if terminated else ids,
                                   skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except OverflowError:
            log_generation("llama_decode_overflow", details)
            raise
        return raw, ids, hit_limit

    return generate


def smoke_test(torch, transformers, gpus, device_map):
    for index in range(len(gpus)):
        torch.cuda.reset_peak_memory_stats(index)
    generate = load_generator(torch, transformers, device_map, diagnostics=True)
    for index in range(len(gpus)):
        torch.cuda.synchronize(index)
    load_peaks = [torch.cuda.max_memory_allocated(index) for index in range(len(gpus))]
    raw, ids, hit_limit = generate(shared.PROMPT.format(question=SMOKE_QUESTION))
    prediction = shared.parse_label(raw)
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
        "prediction": prediction, "raw_response": raw,
        "generated_token_ids": ids, "hit_token_limit": hit_limit,
    }, indent=2))
    return int(prediction != "ANSWERABLE" or hit_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--diagnose-generation", action="store_true")
    parser.add_argument("--run-id", choices=["run_01"])
    args = parser.parse_args()
    if (args.dataset is None) == (args.run_id is not None):
        parser.error("Use a smoke mode alone or --dataset with --run-id run_01")

    import torch
    import transformers

    if args.diagnose_generation and transformers.__version__ != "5.22.2":
        raise RuntimeError("The Job 1808 diagnostic requires Transformers 5.22.2")

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
    if args.smoke_test or args.diagnose_generation:
        print(json.dumps({
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "transformers_distribution_url": transformers_distribution_url(),
            "accelerate_version": accelerate_version,
            "tokenizers_version": version("tokenizers"), "safetensors_version": version("safetensors"),
            "device_map": device_map,
        }, indent=2), file=sys.stderr)
        if args.diagnose_generation:
            generate = load_generator(torch, transformers, device_map, trace_corruption=True)
            return generate(shared.PROMPT.format(question=SMOKE_QUESTION))
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
