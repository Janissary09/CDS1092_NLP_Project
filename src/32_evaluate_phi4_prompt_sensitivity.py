"""Run the standard Phi-4 punctuation sensitivity experiment."""
import argparse
from contextlib import closing
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import signal
import sys


BASE_PATH = Path(__file__).with_name("31_evaluate_phi4_standard.py")
spec = spec_from_file_location("cds1092_phi4_standard", BASE_PATH)
base = module_from_spec(spec)
spec.loader.exec_module(base)

RUN_ID = "run_02"
shared = base.shared
ORIGINAL_LINE = "\nANSWERABLE or UNANSWERABLE.\n"
EXPERIMENT_LINE = "\nANSWERABLE or UNANSWERABLE\n"
if shared.PROMPT.count(ORIGINAL_LINE) != 1:
    raise ValueError("Expected exactly one original Phi-4 label instruction line")
PROMPT = shared.PROMPT.replace(ORIGINAL_LINE, EXPERIMENT_LINE, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=shared.DATASETS)
    mode.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", choices=[RUN_ID])
    parser.add_argument("--allow-failed-smoke", action="store_true",
                        help="Continue a dataset evaluation if either smoke probe fails")
    args = parser.parse_args()
    if (args.dataset is None) == (args.run_id is not None):
        parser.error("Use --smoke-test alone or --dataset with --run-id run_02")
    if args.smoke_test and args.allow_failed_smoke:
        parser.error("--allow-failed-smoke requires --dataset")

    import torch
    import transformers

    if not hasattr(transformers, base.ARCHITECTURE):
        raise RuntimeError(f"Transformers must provide {base.ARCHITECTURE}")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    if ("A100" not in gpu.name or gpu.total_memory < 75 * 1024**3
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"Expected BF16 on an A100 80 GB, found {gpu.name}")

    config = base.load_config()
    torch.cuda.reset_peak_memory_stats()
    generate = base.load_generator(torch, transformers)
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    def generate_run_02(prompt):
        return generate(prompt.replace(ORIGINAL_LINE, EXPERIMENT_LINE, 1))

    print(json.dumps({"run_id": RUN_ID, "prompt": PROMPT}), flush=True)
    smoke_status = base.smoke_test(torch, generate_run_02, gpu, load_peak)
    if args.smoke_test or (smoke_status and not args.allow_failed_smoke):
        return smoke_status
    if smoke_status:
        print("Smoke test failed. Continuing benchmark evaluation with --allow-failed-smoke.",
              file=sys.stderr, flush=True)

    rows, dataset_hash = shared.load_dataset(base.ROOT, args.dataset)
    output = base.output_path(base.ROOT, args.dataset, args.run_id)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset, "dataset_sha256": dataset_hash, "row_count": len(rows),
        "run_id": args.run_id, "model_id": base.MODEL_ID, "model_path": str(base.MODEL_PATH.relative_to(base.ROOT)),
        "model_type": config["model_type"], "model_architectures": config["architectures"],
        "model_sha256": shared.model_fingerprint(base.MODEL_PATH, output / "model_fingerprint_cache.json"),
        "evaluator_sha256": base.source_digest(Path(__file__)),
        "base_evaluator_sha256": base.source_digest(BASE_PATH),
        "shared_evaluator_sha256": base.source_digest(base.SHARED_PATH),
        "prompt": PROMPT,
        "chat_template": "Standard Phi-4 native ChatML template, one unchanged benchmark user message",
        "model_api": base.ARCHITECTURE, "tokenizer_api": "AutoTokenizer",
        "dtype": "bfloat16", "quantization": None, "gpu": gpu.name,
        "requested_temperature": 0, "do_sample": False, "num_beams": 1,
        "max_new_tokens": shared.MAX_NEW_TOKENS, "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__, "cuda_version": torch.version.cuda,
        "transformers_distribution_url": base.transformers_distribution_url(),
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
                shared.evaluate(connection, rows, generate_run_02, lambda: bool(received_signal))
            if not shared.export_results(connection, rows, output, manifest):
                print(f"Incomplete run. Resume using the same dataset and run ID: {output}")
                return 128 + received_signal if received_signal else 1
    print(f"Exported {len(rows)} responses to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
