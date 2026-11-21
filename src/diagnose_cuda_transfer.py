"""Compare standalone PyTorch copies on two CUDA devices without loading a model."""

import json
import math


SHAPES = ((16,), (1, 100, 8192))


def compare_values(expected, observed):
    if len(expected) != len(observed):
        return {"exactly_equal": False, "max_abs_difference": None}
    finite = all(map(math.isfinite, expected)) and all(map(math.isfinite, observed))
    return {
        "exactly_equal": expected == observed,
        "max_abs_difference": max((abs(a - b) for a, b in zip(expected, observed)), default=0.0)
        if finite else None,
    }


def synchronize(torch):
    for device in (0, 1):
        torch.cuda.synchronize(device)


def cpu_snapshot(tensor):
    # tolist materializes independent Python values even when cpu() is a no-op.
    return tensor.detach().cpu().flatten().tolist()


def check_transfer(torch, source, target, expected):
    synchronize(torch)
    before = cpu_snapshot(source)
    destination = source.to(target)
    synchronize(torch)
    observed = cpu_snapshot(destination)
    after = cpu_snapshot(source)
    seed = compare_values(expected, before)
    source_check = compare_values(before, after)
    transfer = compare_values(before, observed)
    return destination, {
        "source": str(source.device), "destination": str(destination.device),
        "shape": list(source.shape), "dtype": str(source.dtype),
        "source_vs_expected": seed, "destination_vs_source_before": transfer,
        "destination_vs_expected": compare_values(expected, observed),
        "source_after_vs_before": source_check,
        "destination_zero_count": observed.count(0),
        "passed": seed["exactly_equal"] and source_check["exactly_equal"] and transfer["exactly_equal"],
    }


def run_diagnostic(torch):
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Expected exactly two visible CUDA GPUs")
    print(json.dumps({
        "event": "cuda_transfer_environment",
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "device_names": [torch.cuda.get_device_name(i) for i in (0, 1)],
        "peer_access": {
            "cuda:0_to_cuda:1": torch.cuda.can_device_access_peer(0, 1),
            "cuda:1_to_cuda:0": torch.cuda.can_device_access_peer(1, 0),
        },
    }), flush=True)

    passed = True
    for shape in SHAPES:
        for dtype in (torch.float32, torch.bfloat16):
            # Signed odd integers are nonzero and exactly representable in both dtypes.
            indices = torch.arange(math.prod(shape), dtype=torch.int32, device="cpu")
            cpu = ((indices % 126) * 2 - 125).reshape(shape).to(dtype=dtype)
            expected = cpu_snapshot(cpu)
            gpu0, host0 = check_transfer(torch, cpu, "cuda:0", expected)
            print(json.dumps({"event": "cuda_transfer_result", **host0}), flush=True)
            gpu1, host1 = check_transfer(torch, cpu, "cuda:1", expected)
            print(json.dumps({"event": "cuda_transfer_result", **host1}), flush=True)
            _, peer01 = check_transfer(torch, gpu0, "cuda:1", expected)
            print(json.dumps({"event": "cuda_transfer_result", **peer01}), flush=True)
            _, peer10 = check_transfer(torch, gpu1, "cuda:0", expected)
            print(json.dumps({"event": "cuda_transfer_result", **peer10}), flush=True)
            passed = passed and all(result["passed"] for result in (host0, host1, peer01, peer10))

    print(json.dumps({"event": "cuda_transfer_summary", "passed": passed}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    import torch

    raise SystemExit(run_diagnostic(torch))
