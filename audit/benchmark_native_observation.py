"""Measure discarded-observation copies; not a kernel or end-to-end GPU benchmark."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from kernel_envolve.training import Session, Workload


def benchmark(device, mib, rounds):
    torch.set_num_threads(1)
    x = torch.ones(4, device=device)
    state = torch.zeros(mib * 1024**2 // 16, device=device)
    output = torch.ones(mib * 1024**2 // 4, device=device)
    gradients = output.clone()
    values = {"output": output, "gradients": (gradients,)}
    validated = []
    session = Session(
        Workload(
            {"x": x, "state": state},
            lambda: values,
            mutable=("state",),
            validate=lambda result: validated.append(True),
        ),
        device,
        device_guard_limit=16,
    )

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    samples = {"capture_every_iteration": [], "capture_final_iteration": []}
    # Same 3 warmup + 5 measured observations as default native timing; paired order.
    for round_index in range(rounds + 1):
        keys = list(samples)
        if round_index % 2:
            keys.reverse()
        for key in keys:
            sync()
            started = time.perf_counter()
            for index in range(8):
                observed = session.observe(
                    values, capture=key == "capture_every_iteration" or index == 7
                )
            sync()
            elapsed = time.perf_counter() - started
            assert torch.equal(observed["output"]["output"], output.cpu())
            assert torch.equal(observed["output"]["gradients"][0], gradients.cpu())
            assert torch.equal(observed["state"]["state"], state.cpu())
            if round_index:
                samples[key].append(elapsed)
            del observed
    bytes_per_capture = sum(t.numel() * t.element_size() for t in (output, gradients, state))
    return {
        "scope": "Observation policy comparison only; excludes kernels, reset, snapshot, IPC and compilation",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "rounds": rounds,
        "observations_per_measure_command": 8,
        "raw_seconds": samples,
        "median_seconds": {key: statistics.median(value) for key, value in samples.items()},
        "output_and_state_bytes_copied_per_command": {
            "capture_every_iteration": bytes_per_capture * 8,
            "capture_final_iteration": bytes_per_capture,
        },
        "validator_calls": len(validated),
        "state_checks": session.input_check_calls,
        "checks_retained": len(validated) == (rounds + 1) * 16
        and session.input_check_calls == (rounds + 1) * 32,
        "final_output_and_state_equal": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mib", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mib < 1 or args.rounds < 1:
        parser.error("mib and rounds must be positive")
    result = benchmark(args.device, args.mib, args.rounds)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
