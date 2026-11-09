"""One isolated process per evaluation; a failed CUDA context is never reused."""

import importlib.metadata
import platform
import sys
import time
import traceback
from pathlib import Path

from .arrays import (
    Mismatch,
    aliases,
    assert_close,
    capture,
    copy_values,
    signature,
    unchanged,
    validate_inputs,
)
from .scoring import plan, score
from .storage import config, digest, load_module, read_json, verify_source, write_json


class Clock:
    def __init__(self, device):
        self.torch = None
        if device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            torch.cuda.set_device(0 if device == "cuda" else device)
            self.torch = torch
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def sync(self):
        if self.torch:
            self.torch.cuda.synchronize()

    def measure(self, function, inputs):
        self.sync()
        if self.torch:
            self.start.record()
            output = function(*inputs)
            self.end.record()
            self.end.synchronize()
            elapsed = self.start.elapsed_time(self.end) * 1000
        else:
            start = time.perf_counter_ns()
            output = function(*inputs)
            elapsed = (time.perf_counter_ns() - start) / 1000
        # Keep the returned storage alive until timing/synchronization finishes.
        del output
        return elapsed


def environment(clock):
    packages = {}
    for name in ("numpy", "torch", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "timer": "cuda_events_current_stream" if clock.torch else "perf_counter_ns",
    }
    if clock.torch:
        t = clock.torch
        device = t.cuda.current_device()
        props = t.cuda.get_device_properties(device)
        env["gpu"] = {
            "name": props.name,
            "device": device,
            "capability": list(t.cuda.get_device_capability()),
            "total_memory": props.total_memory,
            "cuda_runtime": t.version.cuda,
        }
    return env


def execute(run):
    request = read_json(run / "request.json")
    result = {
        "schema": 1,
        "mode": request["mode"],
        "signal_only": request["signal_only"],
        "execution": "error",
        "correctness": "not_run",
        "checks": [],
        "samples": [],
        "identity": request["identity"],
        "comparator": request["comparator"],
        "initial_best_hash": request["initial_best_hash"],
        "task_root": request["task_root"],
    }
    cfg = config(run / "source")
    try:
        verify_source(run, request["identity"])
        clock = Clock(cfg["device"])
        result["environment"] = environment(clock)
        task = load_module(run / "source", "task")
        candidate = load_module(run / "source", "candidate")
        functions = {"candidate": candidate.run}
        if not request["signal_only"]:
            if request["comparator"]["kind"] == "baseline":
                functions["comparator"] = task.baseline
            else:
                old_run = Path(request["comparator"]["run"])
                previous = read_json(old_run / "result.json")
                if digest(previous) != request["comparator"]["result_hash"]:
                    raise ValueError("Comparator result changed")
                if previous["identity"]["contract"] != request["identity"]["contract"]:
                    raise ValueError("Comparator task contract differs")
                verify_source(old_run, previous["identity"])
                functions["comparator"] = load_module(old_run / "source", "candidate").run
        cases, seeds, trials, batches = plan(cfg, request["mode"])

        def inputs(case, seed):
            value = task.make_inputs(case.get("args", {}), seed)
            validate_inputs(value, cfg["device"])
            if any(i >= len(value) for i in cfg["mutable_inputs"]):
                raise ValueError("mutable_inputs index out of range")
            return value

        def check(case, seed, role, kind, args, expected_seed):
            row = {"case": case["id"], "seed": seed, "kind": kind, "role": role, "status": "fail"}
            result["checks"].append(row)
            reference_args = inputs(case, expected_seed)
            if len(args) != len(reference_args) or aliases(args) != aliases(reference_args):
                raise ValueError("Input factory is not deterministic in structure/aliasing")
            for a, b in zip(args, reference_args):
                if signature(a) != signature(b):
                    raise ValueError(
                        "Input factory is not deterministic in shape/layout/dtype/device"
                    )
                assert_close(
                    a, b, {"rtol": 0, "atol": 0, "equal_nan": True}, "deterministic input factory"
                )
            expected = task.reference(*reference_args)
            clock.sync()
            before = capture(args)
            actual = functions[role](*args)
            clock.sync()
            assert_close(actual, expected, cfg)
            unchanged(args, before, cfg["mutable_inputs"])
            for i in cfg["mutable_inputs"]:
                assert_close(args[i], reference_args[i], cfg, f"mutated input[{i}]")
            row["status"] = "pass"

        result["correctness"] = "partial"
        for case in cases:
            for seed in seeds:
                for role in functions:
                    args = inputs(case, seed)
                    check(case, seed, role, "fresh", args, seed)
                    # Keep object identities but change all values; catches cached outputs.
                    sentinel_seed = seed + 1_000_003
                    replacement = inputs(case, sentinel_seed)
                    copy_values(args, replacement)
                    check(case, seed, role, "reuse", args, sentinel_seed)
        result["correctness"] = "pass"

        for batch in range(batches):
            # Reverse case order in confirmation to reduce monotonic run-order bias.
            for case in cases if batch == 0 else reversed(cases):
                for role, function in functions.items():
                    for warm in range(cfg["warmup"]):
                        args = inputs(case, seeds[warm % len(seeds)])
                        function(*args)
                        clock.sync()
                for trial in range(trials):
                    seed = seeds[trial % len(seeds)]
                    if request["signal_only"]:
                        order = ["candidate", "candidate"]
                    elif (batch + trial) % 2:
                        order = ["candidate", "comparator", "comparator", "candidate"]
                    else:
                        order = ["comparator", "candidate", "candidate", "comparator"]
                    repetitions = {role: 0 for role in functions}
                    for role in order:
                        # Factory/reset and compilation warmup are outside the timed scope.
                        args = inputs(case, seed)
                        elapsed = clock.measure(functions[role], args)
                        result["samples"].append(
                            {
                                "case": case["id"],
                                "batch": batch,
                                "trial": trial,
                                "seed": seed,
                                "role": role,
                                "repeat": repetitions[role],
                                "us": elapsed,
                            }
                        )
                        repetitions[role] += 1
        verify_source(run, request["identity"])
        result["execution"] = "completed"
        result["decision"] = score(result, cfg)
    except Mismatch as exc:
        result.update(
            execution="completed",
            correctness="fail",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        result["decision"] = score(result, cfg)
    except Exception as exc:
        result["execution"] = "error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["decision"] = {
            "eligible_for_best": False,
            "performance": "not_measured",
            "reasons": [str(exc)],
        }
        traceback.print_exc()
    write_json(run / "result.json", result)


if __name__ == "__main__":
    execute(Path(sys.argv[1]).resolve())
