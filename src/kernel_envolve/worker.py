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
from .protocol import order, plan, post_seed, roles, timing_seed
from .scoring import score
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
        return elapsed, output


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
        "schema": 2,
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
        "selection": request["selection"],
        "check_only": request["check_only"],
        "same_candidate_as_best": request["same_candidate_as_best"],
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
        if "baseline" in roles(result):
            functions["baseline"] = task.baseline
        cases, seeds, trials, batches = plan(cfg, request["mode"], request["selection"])

        def inputs(case, seed):
            value = task.make_inputs(case.get("args", {}), seed)
            validate_inputs(value, cfg["device"])
            if any(i >= len(value) for i in cfg["mutable_inputs"]):
                raise ValueError("mutable_inputs index out of range")
            return value

        def check(case, seed, role, kind, args, expected_seed, observed=None):
            row = {"case": case["id"], "seed": seed, "kind": kind, "role": role, "status": "fail"}
            result["checks"].append(row)
            reference_args = inputs(case, expected_seed)
            initial_alias = aliases(args) if observed is None else observed[1][1]
            if len(args) != len(reference_args) or initial_alias != aliases(reference_args):
                raise ValueError("Input factory is not deterministic in structure/aliasing")
            initial = args if observed is None else tuple(x[2] for x in observed[1][0])
            for i, (a, b) in enumerate(zip(initial, reference_args)):
                layout = signature(a) if observed is None else observed[1][0][i][0]
                if layout != signature(b):
                    raise ValueError(
                        "Input factory is not deterministic in shape/layout/dtype/device"
                    )
                try:
                    assert_close(
                        a,
                        b,
                        {"rtol": 0, "atol": 0, "equal_nan": True},
                        "deterministic input factory",
                    )
                except Mismatch as exc:
                    raise ValueError("Input factory is not deterministic in values") from exc
            expected = task.reference(*reference_args)
            clock.sync()
            validator = getattr(task, "validate", None)
            if validator:
                try:
                    if validator(reference_args, expected) is not None:
                        raise ValueError("task.validate must return None or raise AssertionError")
                except AssertionError as exc:
                    raise ValueError(f"Oracle violates task.validate: {exc}") from exc
            if observed is None:
                before = capture(args)
                actual = functions[role](*args)
                clock.sync()
            else:
                actual, before = observed
            assert_close(actual, expected, cfg)
            unchanged(args, before, cfg["mutable_inputs"])
            for i in cfg["mutable_inputs"]:
                assert_close(args[i], reference_args[i], cfg, f"mutated input[{i}]")
            if validator:
                try:
                    if validator(args, actual) is not None:
                        raise ValueError("task.validate must return None or raise AssertionError")
                except AssertionError as exc:
                    raise Mismatch(f"task.validate: {exc}") from exc
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

        if request["check_only"]:
            result["execution"] = "completed"
            result["decision"] = score(result, cfg)
            verify_source(run, request["identity"])
            write_json(run / "result.json", result)
            return

        phase_start = time.perf_counter()
        result["measurement_wall_seconds"] = 0.0
        for batch in range(batches):
            # Reverse case order in confirmation to reduce monotonic run-order bias.
            for case in cases if batch == 0 else reversed(cases):
                for role, function in functions.items():
                    for warm in range(cfg["warmup"]):
                        args = inputs(case, seeds[warm % len(seeds)])
                        function(*args)
                        clock.sync()
                last_args = {}
                last_outputs = {}
                for trial in range(trials):
                    repetitions = {role: 0 for role in functions}
                    for role in order(list(functions), batch, trial):
                        seed = timing_seed(case["id"], batch, trial, repetitions[role], seeds)
                        # Factory/reset and compilation warmup are outside the timed scope.
                        args = inputs(case, seed)
                        last_args[role] = args
                        observe = (
                            batch == batches - 1 and trial == trials - 1 and repetitions[role] == 1
                        )
                        before = capture(args) if observe else None
                        elapsed, output = clock.measure(functions[role], args)
                        if observe:
                            last_outputs[role] = (output, before)
                        del output
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
                if batch == batches - 1:
                    # Check actual timed outputs, without rerunning the candidate to manufacture a verdict.
                    seed = timing_seed(case["id"], batch, trials - 1, 1, seeds)
                    for role, args in last_args.items():
                        check(case, seed, role, "timed_output", args, seed, last_outputs[role])
                    # Validate after timing with the last measured objects but new values.
                    seed = post_seed(case["id"], seeds)
                    for role, args in last_args.items():
                        copy_values(args, inputs(case, seed))
                        check(case, seed, role, "post", args, seed)
        result["measurement_wall_seconds"] = time.perf_counter() - phase_start
        verify_source(run, request["identity"])
        if request["comparator"]["kind"] == "best" and not request["signal_only"]:
            if digest(read_json(old_run / "result.json")) != request["comparator"]["result_hash"]:
                raise ValueError("Comparator result changed during evaluation")
            verify_source(old_run, previous["identity"])
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
