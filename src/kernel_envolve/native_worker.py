"""One persistent process per frozen source role. Commands arrive over stdin."""

import contextlib
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path


def serve(request_path):
    import torch

    from .runtime import state as runtime_state
    from .training import Session

    request = json.loads(Path(request_path).read_text())
    cfg, roots = request["config"], request["roots"]
    os.environ.update(cfg.get("env", {}))
    sys.path[:0] = list(roots.values())
    torch.set_num_threads(cfg.get("threads", 1))
    device = cfg["device"]
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback")
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(cfg.get("deterministic", True))
    spec = importlib.util.spec_from_file_location("native_adapter", request["adapter"])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    def build(case, seed):
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        return adapter.create(case, seed, device, request["role"] == "oracle")

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    def provenance():
        for package, root_name in cfg["imports"].items():
            module = sys.modules.get(package)
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(Path(roots[root_name])):
                raise RuntimeError(f"Wrong or unused production import: {package}: {path}")
            for name, loaded in tuple(sys.modules.items()):
                if name.startswith(package + ".") and getattr(loaded, "__file__", None):
                    if not Path(loaded.__file__).resolve().is_relative_to(Path(roots[root_name])):
                        raise RuntimeError(f"Mixed production package versions: {name}")
        return {p: str(Path(sys.modules[p].__file__).resolve()) for p in cfg["imports"]}

    for line in sys.stdin:
        command = json.loads(line)
        reply = Path(command["reply"])
        start = time.perf_counter()
        session = None
        try:
            timing_oracle = command["kind"] == "timing_oracle"
            if timing_oracle and request["role"] != "oracle":
                raise ValueError("Timing oracle observations require the oracle role")
            session = Session(
                build(command["case"], command["seed"]),
                device,
                device_guard_limit=2 * 1024**3 if device.startswith("cuda") else 0,
            )
            if command["kind"] == "check" or timing_oracle:
                observations = []
                # Timing consumes only the fresh reference output. Keep the full
                # fresh/reuse/repeat protocol for both precheck and postcheck.
                repeats = 1 if timing_oracle else (3 if cfg.get("deterministic", True) else 2)
                for index in range(repeats):
                    if index == 1:
                        session.refresh(build(command["case"], command["seed"] + 1))
                    session.reset()
                    output = session.guard.call(session.workload.run)
                    observations.append(session.observe(output))
                    del output
                torch.save(observations, command["output"])
                result = {"status": "pass"}
            else:
                samples, peak, raw_peak, private_guard_bytes = [], 0, 0, []
                for index in range(cfg["warmup"] + cfg["iterations"]):
                    session.reset()
                    sync()
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    session.guard.check()
                    before = time.perf_counter_ns()
                    output = session.workload.run()
                    sync()
                    elapsed = (time.perf_counter_ns() - before) / 1000
                    measured_peak = (
                        torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0
                    )
                    including_guards = measured_peak
                    guard_allocation = session.private_allocated_bytes
                    # Private guards exist before timing. Preserve the workload's
                    # original absolute-memory limit by excluding only their actual
                    # allocator delta; candidate allocations remain fully counted.
                    if device.startswith("cuda"):
                        measured_peak -= guard_allocation
                    # Checking and state reset are outside the measured F+B call.
                    observation = session.observe(
                        output, capture=index == cfg["warmup"] + cfg["iterations"] - 1
                    )
                    if index >= cfg["warmup"]:
                        samples.append(elapsed)
                        if device.startswith("cuda"):
                            peak = max(peak, measured_peak)
                            raw_peak = max(raw_peak, including_guards)
                            private_guard_bytes.append(guard_allocation)
                    del output
                torch.save([observation], command["output"])
                result = {
                    "status": "pass",
                    "samples_us": samples,
                    "peak_bytes": peak,
                    "peak_bytes_including_input_guards": raw_peak,
                    "private_input_guard_bytes_by_iteration": private_guard_bytes,
                    "validated_iterations": cfg["warmup"] + cfg["iterations"],
                    "captured_observations": 1,
                }
            result.update(
                input_metadata=session.specs,
                input_aliases=session.aliases,
                imports=provenance(),
                seconds=time.perf_counter() - start,
                input_guard=session.input_guard_stats(),
                runtime_settings=runtime_state()["torch"],
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                command_id=command["id"],
            )
            session = None
            gc.collect()

        except Exception as exc:
            traceback.print_exc()
            result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "command_id": command["id"],
                "seconds": time.perf_counter() - start,
                "input_guard": session.input_guard_stats() if session is not None else None,
            }
        from .storage import write_json

        write_json(reply, result)
        if result["status"] != "pass":
            return 1  # Never reuse a potentially poisoned CUDA process.
    return 0


def main():
    # All incidental package/compiler output goes to the run log, not the protocol.
    with contextlib.redirect_stdout(sys.stderr):
        return serve(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
