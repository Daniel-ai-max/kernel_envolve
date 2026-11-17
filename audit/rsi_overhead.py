"""Measure bounded context reads only; this is not an LLM/token or GPU benchmark."""

import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

from kernel_envolve.rsi.context import for_task
from kernel_envolve.storage import digest, write_json


def measure(root, task, iterations=2000):
    times = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        for_task(root, task)
        times.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "samples": iterations,
        "median_ms": statistics.median(times),
        "p95_ms": sorted(times)[int(0.95 * len(times))],
        "max_ms": max(times),
    }


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        task = {"id": "fixture", "spec": {"identity": "fixture-contract"}}
        result = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scope": "Warm local filesystem, one advice read per agent round; not end-to-end iteration latency",
        }
        result["unconfigured"] = measure(root, task)
        (root / "rsi-context").mkdir()
        value = {
            "schema": 1,
            "scope": "demo",
            "version": "a" * 64,
            "task_identity": "fixture-contract",
            "text": "x" * 2400,
        }
        value["hash"] = digest(value)
        write_json(root / "rsi-context/fixture.json", value)
        result["enabled_max_text"] = measure(root, task)
        (root / "rsi-context/DISABLED").touch()
        result["disabled"] = measure(root, task)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
