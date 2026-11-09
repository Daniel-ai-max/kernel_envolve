"""Run unchanged AKO4ALL and kernel_envolve CLIs on one idle CUDA GPU.

python audit/probe_ako4all.py --reference /path/to/pinned/bench.py --output result.json

Uses native protocols, not matched cache semantics or a kernel-speed ranking.
Recorded sample counts align for cost probes; correctness work/warmup differ.
No model calls. Each invocation uses a new process, and all runs are sequential.
"""

import argparse
import base64
import gzip
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REFERENCE_COMMIT = "bbd0e1cf1ce2fb19d4322a932680f4c3d175d80e"
REFERENCE_HASH = "424d31220040d53c98da8d2bba85b78fd91078374b5f6bc28f2c3fb89cd5a5b8"
FACTORY = """import torch
def draw(n, strided, generator=None):
    size = n * 2 if strided else n
    values = [torch.randint(1, 9, (size,), device='cuda', generator=generator).float()
              for _ in range(2)]
    return tuple(x[::2] if strided else x for x in values)
"""
CASES = [
    ("valid_add", "return x + y", "return x + y", [], True),
    ("valid_strided_add", "return x + y", "return x + y", [], True),
    ("wrong_value", "return x + y", "return x + y + 2", [], False),
    ("wrong_shape", "return x + y", "return (x + y)[:1]", [], False),
    ("valid_input_alias", "return x", "return x", [], True),
    ("corrupted_input_alias", "return x", "return x.add_(2)", [], False),
    (
        "correct_output_but_input_mutated",
        "return x + y",
        "result = x + y; x.zero_(); return result",
        [],
        False,
    ),
    ("valid_inplace", "return x.add_(y)", "return x.add_(y)", [0], True),
    ("wrong_inplace", "return x.add_(y)", "return x.add_(2 * y)", [0], False),
    (
        "cached_identity",
        "return x + y",
        "for saved_x, value in CACHE:\n        if saved_x is x: return value\n"
        "    value = x + y\n    CACHE.append((x, value))\n    return value",
        [],
        False,
    ),
    (
        "late_exception",
        "return x + y",
        "COUNT[0] += 1\n    if COUNT[0] >= 10: raise RuntimeError('late probe failure')\n"
        "    return x + y",
        [],
        False,
    ),
    ("valid_sqrt_positive_domain", "return torch.sqrt(x)", "return torch.sqrt(x)", [], True),
]


def fixtures(directory, case, size, seeds):
    name, ref_body, candidate_body, mutable, expected = case
    strided = name == "valid_strided_add"
    reference = FACTORY + f"\ndef reference(x, y):\n    {ref_body}\n"
    candidate = "import torch\nCACHE = []\nCOUNT = [0]\n"
    candidate += f"def run(x, y):\n    {candidate_body}\n"
    task_source = reference + "baseline = reference\n"
    task_source += (
        "def make_inputs(case, seed):\n"
        "    g = torch.Generator(device='cuda').manual_seed(seed)\n"
        f"    return draw({size}, {strided}, g)\n"
    )
    cfg = {
        "name": name,
        "objective": "One equally weighted input case; native protocol comparison",
        "device": "cuda:0",
        "rtol": 1e-4,
        "atol": 1e-4,
        "equal_nan": False,
        "mutable_inputs": mutable,
        "quick_seeds": seeds[:1],
        "full_seeds": seeds,
        "quick_trials": 3,
        "full_trials": 10,
        "warmup": 3,
        "cases": [{"id": "case", "args": {}, "weight": 1}],
    }
    task = directory / "task"
    task.mkdir(parents=True)
    (task / "task.py").write_text(task_source)
    (task / "candidate.py").write_text(candidate)
    (task / "task.json").write_text(json.dumps(cfg))
    # Distinct helper names: the CUDA loader shares its reference/solution namespace.
    original = reference + (
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x, y): return reference(x, y)\n"
        f"def get_inputs(): return draw({size}, {strided})\n"
        "def get_init_inputs(): return []\n"
    )
    solution = candidate + (
        "class Model(torch.nn.Module):\n    def forward(self, x, y): return run(x, y)\n"
    )
    (directory / "reference.py").write_text(original)
    (directory / "solution.py").write_text(solution)
    return {
        "case": name,
        "elements_per_input": size,
        "expected_contract_pass": expected,
        "task_config": cfg,
        "task_source": task_source,
        "candidate_source": candidate,
        "ako_reference_source": original,
        "ako_solution_source": solution,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bench = args.reference.resolve()
    if hashlib.sha256(bench.read_bytes()).hexdigest() != REFERENCE_HASH:
        raise ValueError("Use the exact reviewed AKO4ALL bench.py")
    import torch

    from kernel_envolve.storage import evaluator_hash

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This probe requires exactly one visible idle CUDA GPU")
    generator = torch.Generator().manual_seed(42)
    seeds = [torch.randint(0, 2**32 - 1, (1,), generator=generator).item() for _ in range(3)]
    report = {
        "kind": "ako4all-native-cli-comparison",
        "reference_commit": REFERENCE_COMMIT,
        "reference_sha256": REFERENCE_HASH,
        "reference_url": (
            f"https://github.com/TongmingLAIC/AKO4ALL/blob/{REFERENCE_COMMIT}"
            "/bench/kernelbench/bench.py"
        ),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_hash": evaluator_hash(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "correctness_seeds": seeds,
        "protocol": {
            "scope": "Unmodified CLIs; same operation, input domain and float32 tolerance",
            "cache": "AKO4ALL flushes L2 per trial; kernel_envolve does not",
            "warmup": "AKO4ALL 10 per role; kernel_envolve 3 per role per batch",
            "sampling": "12 quick / 80 full recorded timings; upstream additionally discards first",
            "verification": "Base seeds match; reuse/mutation checks and comparator checks differ",
            "cost": "Fresh subprocess command wall time; not kernel speed or search quality",
        },
        "correctness": [],
        "cost": [],
    }
    with tempfile.TemporaryDirectory(prefix="ako-native-probe-") as temporary:
        root = Path(temporary)
        serial = 0

        def one(arm, case, size, mode):
            nonlocal serial
            serial += 1
            folder = root / str(serial)
            row = fixtures(folder, case, size, seeds)
            if arm == "ako4all":
                command = [
                    sys.executable,
                    str(bench),
                    "--ref",
                    str(folder / "reference.py"),
                    "--solution",
                    str(folder / "solution.py"),
                    "--backend",
                    "cuda",
                    "--precision",
                    "float32",
                    "--timing-method",
                    "cuda_event",
                    "--num-correct-trials",
                    "1" if mode == "quick" else "3",
                    "--num-perf-trials",
                    "6" if mode == "quick" else "40",
                    "--fresh-inputs",
                    "--verbose",
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "kernel_envolve.cli",
                    "evaluate",
                    "--task",
                    str(folder / "task"),
                    "--mode",
                    mode,
                    "--timeout",
                    "120",
                ]
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True, timeout=150)
            row.update(
                arm=arm,
                mode=mode,
                command_wall_seconds=time.perf_counter() - started,
                exit_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
            if arm == "ako4all":
                fields = {}
                for line in process.stdout.splitlines():
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        if key in {"COMPILED", "CORRECT", "RUNTIME", "REF_RUNTIME", "SPEEDUP"}:
                            fields[key] = value
                if set(fields) != {"COMPILED", "CORRECT", "RUNTIME", "REF_RUNTIME", "SPEEDUP"}:
                    raise RuntimeError(f"Incomplete upstream report: {row}")
                row["native_fields"] = fields
                row["correctness_pass"] = fields["CORRECT"] == "True"
                row["has_positive_timings"] = (
                    float(fields["RUNTIME"]) > 0 and float(fields["REF_RUNTIME"]) > 0
                )
            else:
                cli = json.loads(process.stdout)
                run = Path(cli["run"])
                row["result"] = json.loads((run / "result.json").read_text())
                row["worker_stderr"] = (run / "stderr.log").read_text()
                row["correctness_pass"] = row["result"]["correctness"] == "pass"
                row["has_positive_timings"] = row["result"]["execution"] == "completed" and bool(
                    row["result"]["samples"]
                )
            print(
                "KE_CASE "
                + json.dumps(
                    {
                        key: row[key]
                        for key in [
                            "case",
                            "arm",
                            "mode",
                            "command_wall_seconds",
                            "exit_code",
                            "correctness_pass",
                            "has_positive_timings",
                        ]
                    }
                ),
                flush=True,
            )
            return row

        report["bootstrap"] = [
            one(arm, CASES[0], 257, "quick") for arm in ["ako4all", "kernel_envolve"]
        ]
        for index, case in enumerate(CASES):
            arms = ["ako4all", "kernel_envolve"]
            for arm in arms if index % 2 == 0 else reversed(arms):
                report["correctness"].append(one(arm, case, 257, "full"))
        for repeat in range(3):
            for size in [257, 1048577] if repeat % 2 == 0 else [1048577, 257]:
                for mode in ["quick", "full"] if repeat % 2 == 0 else ["full", "quick"]:
                    arms = ["ako4all", "kernel_envolve"]
                    for arm in arms if repeat % 2 == 0 else reversed(arms):
                        row = one(arm, CASES[0], size, mode)
                        row["repeat"] = repeat
                        report["cost"].append(row)
    report["cost_medians_seconds"] = [
        {
            "arm": arm,
            "elements_per_input": size,
            "mode": mode,
            "seconds": statistics.median(
                r["command_wall_seconds"]
                for r in report["cost"]
                if (r["arm"], r["elements_per_input"], r["mode"]) == (arm, size, mode)
            ),
        }
        for size in [257, 1048577]
        for mode in ["quick", "full"]
        for arm in ["ako4all", "kernel_envolve"]
    ]
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    blob = base64.b64encode(gzip.compress(args.output.read_bytes())).decode()
    for offset in range(0, len(blob), 32000):
        print(
            "KE_REPORT_PART "
            + json.dumps(
                {
                    "index": offset // 32000,
                    "total": (len(blob) + 31999) // 32000,
                    "sha256": hashlib.sha256(blob.encode()).hexdigest(),
                    "data": blob[offset : offset + 32000],
                }
            ),
            flush=True,
        )
    print("KE_VALIDATION_DONE", flush=True)


if __name__ == "__main__":
    main()
