"""Reproduce a pinned, native BackendBench correctness-component comparison.

Install torch, anthropic, requests and tenacity in the probe environment. No model
client is constructed. Clone BackendBench and check out REFERENCE_COMMIT, then run:
  python audit/probe_backendbench.py --reference /path/to/BackendBench --output result.json

This invokes unchanged upstream functions, not a reimplementation or a full-system
benchmark. No performance comparison or aggregate repository ranking is produced.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from kernel_envolve.arrays import Mismatch, assert_close
from kernel_envolve.cli import evaluate
from kernel_envolve.storage import evaluator_hash

REFERENCE_COMMIT = "2a8d7e19e2c13c789ff2dceb44883ea8ea04dab4"
EVAL_HASH = "81a565616da4da14bd3f06276a60ff1f54842146e1d35dce2fcddd51ba0cf48f"
FACTORY = """import torch
def make_inputs(case, seed):
    g = torch.Generator().manual_seed(seed)
    # Exactly represented positive integer values in float32, far from tolerance edges.
    x = torch.randint(1, 9, (32,), generator=g).float()
    y = torch.randint(1, 9, (32,), generator=g).float()
    if case.get('strided'):
        x, y = x[::2], y[::2]
    return x, y
"""

# Contract truth comes from the stated operation, not either evaluator's verdict.
CASES = [
    ("valid_add", "return x + y", "return x + y", [], True),
    ("valid_strided_add", "return x + y", "return x + y", [], True),
    ("valid_tuple", "return (x + y, x - y)", "return (x + y, x - y)", [], True),
    ("wrong_value", "return x + y", "return x + y + 2", [], False),
    ("wrong_shape", "return x + y", "return (x + y)[:1]", [], False),
    ("wrong_dtype", "return x + y", "return (x + y).double()", [], False),
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
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
    if commit.strip() != REFERENCE_COMMIT or sha(root / "BackendBench/eval.py") != EVAL_HASH:
        raise ValueError("Reference must match the reviewed pinned commit and evaluator hash")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise ValueError("Use a clean upstream checkout")
    sys.path.insert(0, str(root))
    from BackendBench.eval import allclose, eval_correctness

    # Retain diagnostic fields in JSON without duplicating expected failures on stderr.
    logging.getLogger("BackendBench.eval").setLevel(logging.CRITICAL)
    rows = []
    with tempfile.TemporaryDirectory(prefix="ke-reference-probe-") as folder:
        for name, reference_body, candidate_body, mutable, expected in CASES:
            case_args = {"strided": name == "valid_strided_add"}
            task_source = (
                FACTORY + f"\ndef reference(x, y):\n    {reference_body}\nbaseline = reference\n"
            )
            candidate_source = f"def run(x, y):\n    {candidate_body}\n"
            namespace = {}
            exec(compile(task_source + candidate_source, name, "exec"), namespace)
            factory = namespace["make_inputs"]
            # The native evaluator's test protocol only consumes args and kwargs.
            # Construct fresh fixture inputs without importing unrelated suite loaders.
            tests = [
                SimpleNamespace(args=factory(case_args, seed), kwargs={}) for seed in [0, 1, 2]
            ]
            fraction, native_results = eval_correctness(
                namespace["reference"], namespace["run"], tests
            )
            task = Path(folder) / name
            task.mkdir()
            cfg = {
                "name": name,
                "objective": "Single-case correctness probe; no timing or performance claim",
                "device": "cpu",
                "rtol": 0.01,
                "atol": 0.01,
                "equal_nan": True,
                "mutable_inputs": mutable,
                "quick_seeds": [0, 1, 2],
                "cases": [{"id": "case", "args": case_args, "weight": 1}],
            }
            (task / "task.py").write_text(task_source)
            (task / "candidate.py").write_text(candidate_source)
            (task / "task.json").write_text(json.dumps(cfg))
            _, result = evaluate(task, check_only=True)
            if result["execution"] != "completed":
                raise RuntimeError(f"Probe setup error, not a correctness verdict: {result}")
            local_pass = result["correctness"] == "pass"
            if local_pass != expected:
                raise AssertionError(f"kernel_envolve violated the stated contract: {name}")
            rows.append(
                {
                    "name": name,
                    "expected_contract_pass": expected,
                    "task_source": task_source,
                    "candidate_source": candidate_source,
                    "task_config": cfg,
                    "backendbench_pass": fraction == 1.0,
                    "backendbench_results": [dataclasses.asdict(r) for r in native_results],
                    "kernel_envolve_pass": local_pass,
                    "kernel_envolve_result": result,
                }
            )
    tolerance_rows = []
    for name, offset, atol, rtol, expected in [
        ("strict_equal", 0.0, 0.0, 0.0, True),
        ("strict_small_error", 0.005, 0.0, 0.0, False),
        ("loose_allowed_error", 0.05, 0.1, 0.0, True),
        ("loose_excess_error", 0.2, 0.1, 0.0, False),
        ("default_allowed_error", 0.005, 0.01, 0.01, True),
        ("default_excess_error", 0.2, 0.01, 0.01, False),
    ]:
        ref, actual = torch.tensor([1.0]), torch.tensor([1.0 + offset])
        local_pass = True
        try:
            assert_close(actual, ref, {"atol": atol, "rtol": rtol, "equal_nan": True})
        except Mismatch:
            local_pass = False
        if local_pass != expected:
            raise AssertionError(f"Incorrect local tolerance decision: {name}")
        tolerance_rows.append(
            {
                "name": name,
                "reference_values": ref.tolist(),
                "candidate_values": actual.tolist(),
                "atol": atol,
                "rtol": rtol,
                "expected_contract_pass": expected,
                "backendbench_pass": allclose(ref, actual, atol=atol, rtol=rtol),
                "kernel_envolve_pass": local_pass,
            }
        )
    loaded = {}
    for name, module in sys.modules.items():
        if name.startswith("BackendBench") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            loaded[str(path.relative_to(root))] = sha(path)
    receipt = {
        "scope": (
            "Native unchanged BackendBench.eval correctness components on CPU, not its full "
            "CLI/backend workflow. kernel_envolve uses real isolated check-only evaluations. "
            "Both use float32 inputs and matching default tolerances/equal_nan policy. "
            "Base seeds match; kernel_envolve additionally checks reuse and input state. "
            "Input immutability is an extra enforced contract, not a claim that BackendBench "
            "advertises a mutation gate. No timing or agent-search ranking."
        ),
        "reference_commit": REFERENCE_COMMIT,
        "reference_url": f"https://github.com/meta-pytorch/BackendBench/tree/{REFERENCE_COMMIT}",
        "reference_loaded_sources_sha256": loaded,
        "probe_sha256": sha(Path(__file__)),
        "evaluator_hash": evaluator_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "numpy", "anthropic", "requests", "tenacity", "kernel-envolve"]
        },
        "native_correctness_cases": rows,
        "public_tolerance_helper_cases": tolerance_rows,
    }
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    for row in rows + tolerance_rows:
        print(
            row["name"],
            f"expected={row['expected_contract_pass']}",
            f"BackendBench={row['backendbench_pass']}",
            f"kernel_envolve={row['kernel_envolve_pass']}",
        )


if __name__ == "__main__":
    main()
