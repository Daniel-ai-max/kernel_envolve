"""Self-check public development workloads before running workflow experiments.

python audit/probe_search_tasks.py --device cpu --output report.json
The CPU mode changes only device selection in disposable copies. It does not
certify CUDA arithmetic, timing, an optimized kernel, or search effectiveness.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from kernel_envolve.cli import evaluate
from kernel_envolve.storage import evaluator_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "search_tasks"
    rows = []
    for source in sorted(root.iterdir()):
        if not source.is_dir():
            continue
        with tempfile.TemporaryDirectory(prefix="ke-search-preflight-") as temp:
            task = Path(temp)
            for name in ["task.json", "task.py", "candidate.py"]:
                shutil.copyfile(source / name, task / name)
            cfg = json.loads((task / "task.json").read_text())
            cfg["device"] = args.device
            for case in cfg["cases"]:
                case["args"]["device"] = args.device
            (task / "task.json").write_text(json.dumps(cfg))
            for kind in ["original", "wrong-value", "input-mutation"]:
                if kind != "original":
                    body = (
                        (source / "candidate.py").read_text().replace("def run(", "def original(")
                    )
                    body += "\n\ndef run(*inputs):\n    output = original(*inputs)\n"
                    if kind == "wrong-value":
                        body += "    return output + 1\n"
                    else:
                        body += "    inputs[0].add_(1)\n    return output\n"
                    (task / "candidate.py").write_text(body)
                _, result = evaluate(
                    task,
                    cases=[c["id"] for c in cfg["cases"]],
                    seeds=[0, 1, 2],
                    check_only=True,
                    timeout=180,
                )
                expected = "pass" if kind == "original" else "fail"
                assert result["execution"] == "completed", result
                assert result["correctness"] == expected, result
                rows.append({"task": source.name, "probe": kind, "result": result})
            if args.measure:
                shutil.copyfile(source / "candidate.py", task / "candidate.py")
                _, result = evaluate(task, mode="full", timeout=300)
                assert result["execution"] == "completed", result
                assert result["correctness"] == "pass", result
                rows.append({"task": source.name, "probe": "original-full", "result": result})
    report = {
        "scope": "Public workload self-check, not agent search or held-out final acceptance",
        "device": args.device,
        "evaluator_hash": evaluator_hash(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".py"}
        },
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print([(r["task"], r["probe"], r["result"]["correctness"]) for r in rows])


if __name__ == "__main__":
    main()
