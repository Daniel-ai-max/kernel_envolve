"""Self-test the independent final pilot validator; seeds here are public probes."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = Path(__file__).resolve().parent
    rows = []
    with tempfile.TemporaryDirectory(prefix="ke-final-validator-selftest-") as temp:
        root = Path(temp)
        seeds = root / "seeds.json"
        seeds.write_text(json.dumps([5011, 17011, 73009]))
        for task in ["row_sum", "matmul", "fused_relu"]:
            original = (audit / "search_tasks" / task / "candidate.py").read_text()
            wrapped = original.replace("def run(", "def original(")
            variants = {
                "original": original,
                "wrong-value": wrapped + "\ndef run(*args): return original(*args) + 1\n",
                "input-mutation": wrapped
                + (
                    "\ndef run(*args):\n output=original(*args)\n args[0].add_(1)\n return output\n"
                ),
                "cached-output": wrapped
                + (
                    "\ncache=None\ndef run(*args):\n global cache\n"
                    " if cache is None: cache=original(*args)\n return cache\n"
                ),
                "wrong-dtype": wrapped + "\ndef run(*args): return original(*args).double()\n",
                "wrong-shape": wrapped + "\ndef run(*args): return original(*args).unsqueeze(0)\n",
            }
            if args.measure:
                variants["original-full"] = original
            for name, source in variants.items():
                candidate = root / "candidate.py"
                candidate.write_text(source)
                output = root / "report.json"
                command = [
                    sys.executable,
                    str(audit / "search_final_acceptance.py"),
                    "--task",
                    task,
                    "--candidate",
                    str(candidate),
                    "--seeds",
                    str(seeds),
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                ]
                if name != "original-full":
                    command.append("--check-only")
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                report = json.loads(output.read_text())
                expected = name in {"original", "original-full"}
                assert result.returncode == (0 if expected else 1), (command, result.stderr, report)
                assert report["correct"] == expected, report
                if name == "original-full":
                    assert len(report["samples"]) == 240
                    assert len(report["checks"]) == 36
                rows.append(dict(task=task, probe=name, candidate_source=source, result=report))
                print(task, name, "accepted" if report["correct"] else "rejected", flush=True)
    report = {
        "scope": "Self-test only; public probe seeds are not final-search seeds",
        "device": args.device,
        "verifier_sha256": hashlib.sha256(
            (audit / "search_final_acceptance.py").read_bytes()
        ).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenarios": rows,
        "completed": True,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
