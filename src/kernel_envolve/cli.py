"""CLI supervision and atomic promotion; no model API or orchestration service."""

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .protocol import plan
from .scoring import score
from .storage import (
    bundle,
    candidate_identity,
    config,
    contract,
    digest,
    evaluator_hash,
    read_best,
    read_json,
    verify_source,
    write_json,
)


@contextlib.contextmanager
def lock(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another operation holds {path}; inspect it before retrying") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluate(
    task,
    mode="quick",
    signal_only=False,
    timeout=300,
    baseline=False,
    cases=None,
    seeds=None,
    check_only=False,
):
    task = Path(task).resolve()
    cfg = config(task)
    selection = {}
    if cases:
        selection["cases"] = cases
    if seeds:
        selection["seeds"] = seeds
    plan(cfg, mode, selection)
    if check_only and mode != "quick":
        raise ValueError("Check-only diagnostics require quick mode")
    if signal_only and mode != "quick":
        raise ValueError("Candidate-only timing is available in quick mode only")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    state = task / ".kernel_envolve"
    (state / "runs").mkdir(parents=True, exist_ok=True)
    with lock(state / "evaluate.lock"):
        best = read_best(task)
        name = (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        run = state / "runs" / name
        run.mkdir()
        identity = bundle(task, run / "source", cfg)
        comparator = {"kind": "baseline"}
        same_candidate = False
        if best and not baseline:
            pointer, old_run, old_result = best
            if old_result["identity"]["contract"] != identity["contract"]:
                raise ValueError(
                    "Task contract differs from best; start a separate task for a new contract"
                )
            comparator = {
                "kind": "best",
                "run": str(old_run),
                "result_hash": pointer["result_hash"],
            }
            same_candidate = candidate_identity(
                identity, config(run / "source")
            ) == candidate_identity(old_result["identity"], config(old_run / "source"))
        request = {
            "task_root": str(task),
            "mode": mode,
            "signal_only": signal_only,
            "identity": identity,
            "comparator": comparator,
            "initial_best_hash": digest(best[0]) if best else None,
            "selection": selection,
            "check_only": check_only,
            "same_candidate_as_best": same_candidate,
        }
        write_json(run / "request.json", request)
        started = time.perf_counter()
        with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "kernel_envolve.worker", str(run)],
                cwd=run,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            failure = None
            try:
                code = process.wait(timeout=timeout)
                if code:
                    failure = f"Worker exited with code {code}"
            except subprocess.TimeoutExpired:
                failure = f"Worker exceeded {timeout}s timeout"
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if failure or not (run / "result.json").exists():
            result = {
                "schema": 2,
                **request,
                "execution": "error",
                "correctness": "partial",
                "checks": [],
                "samples": [],
                "error": {"type": "WorkerError", "message": failure or "No result file"},
                "decision": {
                    "eligible_for_best": False,
                    "performance": "not_measured",
                    "reasons": [failure or "No result file"],
                },
            }
            write_json(run / "result.json", result)
        result = read_json(run / "result.json")
        result["worker_wall_seconds"] = time.perf_counter() - started
        write_json(run / "result.json", result)
        # Receipt detects accidental later edits to the report and raw evidence.
        write_json(run / "receipt.json", {"result_hash": digest(result)})
        return run, result


def promote(run):
    run = Path(run).resolve()
    result = read_json(run / "result.json")
    task = Path(result["task_root"]).resolve()
    state = task / ".kernel_envolve"
    if not run.is_relative_to(state / "runs"):
        raise ValueError("Run does not belong to the recorded task")
    with lock(state / "promote.lock"):
        if read_json(run / "receipt.json")["result_hash"] != digest(result):
            raise ValueError("Result changed after evaluation")
        verify_source(run, result["identity"])
        if result["identity"]["evaluator_hash"] != evaluator_hash():
            raise ValueError("Evaluator changed; evaluate again")
        cfg = config(run / "source")
        if contract(task, config(task)) != result["identity"]["contract"]:
            raise ValueError("Live task contract changed; evaluate again")
        decision = score(result, cfg)
        if not decision["eligible_for_best"]:
            raise ValueError("Not eligible: " + "; ".join(decision["reasons"]))
        best = read_best(task)
        current_hash = digest(best[0]) if best else None
        if current_hash != result["initial_best_hash"]:
            raise ValueError("Best changed after evaluation; compare against the new best")
        if best and result["comparator"]["kind"] != "best":
            raise ValueError("Baseline-only comparison cannot replace an existing best")
        pointer = {
            "run": str(run.relative_to(state)),
            "result_hash": digest(result),
            "previous": best[0]["run"] if best else None,
        }
        write_json(state / "best.json", pointer)
        return pointer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible kernel iteration evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="Check and measure a frozen task snapshot")
    ev.add_argument("--task", type=Path, required=True)
    ev.add_argument("--mode", choices=("quick", "full"), default="quick")
    ev.add_argument("--signal-only", action="store_true")
    ev.add_argument(
        "--case", action="append", dest="cases", help="Diagnostic case id; repeatable, quick only"
    )
    ev.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Diagnostic seed; repeatable, quick only",
    )
    ev.add_argument(
        "--check-only",
        action="store_true",
        help="Skip timing for correctness debugging; quick only",
    )
    ev.add_argument(
        "--baseline", action="store_true", help="Compare fixed baseline, even if best exists"
    )
    ev.add_argument("--timeout", type=float, default=300)
    pr = sub.add_parser(
        "promote", help="Recheck evidence and atomically point best at its snapshot"
    )
    pr.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "promote":
            print(json.dumps(promote(args.run), indent=2))
            return 0
        run, result = evaluate(
            args.task,
            args.mode,
            args.signal_only,
            args.timeout,
            args.baseline,
            args.cases,
            args.seeds,
            args.check_only,
        )
        print(
            json.dumps(
                {
                    "run": str(run),
                    "execution": result["execution"],
                    "correctness": result["correctness"],
                    **result["decision"],
                },
                indent=2,
            )
        )
        return 2 if result["execution"] == "error" else 1 if result["correctness"] != "pass" else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"kernel-envolve: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
