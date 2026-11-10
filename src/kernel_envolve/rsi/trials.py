"""Matched, bounded solver trials with controller-owned full evaluation.

Configured solver commands are trusted adapters. Candidate artifacts contain only advice.
Use an externally sandboxed adapter when the underlying coding agent is untrusted.
"""

import contextlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from ..manager.evidence import inspect_run
from ..manager.process import exclusive, gpu_idle, lock_path
from ..storage import config, digest, evaluator_hash, file_hash, write_json
from .learning import verify_evidence
from .model import context, read
from .process import invoke


def validator_identity():
    return digest(
        {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("trials.py", "model.py", "process.py")
        }
    )


def files(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in {".git", "__pycache__", ".kernel_native", ".kernel_envolve", ".pytest_cache"}
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted in trial sources")
        if path.is_file():
            result[str(relative)] = file_hash(path)
    return result


def git_snapshot(root):
    env = os.environ | {
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        "GIT_AUTHOR_NAME": "RSI fixture",
        "GIT_COMMITTER_NAME": "RSI fixture",
        "GIT_AUTHOR_EMAIL": "rsi@example.invalid",
        "GIT_COMMITTER_EMAIL": "rsi@example.invalid",
    }
    for args in (
        ["init", "-q"],
        ["add", "-f", "."],
        ["commit", "-q", "--allow-empty", "-m", "Frozen validation fixture"],
    ):
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            timeout=30,
        )


def freeze_template(task, target):
    target.mkdir(parents=True)
    original = Path(task["path"])
    if task["kind"] == "generic":
        from ..storage import bundle

        cfg = config(original)
        bundle(original, target / "task", cfg)
        editable = [f"task/{name}" for name in cfg["candidate_files"]]
    else:
        from ..native import snapshot
        from ..native_config import load_config

        cfg = load_config(original)
        (target / "task").mkdir()
        shutil.copy2(original / "adapter.py", target / "task/adapter.py")
        for role, roots in cfg["sources"].items():
            for name, path in roots.items():
                destination = target / role / name
                snapshot(path, destination, cfg["artifacts"].get(name, []))
                git_snapshot(destination)
                roots[name] = f"../{role}/{name}"
        write_json(target / "task/native.json", cfg)
        editable = task["editable"]
    identity = files(target)
    if any(name not in identity for name in editable):
        raise ValueError("Every editable file must exist in the frozen template")
    return {
        "path": str(target),
        "editable": editable,
        "files": identity,
        "template_hash": digest(identity),
    }


def verify_template(task):
    if files(Path(task["path"])) != task["files"] or digest(task["files"]) != task["template_hash"]:
        raise ValueError("Validation fixture changed; use a new RSI configuration")


def trial(directory, task, suite, rules, seed, timeout, device):
    began = time.monotonic()
    verify_template(task)
    workspace = directory / "workspace"
    shutil.copytree(task["path"], workspace)
    protected = {k: v for k, v in task["files"].items() if k not in task["editable"]}
    remaining = timeout - (time.monotonic() - began)
    if remaining <= 0:
        raise TimeoutError("Validation preparation exhausted its deadline")
    packet = {
        "task": str(workspace / "task"),
        "kind": task["kind"],
        "editable_files": [str(workspace / p) for p in task["editable"]],
        "model": suite["model"],
        "seed": seed,
        "seconds": min(remaining, suite["solver_seconds"]),
        "strategy": context(rules),
        "instructions": "Optimize only editable_files. Preserve task, oracle, tests, baseline and environment. Do not promote. The controller runs independent full evaluation after you exit. Return within the supplied time budget; use the configured model and seed where supported.",
    }
    solver_seconds = invoke(suite["command"], directory, packet, packet["seconds"], device=device)
    after = files(workspace)
    if set(after) != set(task["files"]) or any(after.get(k) != v for k, v in protected.items()):
        raise ValueError("Solver changed protected files or added undeclared files")
    task_path = workspace / "task"
    # A separate process keeps CUDA visibility and adapter imports out of the controller.
    remaining = timeout - (time.monotonic() - began)
    if remaining <= 0:
        raise TimeoutError("Validation deadline exhausted")
    evaluation = directory / "evaluation"
    evaluation.mkdir()
    timeout = min(suite["evaluation_seconds"], remaining)
    code = (
        "import json,sys; from pathlib import Path; "
        "from kernel_envolve.cli import evaluate; from kernel_envolve.native import evaluate_native; "
        "task,kind,seconds=sys.argv[1:]; "
        "run,result=(evaluate(task,mode='full',timeout=float(seconds)) if kind=='generic' else evaluate_native(task,mode='full',timeout=float(seconds))); "
        "Path('evaluated.json').write_text(json.dumps({'run':str(run)}))"
    )
    import sys

    invoke(
        [sys.executable, "-c", code, str(task_path), task["kind"], str(max(0.01, timeout - 1))],
        evaluation,
        {},
        timeout,
        device=device,
    )
    run = Path(read(evaluation / "evaluated.json")["run"])
    state_name = ".kernel_envolve" if task["kind"] == "generic" else ".kernel_native"
    if not run.resolve().is_relative_to((task_path / state_name / "runs").resolve()):
        raise ValueError("Evaluation result escaped the trial")
    result = read(run / "result.json", 16 * 1024 * 1024)
    contract = result["identity"]["contract"] if task["kind"] == "generic" else result["contract"]
    spec = {"kind": task["kind"], "task_dir": str(task_path), "baseline": {"contract": contract}}
    verified = inspect_run(run, spec)
    # Correctness failure cannot be compensated by performance or cost.
    passed = verified["execution"] == "completed" and verified["correctness"] == "pass"
    gains = verified["decision"].get("fixed_baseline_gain_by_batch", [])
    quality = (
        max(0.0, statistics.median(gains))
        if passed and gains and verified["decision"]["eligible_for_best"]
        else 0.0
        if passed
        else -1.0
    )
    return {
        "spec": spec,
        "run": str(run),
        "result_hash": verified["result_hash"],
        "passed": passed,
        "quality": quality,
        "solver_seconds": solver_seconds,
        "seconds": time.monotonic() - began,
        "template_hash": task["template_hash"],
        "seed": seed,
    }


def verdict(pairs, suite):
    """Preserve tasks and require a whole-suite benefit within a declared cost ceiling."""
    reasons, repeats = [], []
    result = {
        "accepted": False,
        "reasons": reasons,
        "repeats": repeats,
        "scope": "Equal-weight declared tasks only; wall time is not token cost or a significance test",
    }
    expected = {(r, t) for r in range(suite["repeats"]) for t in range(len(suite["tasks"]))}
    try:
        complete = (
            bool(expected)
            and len(pairs) == len(expected)
            and {(p["repeat"], p["task"]) for p in pairs} == expected
        )
        metrics_valid = all(
            type(p[role]["passed"]) is bool
            and all(
                type(p[role][field]) in (int, float) and math.isfinite(p[role][field])
                for field in ("quality", "solver_seconds", "seconds")
            )
            and p[role]["solver_seconds"] > 0
            and p[role]["seconds"] > 0
            and (p[role]["quality"] >= 0 if p[role]["passed"] else p[role]["quality"] == -1)
            for p in pairs
            for role in ("control", "candidate")
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        complete = metrics_valid = False
    if not complete or not metrics_valid:
        reasons.append("Incomplete matched validation or invalid trial metrics")
        return result

    for pair in pairs:
        old, new = pair["control"], pair["candidate"]
        if not new["passed"]:
            reasons.append("Candidate did not pass full correctness")
        if old["passed"] and new["quality"] < old["quality"] - suite["max_quality_regression"]:
            reasons.append("Per-task quality regression")
    # Failures are a separate correctness outcome, not fictitious negative speedups.
    # Never pool repeats: one lucky run must not cancel another run's regression.
    for repeat in range(suite["repeats"]):
        group = [p for p in pairs if p["repeat"] == repeat]
        totals = {"repeat": repeat}
        for role in ("control", "candidate"):
            totals[f"{role}_quality"] = statistics.mean(max(0.0, p[role]["quality"]) for p in group)
            totals[f"{role}_passed"] = sum(p[role]["passed"] for p in group)
            for field in ("solver_seconds", "seconds"):
                totals[f"{role}_{field}"] = math.fsum(p[role][field] for p in group)
        gain = totals["candidate_quality"] - totals["control_quality"]
        totals["quality_gain"] = gain
        if gain < -1e-12:
            reasons.append("Aggregate quality regression")
        for field in ("solver_seconds", "seconds"):
            if totals[f"candidate_{field}"] > totals[f"control_{field}"] * (
                1 + suite.get("max_cost_regression", 0.05)
            ):
                reasons.append(f"Aggregate {field} regression")
        quality_win = gain > suite["min_quality_gain"]
        coverage_win = totals["candidate_passed"] > totals["control_passed"]
        cost_win = (
            totals["control_quality"] > 0
            and gain >= -1e-12
            and totals["candidate_solver_seconds"]
            < totals["control_solver_seconds"] * (1 - suite["min_cost_gain"])
        )
        if not (quality_win or coverage_win or cost_win):
            reasons.append("No reproducible whole-suite quality, correctness or cost benefit")
        repeats.append(totals)
    result.update(accepted=not reasons, reasons=sorted(set(reasons)))
    return result


def validate(store, identity):
    candidate = store.candidate(identity)
    scope = candidate["scope"]
    if not store.get("enabled", True):
        raise ValueError("RSI is disabled")
    if candidate["parent"] != store.active(scope) or candidate["status"] != "candidate":
        raise ValueError(
            "Candidate is stale or has already been evaluated; repeated confirmation is forbidden"
        )
    if scope not in store.config["suites"]:
        raise ValueError("No operator-configured validation suite")
    suite = store.config["suites"][scope]
    verify_evidence(store, [e for r in candidate["rules"] for e in r["evidence"]], scope)
    device = store.config["validation_device"]
    lock = exclusive(lock_path("gpu", device)) if device != "cpu" else contextlib.nullcontext(True)
    with lock as handle:
        if handle is None:
            return {"status": "resource_busy"}
        if device != "cpu":
            gpu_idle(device)
        key = store.reserve("validation", suite["max_seconds"])
        store.db.execute("UPDATE candidates SET status='validating' WHERE id=?", (identity,))
        directory = store.root / "validation" / key
        directory.mkdir(parents=True)
        began, error, pairs = time.monotonic(), None, []
        try:
            control = store.candidate(candidate["parent"])
            for repeat in range(suite["repeats"]):
                for index, task in enumerate(suite["tasks"]):
                    pair = {"repeat": repeat, "task": index}
                    order = (
                        ("control", "candidate")
                        if (repeat + index) % 2 == 0
                        else ("candidate", "control")
                    )
                    for role in order:
                        remaining = suite["max_seconds"] - (time.monotonic() - began)
                        if remaining <= 1:
                            raise TimeoutError("Validation budget exhausted")
                        path = directory / f"{repeat}-{index}-{role}"
                        path.mkdir()
                        pair[role] = trial(
                            path,
                            task,
                            suite,
                            (control if role == "control" else candidate)["rules"],
                            repeat,
                            remaining,
                            device,
                        )
                    pairs.append(pair)
            report = {
                "candidate": identity,
                "parent": candidate["parent"],
                "scope": scope,
                "suite_hash": digest(suite),
                "evaluator_hash": evaluator_hash(),
                "validator_hash": validator_identity(),
                "pairs": pairs,
                "decision": verdict(pairs, suite),
                "model": suite["model"],
                "at": time.time(),
            }
            receipt = digest(report)
            write_json(directory / "validation.json", report)
            with store.transaction():
                store.db.execute(
                    "INSERT INTO validations VALUES(?,?,?)", (receipt, identity, json.dumps(report))
                )
                store.db.execute(
                    "UPDATE candidates SET status=? WHERE id=?",
                    ("validated" if report["decision"]["accepted"] else "rejected", identity),
                )
            return {"status": "validated", "receipt": receipt, **report["decision"]}
        except BaseException as exc:
            error = str(exc)
            write_json(
                directory / "failure.json", {"candidate": identity, "error": error, "pairs": pairs}
            )
            store.db.execute("UPDATE candidates SET status='rejected' WHERE id=?", (identity,))
            raise
        finally:
            store.finish(key, time.monotonic() - began, error)
            if device != "cpu":
                gpu_idle(device)


def checked_receipt(store, receipt):
    row = store.db.execute("SELECT data FROM validations WHERE id=?", (receipt,)).fetchone()
    if not row:
        raise ValueError("Unknown validation receipt")
    report = json.loads(row[0])
    if digest(report) != receipt:
        raise ValueError("Validation receipt mismatch")
    if (
        report.get("evaluator_hash") != evaluator_hash()
        or report.get("validator_hash") != validator_identity()
    ):
        raise ValueError(
            "Evaluator or strategy validator changed; create a new validation experiment"
        )
    candidate = store.candidate(report["candidate"])
    scope = candidate["scope"]
    suite = store.config["suites"][scope]
    if candidate["status"] != "validated" or digest(suite) != report["suite_hash"]:
        raise ValueError("Candidate is not validated against the current suite")
    expected_pairs = {(r, i) for r in range(suite["repeats"]) for i in range(len(suite["tasks"]))}
    if (
        len(report["pairs"]) != len(expected_pairs)
        or {(p["repeat"], p["task"]) for p in report["pairs"]} != expected_pairs
    ):
        raise ValueError("Incomplete matched validation")
    for task in suite["tasks"]:
        verify_template(task)
    for pair in report["pairs"]:
        for role in ("control", "candidate"):
            result = pair[role]
            verified = inspect_run(Path(result["run"]), result["spec"])
            if verified["result_hash"] != result["result_hash"]:
                raise ValueError("Validation evidence changed")
    if not verdict(report["pairs"], suite)["accepted"]:
        raise ValueError("Validation did not pass")
    verify_evidence(store, [e for r in candidate["rules"] for e in r["evidence"]], scope)
    return candidate


def adopt(store, receipt):
    candidate = checked_receipt(store, receipt)
    scope = candidate["scope"]
    with store.transaction():
        if not store.get("enabled", True):
            raise ValueError("RSI was disabled; no strategy activated")
        if store.active(scope) != candidate["parent"]:
            raise ValueError(
                "Active strategy changed; repeat validation against its current version"
            )
        store.db.execute("INSERT OR REPLACE INTO active VALUES(?,?)", (scope, candidate["id"]))
        store.db.execute(
            "INSERT INTO history(at,scope,previous,current,reason) VALUES(?,?,?,?,?)",
            (time.time(), scope, candidate["parent"], candidate["id"], receipt),
        )
    store.publish()
    return candidate["id"]


def rollback(store, scope, reason):
    text_reason = str(reason).strip()
    if not text_reason or scope not in store.config["scopes"]:
        raise ValueError("Rollback requires a scope and reason")
    with store.transaction():
        current = store.active(scope)
        if current == "base":
            return "base"
        candidate = store.candidate(current)
        target = candidate["parent"]
        while target != "base" and store.candidate(target)["status"] == "revoked":
            target = store.candidate(target)["parent"]
        store.db.execute("UPDATE candidates SET status='revoked' WHERE id=?", (current,))
        store.db.execute("INSERT OR REPLACE INTO active VALUES(?,?)", (scope, target))
        store.db.execute(
            "INSERT INTO history(at,scope,previous,current,reason) VALUES(?,?,?,?,?)",
            (time.time(), scope, current, target, text_reason),
        )
    store.publish()
    return target
