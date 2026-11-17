"""Candidate-level validation: cheap rejection before expensive native checks.

These receipts explain a candidate's outcome. They never promote a best or
replace the native evaluator's full/holdout qualification.
"""

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

from kernel_envolve._io import durable_json

MARKER = "KE_VALIDATION_RECEIPT_SHA256="


def performance_decision(
    report,
    *,
    cases,
    scope,
    mode,
    baseline_revision,
    source_sha256,
    min_gain=0.01,
    max_regression=0.0,
):
    """Conservative, prespecified screen of paired complete-call timings.

    Each case supplies matching baseline/candidate duration arrays, at least
    three pairs, and correctness=True. All cases must fit the regression limit;
    at least one must exceed min_gain in >=2/3 pairs and in the paired median.
    This is a screening policy, not a significance test or delivery approval.
    """
    if (
        not cases
        or len(set(cases)) != len(cases)
        or not all(isinstance(c, str) and c.strip() for c in cases)
        or any(
            type(v) not in (float, int) or not math.isfinite(v) or v < 0
            for v in (min_gain, max_regression)
        )
    ):
        raise ValueError("Invalid required cases or performance policy")
    if not isinstance(report, dict) or any(
        report.get(k) != v
        for k, v in {
            "scope": scope,
            "mode": mode,
            "baseline_revision": baseline_revision,
            "source_sha256": source_sha256,
        }.items()
    ):
        raise ValueError("Timing scope, mode or source identity differs")
    rows = report.get("cases")
    if (
        not isinstance(rows, list)
        or len(rows) != len(cases)
        or any(not isinstance(r, dict) for r in rows)
        or {r.get("name") for r in rows} != set(cases)
    ):
        raise ValueError("Required timing cases are incomplete or duplicated")
    medians, promising = [], False
    for row in rows:
        a, b = row.get("baseline"), row.get("candidate")
        if (
            row.get("correctness") is not True
            or not isinstance(a, list)
            or not isinstance(b, list)
            or len(a) != len(b)
            or len(a) < 3
            or len(a) > 10000
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in a + b)
        ):
            raise ValueError("Need correct outputs and finite positive paired timings")
        ratios = [x / y for x, y in zip(a, b)]
        if not all(math.isfinite(v) and v > 0 for v in ratios):
            raise ValueError("Derived paired speedups are invalid")
        median = statistics.median(ratios)
        medians.append(median)
        promising |= median >= 1 + min_gain and sum(
            v >= 1 + min_gain for v in ratios
        ) * 3 >= 2 * len(ratios)
    if all(v <= 1 for v in medians):
        return "no_benefit", "完整调用场景未显示收益，跳过后续昂贵检查。"
    if promising and all(v >= 1 / (1 + max_regression) for v in medians):
        return "continue", "完整调用筛选有收益，继续原生检查。"
    return "inconclusive", "收益不稳定或存在退化，保留结果供后续研究。"


def validate_candidate(
    probe, checks, *, cases, scope, mode, target=None, min_gain=0.01, max_regression=0.0
):
    """Run once inside a research manage gpu-run, persist each stage automatically.

    probe() returns performance_decision's report. checks is (name, callable)
    pairs; each callable must return True after its native checks pass, or raise.
    Results stay in the job directory, bound later by the supervisor. Interrupted
    attempts are retained; a repeated call in the same job is rejected, not replayed.
    """
    state, job = os.environ.get("KERNEL_ENVOLVE_STATE"), os.environ.get("KERNEL_ENVOLVE_GPU_JOB")
    if not state or not job or Path(job).name != job:
        raise ValueError("Use validate_candidate inside manage gpu-run")
    folder = Path(state) / "jobs" / job
    before = json.loads((folder / "research-source-before.json").read_text())
    if before.get("complete") is not True:
        raise ValueError("Complete research source capture required")
    checks = list(checks)
    names = ["performance", *[name for name, _ in checks]]
    if (
        not checks
        or len(names) != len(set(names))
        or len(names) > 1000
        or any(not isinstance(n, str) or not n.strip() or len(n) > 300 for n in names)
        or not all(callable(fn) for _, fn in checks)
        or not callable(probe)
        or not isinstance(scope, str)
        or not scope.strip()
        or len(scope) > 300
        or not isinstance(mode, str)
        or not mode.strip()
    ):
        raise ValueError("Use named native checks and an explicit timing scope/mode")
    path = folder / "validation.json"
    # Exclusive reservation also rejects concurrent or accidental repeated calls.
    with path.open("x"):
        pass
    record = {
        "schema": "candidate-validation-v1",
        "target": target or os.environ["KERNEL_ENVOLVE_TASK"],
        "baseline_revision": before["baseline_revision"],
        "source_sha256": before["fingerprint"],
        "scope": scope,
        "mode": mode,
        "status": "interrupted",
        "summary": "验收执行中断。",
        "limitations": [],
        "checks": [{"name": n, "status": "skipped"} for n in names],
        "policy": {"cases": list(cases), "min_gain": min_gain, "max_regression": max_regression},
        "performance": None,
        "stage_seconds": {},
    }

    def save():
        if record["status"] in ("passed", "no_benefit", "inconclusive", "failed"):
            print(
                json.dumps(
                    {"validation_status": record["status"], "summary": record["summary"]},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )
        durable_json(path, record)
        print(MARKER + hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)

    current, started = record["checks"][0], time.perf_counter()
    try:
        current["status"] = "interrupted"
        save()
        report = probe()
        # Refuse nonportable evidence before it reaches the persistent receipt.
        raw = json.dumps(report, allow_nan=False)
        if len(raw.encode()) > 2 * 1024 * 1024:
            raise ValueError("Timing report exceeds 2 MiB")
        record["performance"] = report
        status, summary = performance_decision(
            report,
            cases=cases,
            scope=scope,
            mode=mode,
            baseline_revision=before["baseline_revision"],
            source_sha256=before["fingerprint"],
            min_gain=min_gain,
            max_regression=max_regression,
        )
        record["stage_seconds"]["performance"] = time.perf_counter() - started
        current["status"] = "passed" if status == "continue" else status
        record.update(status="interrupted" if status == "continue" else status, summary=summary)
        save()
        if status != "continue":
            return record
        for current, (_, operation) in zip(record["checks"][1:], checks):
            started = time.perf_counter()
            current["status"] = "interrupted"
            record["summary"] = "原生检查尚未完成。"
            save()
            if operation() is not True:
                raise AssertionError("Native check must explicitly return True")
            current["status"] = "passed"
            record["stage_seconds"][current["name"]] = time.perf_counter() - started
            save()
        record.update(status="passed", summary="指定范围的完整调用筛选与原生检查通过。")
        save()
        return record
    except BaseException as exc:
        status = (
            "interrupted"
            if isinstance(
                exc, (KeyboardInterrupt, SystemExit, TimeoutError, subprocess.TimeoutExpired)
            )
            else "inconclusive"
            if current["name"] == "performance"
            else "failed"
        )
        current["status"] = status
        record.update(
            status=status, summary=f"{current['name']}: {type(exc).__name__}: {exc}"[:2000]
        )
        # An invalid/nonfinite probe must not prevent the interruption receipt.
        try:
            json.dumps(record, allow_nan=False)
        except (ValueError, TypeError):
            record["performance"] = None
        record["stage_seconds"][current["name"]] = time.perf_counter() - started
        save()
        raise
