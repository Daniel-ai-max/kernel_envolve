"""Small in-process research helpers; results are never promotion evidence."""

import json
import math
import statistics
import time


def paired_probe(baseline, candidate, *, measure, check, trials=3, warmup=1):
    """Reuse two explicit callables, measuring ABBA then BAAB without subprocesses.

    measure(fn) runs fn and returns a positive duration in caller-chosen units.
    It owns device synchronization, state reset and timing scope. check() must
    independently validate outputs/gradients/state and raise on failure (or
    return False); success returns None or True. Called before warmup and after
    timing; never inside timing.
    No source isolation, automatic threshold or statistical acceptance is implied.
    """
    if type(trials) is not int or trials < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("trials must be positive and warmup nonnegative integers")
    functions = {"baseline": baseline, "candidate": candidate}
    if not all(callable(fn) for fn in (*functions.values(), measure, check)):
        raise TypeError("Explicit baseline, candidate, measure and check callables are required")

    def validate():
        verdict = check()
        if verdict is False:
            raise AssertionError("Research correctness check failed")
        if verdict is not None and verdict is not True:
            raise TypeError("check must raise on failure or return None/True/False")

    def sample(role):
        value = measure(functions[role])
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Research timings must be finite and positive")
        return value

    started = time.perf_counter()
    validate()
    for _ in range(warmup):
        for role in functions:
            sample(role)
    preparation_seconds = time.perf_counter() - started
    samples = []
    for trial in range(trials):
        front = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
        for role in front + front[::-1]:
            samples.append({"trial": trial, "role": role, "duration": sample(role)})
    validate()
    medians = {
        role: statistics.median(row["duration"] for row in samples if row["role"] == role)
        for role in functions
    }
    ratio = medians["candidate"] / medians["baseline"]
    if not math.isfinite(ratio):
        raise ValueError("Derived research timing ratio must be finite")
    return {
        "research_only": True,
        "eligible_for_best": False,
        "samples": samples,
        "medians": medians,
        "candidate_over_baseline": ratio,
        "preparation_seconds": preparation_seconds,
        "seconds": time.perf_counter() - started,
    }


def _emit(record):
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def run_stages(stages, *, emit=_emit):
    """Run a small research batch until reject/inconclusive or an exception.

    stages is a sequence of (name, callable). Each callable returns a JSON dict
    with an explicit decision: continue, reject or inconclusive, and a reason.
    Raw evidence/paths may be included. Only continue admits the next stage.
    Exceptions are reported and re-raised; no retry, scheduling or promotion.
    emit may persist records to a caller-owned JSONL file (flush each record).
    GPU callers still wrap the entire batch in one bounded manage gpu-run.
    """
    stages = list(stages)
    names = [name for name, _ in stages]
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or any(not callable(operation) for _, operation in stages)
    ):
        raise ValueError("Use distinct nonempty stage names and callable operations")
    records = []
    for name, operation in stages:
        started = time.perf_counter()
        emit({"stage": name, "status": "started"})
        try:
            outcome = operation()
            if (
                not isinstance(outcome, dict)
                or outcome.get("decision") not in ("continue", "reject", "inconclusive")
                or not isinstance(outcome.get("reason"), str)
                or not outcome["reason"].strip()
            ):
                raise ValueError("Stage must return an explicit decision and reason")
            # Do not emit a success record that cannot actually be persisted.
            json.dumps(outcome, allow_nan=False)
        except BaseException as exc:
            emit(
                {
                    "stage": name,
                    "status": "error",
                    "seconds": time.perf_counter() - started,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            raise
        record = {
            "stage": name,
            "status": "completed",
            "seconds": time.perf_counter() - started,
            "outcome": outcome,
        }
        emit(record)
        records.append(record)
        if outcome["decision"] != "continue":
            break
    return {"research_only": True, "eligible_for_best": False, "stages": records}
