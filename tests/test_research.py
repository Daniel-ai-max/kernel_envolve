import math

import pytest

from kernel_envolve.research import paired_probe, run_stages


def test_probe_reuses_callables_and_preserves_paired_order():
    calls = []

    def baseline():
        calls.append("baseline")

    def candidate():
        calls.append("candidate")

    def measure(fn):
        fn()
        return 10.0 if fn is baseline else 8.0

    result = paired_probe(
        baseline, candidate, measure=measure, check=lambda: calls.append("check"), trials=2
    )
    assert calls == [
        "check",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
        "check",
    ]
    assert result["candidate_over_baseline"] == 0.8
    assert len(result["samples"]) == 8
    assert not result["eligible_for_best"] and result["research_only"]


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan, True])
def test_probe_rejects_invalid_timing(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        paired_probe(lambda: None, lambda: None, measure=lambda _: bad, check=lambda: None)


@pytest.mark.parametrize("after_timing", [False, True])
def test_probe_cannot_hide_correctness_failure(after_timing):
    checks, measured = [], []

    def check():
        checks.append(1)
        if not after_timing or len(checks) == 2:
            raise AssertionError("wrong gradient")

    def measure(_):
        measured.append(1)
        return 1

    with pytest.raises(AssertionError, match="wrong gradient"):
        paired_probe(lambda: None, lambda: None, measure=measure, check=check)
    assert bool(measured) == after_timing


@pytest.mark.parametrize("decision", ["reject", "inconclusive"])
def test_batch_stops_before_expensive_work_and_retains_evidence(decision):
    records = []

    def boundary():
        return {"decision": decision, "reason": "full support regresses", "ratio": 1.33}

    def expensive():
        pytest.fail("A rejected or inconclusive screen must not run full evaluation")

    result = run_stages([("boundary", boundary), ("full", expensive)], emit=records.append)
    assert [r["status"] for r in records] == ["started", "completed"]
    assert result["stages"][0]["outcome"]["ratio"] == 1.33
    assert not result["eligible_for_best"]


@pytest.mark.parametrize(
    "outcome",
    [
        None,
        {},
        {"decision": "continue"},
        {"decision": "continue", "reason": "", "ratio": 1},
        {"decision": "continue", "reason": "ready", "ratio": math.nan},
    ],
)
def test_batch_fails_closed_on_missing_or_invalid_decision(outcome):
    records = []
    with pytest.raises(ValueError):
        run_stages(
            [("screen", lambda: outcome), ("full", lambda: pytest.fail())], emit=records.append
        )
    assert records[-1]["status"] == "error"
    assert all(r["stage"] == "screen" for r in records)


def test_batch_propagates_subprocess_failure_without_running_next_stage():
    import subprocess
    import sys

    records = []

    def command():
        subprocess.run([sys.executable, "-c", "raise SystemExit(3)"], check=True)

    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_stages([("command", command), ("full", lambda: pytest.fail())], emit=records.append)
    assert failure.value.returncode == 3
    assert records[-1]["error"]["type"] == "CalledProcessError"


def test_batch_explicit_continue_is_required_and_emits_before_next_stage():
    records = []

    def next_stage():
        assert records[-2]["outcome"]["decision"] == "continue"
        assert records[-1] == {"stage": "module", "status": "started"}
        return {"decision": "inconclusive", "reason": "below noise"}

    result = run_stages(
        [
            ("screen", lambda: {"decision": "continue", "reason": "boundary checks passed"}),
            ("module", next_stage),
        ],
        emit=records.append,
    )
    assert len(result["stages"]) == 2


@pytest.mark.parametrize("verdict", [False, 0, math.nan, {"passed": False}])
def test_probe_does_not_silently_accept_false_or_ambiguous_check(verdict):
    def measure(_):
        pytest.fail("A failed or ambiguous check must not start timing")

    with pytest.raises((AssertionError, TypeError)):
        paired_probe(lambda: None, lambda: None, measure=measure, check=lambda: verdict)
