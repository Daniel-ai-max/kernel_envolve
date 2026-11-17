import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "examples/apodex_dsa_research/query_probe.py"
spec = importlib.util.spec_from_file_location("dsa_research_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_wrong_output_rejects_one_candidate_without_cancelling_control():
    calls, events = [], []
    aa = {
        "samples": [
            {"trial": 0, "role": role, "duration": 1.0} for role in ("baseline", "candidate")
        ]
    }

    def stages(candidate):
        def check(a, b):
            calls.append(candidate)
            if candidate == "wrong":
                raise probe.QueryMismatch("wrong dQ")

        return [
            (
                "guard",
                lambda: probe.compare_candidate(
                    lambda: None,
                    lambda: None,
                    aa=aa,
                    check=check,
                    measure=lambda fn: 1.0,
                    trials=2,
                    guard=True,
                    margin=0.05,
                ),
            )
        ]

    results = probe.screen([("bad", "wrong"), ("good", "control")], stages, events.append)
    assert calls == ["wrong", "control", "control"]
    rejected = results["bad"]["stages"][0]["outcome"]
    assert rejected == {"decision": "reject", "reason": "wrong dQ", "correctness": "fail"}
    assert results["good"]["stages"][0]["outcome"]["decision"] == "continue"
    assert all(event["status"] != "error" for event in events)


@pytest.mark.parametrize(
    "failure", [RuntimeError("CUDA failure"), AssertionError("input mutation")]
)
def test_candidate_comparison_never_reclassifies_device_or_unknown_errors(failure):
    def check(a, b):
        raise failure

    with pytest.raises(type(failure), match=str(failure)):
        probe.compare_candidate(
            lambda: None,
            lambda: None,
            aa={},
            check=check,
            measure=lambda fn: 1.0,
            trials=2,
            guard=True,
            margin=0.05,
        )


@pytest.mark.parametrize("mutated,device_error", [(False, False), (True, False), (False, True)])
def test_output_mismatch_cannot_hide_unsafe_shared_state(mutated, device_error):
    synchronized = []

    def synchronize():
        synchronized.append(True)
        if device_error:
            raise RuntimeError("CUDA failure")

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize), equal=np.array_equal)
    values, originals = [np.array([2 if mutated else 1])], [np.array([1])]
    # Missing outputs is intentionally simultaneous with mutation/device failure.
    expected_error = (
        RuntimeError if device_error else AssertionError if mutated else probe.QueryMismatch
    )
    with pytest.raises(expected_error) as error:
        probe.check_outputs([], [], values, originals, torch)
    assert synchronized == [True]
    assert isinstance(error.value, probe.QueryMismatch) == (not mutated and not device_error)


def test_rejected_candidate_does_not_cancel_independent_candidate():
    events, calls = [], []

    def stages(candidate):
        def guard():
            calls.append((candidate, "guard"))
            return {"decision": "reject" if candidate == "slow" else "continue", "reason": "screen"}

        def benefit():
            calls.append((candidate, "benefit"))
            return {"decision": "inconclusive", "reason": "same implementation control"}

        return [("guard", guard), ("benefit", benefit)]

    result = probe.screen([("first", "slow"), ("second", "control")], stages, events.append)
    assert calls == [("slow", "guard"), ("control", "guard"), ("control", "benefit")]
    assert len(result["first"]["stages"]) == 1
    assert len(result["second"]["stages"]) == 2
    assert all(not value["eligible_for_best"] for value in result.values())
    assert {event["candidate"] for event in events} == {"first", "second"}


def test_cuda_error_aborts_instead_of_reusing_a_failed_process():
    events = []

    def stages(candidate):
        assert candidate == "broken"

        def fail():
            raise RuntimeError("CUDA illegal memory access")

        return [("guard", fail)]

    with pytest.raises(RuntimeError, match="illegal memory access"):
        probe.screen([("first", "broken"), ("second", "never")], stages, events.append)
    assert events[-1]["status"] == "error"


@pytest.mark.parametrize(
    "aa,ab,guard,decision",
    [
        ([1.0, 1.0], [1.2, 1.3], True, "reject"),
        ([1.0, 1.0], [1.01, 0.99], True, "continue"),
        ([1.0, 1.0], [1.01, 0.99], False, "inconclusive"),
        ([1.0, 1.0], [0.8, 0.9], False, "continue"),
        ([1.2, 1.0], [0.8, 0.9], False, "inconclusive"),
        ([1.0, 1.0], [0.8, 1.2], False, "inconclusive"),
    ],
)
def test_guard_preserves_neutral_path_but_benefit_requires_consistent_signal(
    aa, ab, guard, decision
):
    assert probe.decide(aa, ab, guard=guard, margin=0.05)[0] == decision
