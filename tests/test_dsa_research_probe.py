import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "examples/apodex_dsa_research/query_probe.py"
spec = importlib.util.spec_from_file_location("dsa_research_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


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
