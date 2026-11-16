"""No GPU required: screen decisions gate real callback execution and persist failures."""

import copy
import hashlib
import json

import pytest

from kernel_envolve.storage import digest
from kernel_envolve.validation import MARKER, performance_decision, validate_candidate


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    job = "a" * 32
    folder = tmp_path / "jobs" / job
    folder.mkdir(parents=True)
    before = dict(baseline_revision="baseline", tracked_patch=None, untracked_scripts={})
    before["fingerprint"] = digest(before)
    before["complete"] = True
    (folder / "research-source-before.json").write_text(json.dumps(before))
    monkeypatch.setenv("KERNEL_ENVOLVE_STATE", str(tmp_path))
    monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", job)
    monkeypatch.setenv("KERNEL_ENVOLVE_TASK", "toy")
    report = dict(
        scope="full forward/backward",
        mode="wall",
        baseline_revision="baseline",
        source_sha256=before["fingerprint"],
        cases=[
            dict(name="short", correctness=True, baseline=[100] * 7, candidate=[110] * 7),
            dict(name="long", correctness=True, baseline=[100] * 7, candidate=[105] * 7),
        ],
    )
    return folder, report


def run(report, checks):
    return validate_candidate(
        lambda: report, checks, cases=["short", "long"], scope="full forward/backward", mode="wall"
    )


def test_no_benefit_stops_native_gates_and_persists(attempt, capsys):
    folder, report = attempt
    calls = []
    result = run(report, [("expensive", lambda: calls.append("called"))])
    assert result["status"] == "no_benefit" and not calls
    assert result["checks"] == [
        dict(name="performance", status="no_benefit"),
        dict(name="expensive", status="skipped"),
    ]
    raw = (folder / "validation.json").read_bytes()
    assert json.loads(raw) == result
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == MARKER + hashlib.sha256(raw).hexdigest()
    assert json.loads(lines[-2])["validation_status"] == "no_benefit"
    with pytest.raises(FileExistsError):
        run(report, [("expensive", lambda: True)])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(mode="graph"),
        lambda r: r.update(source_sha256="b" * 64),
        lambda r: r.update(baseline_revision="other"),
        lambda r: r["cases"].pop(),
        lambda r: r["cases"][0]["baseline"].__setitem__(0, float("nan")),
        lambda r: r["cases"][0].update(correctness=False),
        lambda r: r["cases"][0].update(candidate=[100]),
        lambda r: r["cases"][1].update(name="short"),
    ],
)
def test_invalid_data_never_opens_expensive_gates(attempt, mutation):
    folder, report = attempt
    mutation(report)
    calls = []
    with pytest.raises(ValueError):
        run(report, [("expensive", lambda: calls.append(1))])
    assert not calls
    saved = json.loads((folder / "validation.json").read_text())
    assert saved["status"] == "inconclusive"
    assert saved["checks"][1]["status"] == "skipped"


def test_gain_still_requires_all_checks_and_failure_stops_rest(attempt):
    folder, report = attempt
    for row in report["cases"]:
        row["candidate"] = [80] * 7
    calls = []

    def fail():
        calls.append("first")
        raise AssertionError("native mismatch")

    with pytest.raises(AssertionError):
        run(report, [("first", fail), ("holdout", lambda: calls.append("second"))])
    assert calls == ["first"]
    saved = json.loads((folder / "validation.json").read_text())
    assert saved["status"] == "failed" and saved["checks"][2]["status"] == "skipped"


def test_interrupt_retains_completed_stages(attempt):
    folder, report = attempt
    for row in report["cases"]:
        row["candidate"] = [80] * 7

    def stop():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(report, [("first", lambda: True), ("second", stop), ("third", lambda: True)])
    saved = json.loads((folder / "validation.json").read_text())
    assert saved["status"] == "interrupted"
    assert [r["status"] for r in saved["checks"]] == ["passed", "passed", "interrupted", "skipped"]


def test_noise_is_not_pass_or_no_gain(attempt):
    _, report = attempt
    for row in report["cases"]:
        row["candidate"] = [99, 102, 98, 100, 99, 100, 99]
    result = run(report, [("native", lambda: pytest.fail("must not run"))])
    assert result["status"] == "inconclusive"


def test_pass_is_only_scoped_receipt_not_promotion(attempt):
    _, report = attempt
    for row in report["cases"]:
        row["candidate"] = [80] * 7
    result = run(report, [("native", lambda: True), ("holdout", lambda: True)])
    assert result["status"] == "passed"
    assert all(r["status"] == "passed" for r in result["checks"])
    assert "best" not in result and "promotion" not in result


def test_independent_complete_cases_required(attempt):
    _, report = attempt
    duplicate = copy.deepcopy(report)
    duplicate["cases"].append(duplicate["cases"][0])
    with pytest.raises(ValueError):
        performance_decision(
            duplicate,
            cases=["short", "long"],
            scope=report["scope"],
            mode="wall",
            baseline_revision="baseline",
            source_sha256=report["source_sha256"],
        )
