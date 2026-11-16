"""An installed verifier publishes once; reports and HTTP clients cannot assert gains."""

import json
import time

import pytest

from kernel_envolve.manager.assessment import refresh_sources, watch
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import detail, task_view

pytest_plugins = ["test_manager"]


def setup_source(store, tmp_path):
    receipt = tmp_path / "receipt.json"
    verifier = tmp_path / "verifier.py"
    counter = tmp_path / "calls.txt"
    # Synthetic trusted verifier: real adapters must check their native artifacts.
    verifier.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "context=json.load(sys.stdin)\n"
        f"with Path({str(counter)!r}).open('a') as f:f.write('verify\\n')\n"
        "value=json.loads(Path(context['receipt']).read_text())\n"
        "assert value['fixture_passed']\n"
        "print(json.dumps(value['assessment']))\n"
    )
    value = dict(
        schema="scoped-assessment-v1",
        status="verified",
        promotion=False,
        task="toy",
        baseline_revision=store.task("toy")["spec"]["baseline"]["revision"],
        source_sha256="a" * 64,
        evidence_sha256="b" * 64,
        recorded_at=time.time() - 10,
        metrics=[dict(label="算子", baseline_speedup=1.2, cases=2)],
    )
    receipt.write_text(json.dumps(dict(fixture_passed=True, assessment=value)))
    watch(store, "toy", receipt, verifier)
    return receipt, verifier, counter, value


def test_assessment_survives_restart_and_failed_replacement_is_not_promoted(managed, tmp_path):
    store, _, _ = managed
    receipt, verifier, counter, value = setup_source(store, tmp_path)
    refresh_sources(store)
    refresh_sources(store)
    assert counter.read_text().splitlines() == ["verify"]
    reopened = Store(store.root)
    try:
        view = task_view(reopened, "toy")
        assert view["verified_assessment"] == value
        assert view["best"] is None and not detail(reopened, "toy")["runs"]
        receipt.write_text(json.dumps(dict(fixture_passed=False, assessment=value)))
        refresh_sources(reopened)
        refresh_sources(reopened)
        view = task_view(reopened, "toy")
        assert view["verified_assessment"] == value  # Existing verified history survives.
        assert "verifier failed" in view["assessment_sync"]["error"]
        assert len(counter.read_text().splitlines()) == 2
        assert len(detail(reopened, "toy")["assessment_history"]) == 1
        assert not reopened.rows("SELECT * FROM promotions")
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "different"),
        ("baseline_revision", "different"),
        ("promotion", True),
        ("status", "failed"),
        ("recorded_at", float("nan")),
    ],
)
def test_invalid_identity_never_creates_verified_assessment(managed, tmp_path, field, value):
    store, _, _ = managed
    receipt, _, _, assessment = setup_source(store, tmp_path)
    assessment[field] = value
    receipt.write_text(json.dumps(dict(fixture_passed=True, assessment=assessment)))
    refresh_sources(store)
    view = task_view(store, "toy")
    assert view["verified_assessment"] is None
    assert view["assessment_sync"]["error"]


def test_changed_verifier_is_rejected_and_no_candidate_verifier_is_allowed(managed, tmp_path):
    store, _, work = managed
    receipt, verifier, counter, _ = setup_source(store, tmp_path)
    verifier.write_text(verifier.read_text() + "\n# changed\n")
    refresh_sources(store)
    assert not counter.exists()
    assert "Pinned" in task_view(store, "toy")["assessment_sync"]["error"]
    unsafe = work / "verify.py"
    unsafe.write_text("pass")
    with pytest.raises(ValueError, match="outside"):
        watch(store, "toy", receipt, unsafe)


def test_ended_task_accepts_late_assessment_without_resuming(managed, tmp_path):
    store, _, _ = managed
    setup_source(store, tmp_path)
    store.control("toy", "pause")
    refresh_sources(store)
    assert task_view(store, "toy")["verified_assessment"]
    assert store.task("toy")["desired"] == "paused"
    assert not store.schedule()


def test_verified_receipt_and_summary_remain_in_portable_archive(managed, tmp_path):
    from kernel_envolve.manager.archive import Archive, export_archive, hash_file

    store, _, _ = managed
    receipt, _, _, value = setup_source(store, tmp_path)
    refresh_sources(store)
    original_hash = hash_file(receipt)
    root = tmp_path / "archive"
    result = export_archive(store, root)
    receipt.unlink()
    view = Archive(root).read()
    assert view["details"]["toy"]["verified_assessment"] == value
    saved_input = list((root / result["generation"] / "assessments").glob("*/input.json"))
    assert len(saved_input) == 1 and hash_file(saved_input[0]) == original_hash


@pytest.mark.parametrize("stream", [1, 2])
def test_verifier_output_is_limited_during_execution(managed, tmp_path, stream):
    from kernel_envolve.manager.archive import hash_file
    from kernel_envolve.manager.assessment import verify

    store, _, _ = managed
    receipt, verifier, _, value = setup_source(store, tmp_path)
    verifier.write_text(
        "import os\n"
        f"for _ in range(12): os.write({stream}, b'x' * (256 * 1024))\n"
        f"print({json.dumps(value)!r})\n"
    )
    source = dict(receipt=str(receipt), verifier=str(verifier), verifier_sha256=hash_file(verifier))
    with pytest.raises(ValueError, match="verifier failed"):
        verify(source, store.task("toy"))
