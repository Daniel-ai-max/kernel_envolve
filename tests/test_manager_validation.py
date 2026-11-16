"""External validation facts do not mutate optimization or qualification state."""

import copy
import json
import time

import pytest

from kernel_envolve.manager.archive import Archive, export_archive
from kernel_envolve.manager.assessment import refresh_sources, validate_validation, watch
from kernel_envolve.manager.narrator import collect, validate
from kernel_envolve.manager.views import detail, task_view

pytest_plugins = ["test_manager"]


def setup_validation(store, tmp_path, status="no_benefit"):
    receipt = tmp_path / "validation-input.json"
    verifier = tmp_path / "validation-verifier.py"
    # Test-only pinned verifier; production adapters must inspect external source/job artifacts.
    verifier.write_text(
        "import json,sys,pathlib\nc=json.load(sys.stdin)\n"
        "print(pathlib.Path(c['receipt']).read_text())\n"
    )
    baseline = store.task("toy")["spec"]["baseline"]["revision"]
    value = dict(
        schema="validation-result-v1",
        task="toy",
        baseline_revision=baseline,
        source_sha256="a" * 64,
        evidence_sha256="b" * 64,
        recorded_at=time.time() - 10,
        promotion=False,
        status=status,
        scope="public forward+backward",
        summary="调用链未见收益，未运行昂贵统计门禁",
        limitations=["未完成完整验收"],
        external=dict(
            task="external-task",
            job="external-job",
            receipt_sha256="c" * 64,
            source_sha256="a" * 64,
            evidence_sha256="b" * 64,
            baseline_revision=baseline,
        ),
        checks=[dict(name="public F+B", status=status)],
    )
    receipt.write_text(json.dumps(value))
    watch(store, "toy", receipt, verifier, validation=True)
    return receipt, verifier, value


@pytest.mark.parametrize(
    "status", ["passed", "no_benefit", "failed", "interrupted", "inconclusive"]
)
def test_external_outcomes_are_facts_not_verified_gains(managed, tmp_path, status):
    store, _, _ = managed
    _, _, value = setup_validation(store, tmp_path, status)
    before = store.rows("SELECT * FROM tasks")
    refresh_sources(store)
    view = task_view(store, "toy")
    assert view["latest_validation"] == value
    assert view["verified_assessment"] is None and view["best"] is None
    assert store.rows("SELECT * FROM tasks") == before
    assert not store.rows("SELECT * FROM promotions")


def test_force_replay_is_idempotent_and_old_late_outcome_keeps_latest(managed, tmp_path):
    store, _, _ = managed
    receipt, _, value = setup_validation(store, tmp_path)
    refresh_sources(store)
    refresh_sources(store, force=True)
    assert len(store.rows("SELECT * FROM events WHERE kind='validation_indexed'")) == 1
    assert len(detail(store, "toy")["validation_history"]) == 1
    older = copy.deepcopy(value)
    older.update(recorded_at=value["recorded_at"] - 1, status="failed", summary="earlier failure")
    receipt.write_text(json.dumps(older))
    refresh_sources(store)
    assert task_view(store, "toy")["latest_validation"] == value
    assert len(detail(store, "toy")["validation_history"]) == 2
    receipt.write_text(json.dumps(value | {"task": "other"}))
    refresh_sources(store)
    assert task_view(store, "toy")["latest_validation"] == value
    assert task_view(store, "toy")["validation_sync"]["error"]


def test_late_validation_archives_after_source_removal_and_never_resumes(managed, tmp_path):
    store, _, _ = managed
    receipt, _, value = setup_validation(store, tmp_path)
    store.control("toy", "pause")
    refresh_sources(store)
    assert store.task("toy")["desired"] == "paused"
    assert not store.schedule()
    output = tmp_path / "saved"
    saved = export_archive(store, output)
    receipt.unlink()
    archived = Archive(output).read()
    assert archived["details"]["toy"]["latest_validation"] == value
    assert len(list((output / saved["generation"] / "validations").glob("*/input.json"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(task="other"),
        lambda v: v.update(baseline_revision="other"),
        lambda v: v.update(promotion=True),
        lambda v: v.update(status="verified"),
        lambda v: v.update(recorded_at=float("nan")),
        lambda v: v["external"].update(source_sha256="d" * 64),
        lambda v: v["external"].update(job="../../job"),
        lambda v: v.update(status="passed"),
        lambda v: v.update(checks=[]),
    ],
)
def test_invalid_external_identity_or_fake_pass_is_rejected(managed, tmp_path, change):
    store, _, _ = managed
    _, _, value = setup_validation(store, tmp_path)
    change(value)
    with pytest.raises(ValueError):
        validate_validation(value, store.task("toy"))


def test_validation_coexists_with_research_and_emits_narrator_source(managed, tmp_path):
    store, _, _ = managed
    _, verifier, value = setup_validation(store, tmp_path)
    # Separate sources may use separate formats/verifiers without repinning one another.
    watch(store, "toy", tmp_path / "future-research.json", verifier, research=True)
    refresh_sources(store)
    store.control("toy", "pause")
    batches = collect(store, scan_logs=False)
    batch = next(b for b in batches if b["task"] == "toy")
    ref = next(ref for ref, s in batch["sources"].items() if s["kind"] == "validation")
    assert value["summary"] in batch["sources"][ref]["text"]
    claims = dict(
        direction=None, finding=None, failure=None, next=dict(text="invented plan", sources=[ref])
    )
    assert validate({"toy": claims}, [batch])["toy"]["next"] is None
    assert not batch["can_continue"]
