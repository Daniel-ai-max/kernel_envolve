"""Synthetic external GPU ledgers exercise import and archive without a GPU."""

import copy
import hashlib
import json
import shutil
import time

import pytest

from kernel_envolve.manager.archive import Archive, export_archive
from kernel_envolve.manager.assessment import refresh_sources
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.validation_import import verify, verify_saved, watch_external
from kernel_envolve.manager.views import detail, task_view
from kernel_envolve.storage import digest

pytest_plugins = ["test_manager"]


@pytest.fixture
def external_manager(managed, tmp_path):
    target, deployment, _ = managed
    copied = copy.deepcopy(deployment)
    copied["tasks"][0]["id"] = "external"
    source = Store(tmp_path / "external-state")
    source.register(copied)
    try:
        yield target, source
    finally:
        source.close()


def terminal(source, *, at=None, status="no_benefit", target="external", terminal=True):
    baseline = source.task("external")["spec"]["baseline"]["revision"]
    endpoint = dict(
        schema="research-source-v1",
        complete=True,
        baseline_revision=baseline,
        tracked_patch=None,
        untracked_scripts={},
    )
    endpoint["fingerprint"] = digest(
        {k: endpoint[k] for k in ("baseline_revision", "tracked_patch", "untracked_scripts")}
    )
    report = dict(
        schema="candidate-validation-v1",
        target=target,
        baseline_revision=baseline,
        source_sha256=endpoint["fingerprint"],
        scope="public F+B",
        mode="eager",
        status=status,
        summary=status,
        limitations=["Scoped result only"],
        checks=[dict(name="performance", status=status)],
        policy=dict(cases=["one"], min_gain=0.01, max_regression=0),
        performance=dict(
            scope="public F+B",
            mode="eager",
            baseline_revision=baseline,
            source_sha256=endpoint["fingerprint"],
            cases=[dict(name="one", correctness=True, baseline=[1, 1, 1], candidate=[2, 2, 2])],
        ),
    )
    ended = time.time() - 10 if at is None else at
    command = ["python", "host-installed-validation.py"]
    job = source.enqueue(source.task("external"), "gpu", command)
    job_status = "succeeded" if status in ("passed", "no_benefit", "inconclusive") else "failed"
    code = 0 if job_status == "succeeded" else 124
    source.db.execute(
        "UPDATE jobs SET status=?,started=?,ended=?,exit_code=? WHERE id=?",
        (job_status if terminal else "running", ended - 1, ended if terminal else None, code, job),
    )
    folder = source.root / "jobs" / job
    folder.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(report).encode()
    (folder / "validation.json").write_bytes(raw)
    log = b"KE_VALIDATION_RECEIPT_SHA256=" + hashlib.sha256(raw).hexdigest().encode() + b"\n"
    (folder / "stdout.log").write_bytes(log)
    (folder / "research-source-before.json").write_text(json.dumps(endpoint))
    bound = dict(
        schema="research-job-source-v1",
        task="external",
        job=job,
        command=command,
        status=job_status if terminal else "running",
        exit_code=code,
        endpoints_complete=True,
        source_changed=False,
        before=endpoint,
        after=endpoint,
        logs={
            "stdout.log": dict(
                complete=True,
                bytes=len(log),
                hashed_bytes=len(log),
                sha256=hashlib.sha256(log).hexdigest(),
            )
        },
    )
    bound_raw = json.dumps(bound).encode()
    (folder / "research-source.json").write_bytes(bound_raw)
    source.db.execute(
        "INSERT INTO metadata VALUES(?,?)",
        ("research_source:" + job, json.dumps(dict(sha256=hashlib.sha256(bound_raw).hexdigest()))),
    )
    return job, folder, report


def test_feed_imports_multiple_jobs_once_and_late_results_do_not_resume(external_manager):
    target, source = external_manager
    watch_external(target, "toy", source.root, "external")
    first, _, _ = terminal(source)
    refresh_sources(target)
    latest = task_view(target, "toy")["latest_validation"]
    assert latest["external"]["job"] == first
    target.control("toy", "pause")
    second, _, _ = terminal(source, at=latest["recorded_at"] + 1, status="interrupted")
    refresh_sources(target)
    refresh_sources(target, force=True)
    assert len(detail(target, "toy")["validation_history"]) == 2
    assert len(target.rows("SELECT * FROM events WHERE kind='validation_indexed'")) == 2
    assert task_view(target, "toy")["latest_validation"]["external"]["job"] == second
    terminal(source, at=latest["recorded_at"] - 1, status="failed")
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"]["external"]["job"] == second
    assert target.task("toy")["desired"] == "paused"
    assert not target.schedule()
    assert not target.rows("SELECT * FROM promotions")


@pytest.mark.parametrize("mutation", ["receipt", "bound", "log", "target", "source", "running"])
def test_wrong_external_proofs_never_replace_an_imported_result(external_manager, mutation):
    target, source = external_manager
    watch_external(target, "toy", source.root, "external")
    terminal(source)
    refresh_sources(target)
    old = task_view(target, "toy")["latest_validation"]
    job, folder, report = terminal(source, at=old["recorded_at"] + 1)
    if mutation in ("receipt", "bound", "log"):
        name = {"receipt": "validation.json", "bound": "research-source.json", "log": "stdout.log"}[
            mutation
        ]
        with (folder / name).open("ab") as stream:
            stream.write(b" ")
    elif mutation in ("target", "source"):
        report["target" if mutation == "target" else "source_sha256"] = (
            "wrong" if mutation == "target" else "f" * 64
        )
        raw = json.dumps(report).encode()
        (folder / "validation.json").write_bytes(raw)
        log = b"KE_VALIDATION_RECEIPT_SHA256=" + hashlib.sha256(raw).hexdigest().encode() + b"\n"
        (folder / "stdout.log").write_bytes(log)
        bound = json.loads((folder / "research-source.json").read_text())
        bound["logs"]["stdout.log"].update(
            bytes=len(log), hashed_bytes=len(log), sha256=hashlib.sha256(log).hexdigest()
        )
        raw = json.dumps(bound).encode()
        (folder / "research-source.json").write_bytes(raw)
        source.db.execute(
            "UPDATE metadata SET data=? WHERE key=?",
            (json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest())), "research_source:" + job),
        )
    else:
        source.db.execute("UPDATE jobs SET status='running' WHERE id=?", (job,))
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"] == old
    assert task_view(target, "toy")["validation_sync"]["error"]
    assert len(detail(target, "toy")["validation_history"]) == 1


def test_running_job_is_not_discovered_until_terminal(external_manager):
    target, source = external_manager
    job, _, _ = terminal(source, terminal=False)
    watch_external(target, "toy", source.root, "external")
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"] is None
    with pytest.raises(ValueError, match="terminal"):
        verify(dict(state=str(source.root), external_task="external", job=job), target.task("toy"))


def test_external_archive_import_and_target_archive_survive_source_removal(
    external_manager, tmp_path
):
    target, source = external_manager
    job, _, _ = terminal(source)
    path = tmp_path / "external-archive"
    exported = export_archive(source, path, include_logs=True)
    assert (path / exported["generation"] / "research-jobs" / job / "validation.json").is_file()
    watch_external(target, "toy", path, "external")
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"]["external"]["job"] == job
    final = tmp_path / "target-archive"
    export_archive(target, final)
    shutil.rmtree(path)
    value = Archive(final).read()["details"]["toy"]
    current = final / json.loads((final / "CURRENT.json").read_text())["generation"]
    folders = list((current / "validations").iterdir())
    assert len(folders) == 1
    assert verify_saved(folders[0], target.task("toy")) == value["latest_validation"]
    log = folders[0] / "proof" / "stdout.log"
    log.write_bytes(log.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="proof changed"):
        verify_saved(folders[0], target.task("toy"))
    assert value["latest_validation"]["external"]["job"] == job
    assert value["best"] is None and value["verified_assessment"] is None


def test_feed_recovers_without_new_job_or_force(external_manager):
    target, source = external_manager
    terminal(source)
    watch_external(target, "toy", source.root, "external")
    refresh_sources(target)
    moved = source.root.with_name("temporarily-offline")
    source.root.rename(moved)
    try:
        refresh_sources(target)
        assert task_view(target, "toy")["validation_sync"]["error"]
    finally:
        moved.rename(source.root)
    refresh_sources(target)
    assert task_view(target, "toy")["validation_sync"]["error"] is None
    assert len(detail(target, "toy")["validation_history"]) == 1


@pytest.mark.parametrize("content", [b"", b"{", b'{"unbound":"receipt"}'])
def test_partial_validation_receipt_cannot_block_whole_archive(external_manager, tmp_path, content):
    target, source = external_manager
    job, folder, _ = terminal(source, status="interrupted")
    (folder / "validation.json").write_bytes(content)
    path = tmp_path / "partial-archive"
    saved = export_archive(source, path, include_logs=True)
    root = path / saved["generation"]
    manifest = json.loads((root / "manifest.json").read_text())
    problem = manifest["unverified_validation_receipts"][0]
    assert problem["job"] == job and problem["reason"]
    assert (root / problem["artifact"]).read_bytes() == content
    assert not (root / "research-jobs" / job / "validation.json").exists()
    assert Archive(path).read()["details"]["external"]["best"] is None
    watch_external(target, "toy", path, "external")
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"] is None


def test_symlink_validation_receipt_is_neither_imported_nor_archived(external_manager, tmp_path):
    target, source = external_manager
    job, folder, _ = terminal(source)
    original = folder / "validation.json"
    other = tmp_path / "not-a-receipt"
    other.write_text("not an allowed artifact")
    original.unlink()
    original.symlink_to(other)
    path = tmp_path / "symlink-archive"
    saved = export_archive(source, path, include_logs=True)
    root = path / saved["generation"]
    problem = json.loads((root / "manifest.json").read_text())["unverified_validation_receipts"][0]
    assert problem["job"] == job and "artifact_error" in problem
    assert not (root / "unverified-validations" / job / "validation.json").exists()
    watch_external(target, "toy", source.root, "external")
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"] is None


def test_success_report_requires_successful_terminal_status(external_manager):
    target, source = external_manager
    job, folder, _ = terminal(source)
    source.db.execute("UPDATE jobs SET status='cancelled',exit_code=0 WHERE id=?", (job,))
    bound = json.loads((folder / "research-source.json").read_text())
    bound["status"] = "cancelled"
    raw = json.dumps(bound).encode()
    (folder / "research-source.json").write_bytes(raw)
    source.db.execute(
        "UPDATE metadata SET data=? WHERE key=?",
        (json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest())), "research_source:" + job),
    )
    watch_external(target, "toy", source.root, "external")
    refresh_sources(target)
    assert task_view(target, "toy")["latest_validation"] is None
    assert task_view(target, "toy")["validation_sync"]["error"]
