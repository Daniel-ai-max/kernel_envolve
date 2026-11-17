"""Recovery is an immutable, provenance-bound presentation, never resumed research."""

import hashlib
import json

import pytest

from kernel_envolve.manager.dashboard import narrator
from kernel_envolve.manager.dashboard.row_facts import row_facts
from kernel_envolve.manager.results.archive import Archive
from kernel_envolve.manager.results.recovery import recover_archive
from kernel_envolve.manager.store import Store

pytest_plugins = ["test_manager"]


def test_recovery_preserves_input_and_persists_validated_narration(managed, tmp_path, monkeypatch):
    store, _, _ = managed
    evidence = tmp_path / "platform.json"
    evidence.write_text('{"stopped": true}')
    source = tmp_path / "source"
    source.mkdir()
    import sqlite3

    with sqlite3.connect(source / "manager.sqlite") as copy:
        store.db.backup(copy)
    before = (source / "manager.sqlite").read_bytes()
    original_task = store.task("toy")

    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must never inspect live logs")

    monkeypatch.setattr(narrator, "log_paths", forbidden)

    def respond(command, request, stop):
        batch = request["tasks"][0]
        assert batch["can_continue"] is False
        ref = next(k for k in batch["sources"] if k.startswith("recovery:"))
        claim = {"text": "会话额度中断，保留已有候选，未形成完整验收结论。", "sources": [ref]}
        return {
            "toy": {
                "direction": None,
                "finding": claim,
                "failure": None,
                "next": None,
                "row": {
                    "revision": batch["row_facts"]["revision"],
                    "benefit": "system:benefit",
                    "optimization_pr": "system:optimization_pr",
                    "current_progress": "finding",
                    "latest_result": {"ref": "system:latest_result", **claim},
                },
            }
        }

    result = recover_archive(
        source,
        tmp_path / "archive",
        observed_at=100,
        recorded_at=200,
        coverage="Last consistent snapshot; terminal logs unavailable",
        interruption="Session quota exhausted; allocation stopped",
        attachments=[evidence],
        command=["synthetic"],
        call=respond,
    )
    archive = Archive(tmp_path / "archive")
    view = archive.read()
    task = view["details"]["toy"]
    assert view["state"]["as_of"] == 100
    assert task["execution_state"] == "archived"
    assert task["narration"]["row"]["revision"] == task["row_facts"]["revision"]
    assert task["narration"]["evidence_revision"] == task["row_facts"]["evidence_revision"]
    assert task["archive_recovery"]["sources"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    saved = Store(tmp_path / "archive" / result["generation"], readonly=True)
    try:
        assert saved.task("toy") == original_task
    finally:
        saved.close()
    assert (source / "manager.sqlite").read_bytes() == before
    assert Archive(tmp_path / "archive").read() == view
    changed = dict(task, archive_recovery={**task["archive_recovery"], "interruption": "changed"})
    assert row_facts(changed)["evidence_revision"] != task["row_facts"]["evidence_revision"]


def test_replay_is_readonly_and_does_not_scan_files(managed, monkeypatch):
    store, _, _ = managed
    store.event("toy", "failed", {"error": "fixture failure"})
    before = store.rows("SELECT * FROM metadata")
    monkeypatch.setattr(narrator, "log_paths", lambda *args: pytest.fail("live log scan"))
    readonly = Store(store.root, readonly=True)
    try:
        assert narrator.collect(readonly, replay=True)
    finally:
        readonly.close()
    assert store.rows("SELECT * FROM metadata") == before


def test_recovery_missing_receipts_are_explicit(managed, tmp_path):
    store, _, _ = managed
    source = tmp_path / "source"
    source.mkdir()
    import sqlite3

    with sqlite3.connect(source / "manager.sqlite") as copy:
        store.db.backup(copy)
        copy.execute(
            "INSERT INTO metadata VALUES(?,?)",
            ("assessment_history:" + "a" * 64, json.dumps({"input_sha256": "b" * 64})),
        )
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    result = recover_archive(
        source,
        tmp_path / "archive",
        observed_at=100,
        recorded_at=200,
        coverage="missing terminal archive",
        interruption="operator stopped",
        attachments=[evidence],
    )
    manifest = json.loads(
        (tmp_path / "archive" / result["generation"] / "manifest.json").read_text()
    )
    assert manifest["missing_records"] == [
        {"record": "assessment_history:" + "a" * 64, "reason": "original receipt files unavailable"}
    ]
    assert manifest["research_source_artifacts_included"] is False
    assert Archive(tmp_path / "archive").read()["state"]["archived"]
