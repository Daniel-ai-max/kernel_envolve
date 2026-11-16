"""Recovery uses saved evidence on another host, without workspaces or executors."""

import json
import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.manager import archive
from kernel_envolve.manager.evidence import refresh
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.web import server

pytest_plugins = ["test_manager"]


def test_portable_history_retains_best_notes_ledger_and_rejects_writes(managed, tmp_path):
    store, _, work = managed
    run, result = evaluate(work, "full")
    promote(run)
    refresh(store)
    (work / "notes.md").write_text("Saved research conclusion")
    archive.capture_notes(store)
    store.control("toy", "pause")
    target = tmp_path / "archive"
    exported = archive.export_archive(store, target)
    original = archive.Archive(target).read()
    assert original["state"]["tasks"][0]["best"]["run"] == str(run)
    assert original["details"]["toy"]["notes"] == "Saved research conclusion"
    shutil.rmtree(work)
    moved = tmp_path / "moved"
    shutil.move(target, moved)
    restored = archive.Archive(moved).read()
    assert restored == original
    # No migration, writable Store, job scheduling or evidence reindex on replay.
    with server(moved, 0, archive=True) as http:
        thread = threading.Thread(target=http.serve_forever)
        thread.start()
        url = f"http://127.0.0.1:{http.server_port}"
        try:
            with urllib.request.urlopen(url + "/api/state?task=toy") as response:
                data = json.load(response)
            assert data["detail"]["best"] == restored["details"]["toy"]["best"]
            assert data["archived"] and "token" not in data
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url + "/api/control", data=b"{}"))
            assert error.value.code == 405
        finally:
            http.shutdown()
            thread.join()
    # The portable SQLite keeps full history, not just the table's display limit.
    saved = Store(moved / exported["generation"], readonly=True)
    try:
        assert saved.one("SELECT COUNT(*) AS n FROM runs")["n"] == 1
        assert saved.task("toy")["desired"] == "paused"
    finally:
        saved.close()


def test_publish_failure_keeps_previous_archive_and_checksum_detects_damage(
    managed, tmp_path, monkeypatch
):
    store, _, _ = managed
    target = tmp_path / "archive"
    archive.export_archive(store, target)
    pointer = (target / "CURRENT.json").read_bytes()
    real = archive.durable_json

    def interrupt(path, data):
        if Path(path).name == "CURRENT.json":
            raise OSError("simulated crash before publish")
        return real(path, data)

    monkeypatch.setattr(archive, "durable_json", interrupt)
    with pytest.raises(OSError, match="simulated crash"):
        archive.export_archive(store, target)
    assert (target / "CURRENT.json").read_bytes() == pointer
    assert archive.Archive(target).read()["state"]["archived"]
    generation = json.loads(pointer)["generation"]
    (target / generation / "view.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        archive.Archive(target)


def test_checkpoint_yields_to_gpu_and_unchanged_records_do_not_reexport(managed, monkeypatch):
    store, _, _ = managed
    from kernel_envolve.manager import evidence

    monkeypatch.setattr(evidence, "gpu_work_pending", lambda _: True)
    assert archive.checkpoint(store, force=True) is False
    assert not (store.root / "dashboard-archive").exists()
    monkeypatch.setattr(evidence, "gpu_work_pending", lambda _: False)
    assert archive.checkpoint(store, force=True)
    pointer = (store.root / "dashboard-archive/CURRENT.json").read_bytes()
    assert not archive.checkpoint(store, force=True)
    assert (store.root / "dashboard-archive/CURRENT.json").read_bytes() == pointer
    store.control("toy", "pause")
    assert archive.checkpoint(store, force=True)
    assert len(list((store.root / "dashboard-archive").glob("snapshot-*"))) == 2
    assert (
        archive.Archive(store.root / "dashboard-archive").read()["state"]["tasks"][0]["phase"]
        == "paused"
    )


def test_archive_live_work_is_labeled_history_not_still_running(managed, tmp_path):
    store, _, _ = managed
    archive.export_archive(store, tmp_path / "archive")
    view = archive.Archive(tmp_path / "archive").read()
    assert view["state"]["tasks"][0]["phase"] == "archived"
    assert view["state"]["tasks"][0]["execution_state"] == "archived"
    assert view["state"]["coordinator"] is None


def test_archive_reader_reloads_atomic_new_generation(managed, tmp_path):
    store, _, _ = managed
    root = tmp_path / "archive"
    archive.export_archive(store, root)
    reader = archive.Archive(root)
    assert reader.read()["state"]["tasks"][0]["desired"] == "active"
    store.control("toy", "pause")
    archive.export_archive(store, root)
    assert reader.read()["state"]["tasks"][0]["desired"] == "paused"


def test_terminal_snapshot_and_explicit_exports_are_not_pruned(managed):
    store, _, _ = managed
    root = store.root / "dashboard-archive"
    store.control("toy", "pause")
    assert archive.checkpoint(store, force=True)
    final = json.loads((root / "CURRENT.json").read_text())["generation"]
    store.control("toy", "resume")
    for n in range(4):
        store.event("toy", "fixture_changed", {"n": n})
        assert archive.checkpoint(store, force=True)
    assert (root / final).is_dir()
    assert len(list(root.glob("snapshot-*"))) == 3
    assert archive.Archive(root / final).read()["state"]["tasks"][0]["desired"] == "paused"
