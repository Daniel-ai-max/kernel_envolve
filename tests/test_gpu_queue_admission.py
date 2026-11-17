"""CPU-only admission regressions; retain real SQLite Store.claim and lifecycle writes."""

import contextlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from kernel_envolve.manager import gpu
from kernel_envolve.manager.store import Store

PROBLEM = "GPU cleanup unconfirmed: orphan worker"


@pytest.fixture
def admission(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    store.db.execute(
        "INSERT INTO resources(id,spec) VALUES(?,?)",
        ("r", json.dumps({"device": "GPU-aabb", "sessions": 1})),
    )
    spec = {"resource": "r", "wall_budget_seconds": None, "gpu_budget_seconds": None}
    store.db.execute(
        "INSERT INTO tasks(id,spec,created) VALUES(?,?,?)", ("t", json.dumps(spec), time.time())
    )
    parent = store.enqueue(store.task("t"), "agent", ["not-executed"])
    assert store.claim(parent, os.getpid(), "test-birth")
    calls = {"locks": 0, "probes": 0, "launches": 0, "waits": 0}

    def locks(_):
        calls["locks"] += 1
        return contextlib.nullcontext((object(),))

    def probe(_):
        calls["probes"] += 1

    def supervise(current_store, key, _):
        calls["launches"] += 1
        assert current_store.job(key)["status"] == "running"
        current_store.finish(key, "succeeded", exit_code=0)
        return 0

    def unexpected_wait(_):
        raise AssertionError("Unexpected wait; quarantine should fail immediately")

    monkeypatch.setattr(gpu, "identity", lambda _: "test-birth")
    monkeypatch.setattr(gpu, "alive", lambda *args: True)
    monkeypatch.setattr(gpu, "gpu_locks", locks)
    monkeypatch.setattr(gpu, "gpu_group_idle", probe)
    monkeypatch.setattr(gpu, "supervise", supervise)
    monkeypatch.setattr(gpu, "time", SimpleNamespace(time=time.time, sleep=unexpected_wait))
    monkeypatch.delenv("KERNEL_ENVOLVE_GPU_JOB", raising=False)
    yield store, parent, calls
    store.close()


def quarantine(store):
    store.db.execute("UPDATE resources SET problem=? WHERE id='r'", (PROBLEM,))


def rejected(store, capsys):
    response = json.loads(capsys.readouterr().out)
    assert response["error_type"] == "gpu_resource_quarantined"
    assert response["resource"] == "r"
    assert response["problem"] == PROBLEM
    assert PROBLEM in response["error"]
    assert store.resource("r")["problem"] == PROBLEM
    assert store.task("t")["gpu_seconds"] == 0
    for job in store.rows("SELECT * FROM jobs WHERE kind='gpu'"):
        assert job["status"] == "cancelled"
        assert PROBLEM in job["error"]
        assert job["child_pid"] is None


def test_existing_quarantine_rejects_before_enqueue(admission, capsys):
    store, parent, calls = admission
    quarantine(store)
    assert gpu.run(store.root, "t", parent, ["must-not-run"]) == 2
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")
    assert calls == {"locks": 0, "probes": 0, "launches": 0, "waits": 0}
    rejected(store, capsys)


@pytest.mark.parametrize("end_parent", [False, True])
def test_quarantine_while_waiting_retains_cause(admission, monkeypatch, capsys, end_parent):
    store, parent, calls = admission
    monkeypatch.setattr(gpu, "gpu_locks", lambda _: contextlib.nullcontext(None))

    def wait(_):
        calls["waits"] += 1
        assert calls["waits"] == 1
        quarantine(store)
        if end_parent:
            store.db.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (parent,))

    monkeypatch.setattr(gpu.time, "sleep", wait)
    assert gpu.run(store.root, "t", parent, ["must-not-run"]) == 2
    assert calls["probes"] == calls["launches"] == 0
    assert calls["waits"] == 1
    rejected(store, capsys)


@pytest.mark.parametrize("older_status", ["queued", "running"])
def test_ineligible_waiter_does_not_lock_or_probe(admission, monkeypatch, older_status):
    store, parent, calls = admission
    older = store.enqueue(store.task("t"), "gpu", ["older"], parent)
    store.db.execute("UPDATE jobs SET created=created-1 WHERE id=?", (older,))
    if older_status == "running":
        assert store.claim(older, os.getpid(), "test-birth")

    def wait(_):
        calls["waits"] += 1
        assert calls == {"locks": 0, "probes": 0, "launches": 0, "waits": 1}
        store.finish(older, "cancelled", "fixture predecessor ended")

    monkeypatch.setattr(gpu.time, "sleep", wait)
    assert gpu.run(store.root, "t", parent, ["mock-command"]) == 0
    assert calls == {"locks": 1, "probes": 1, "launches": 1, "waits": 1}


def test_quarantine_during_lock_acquisition_prevents_probe(admission, monkeypatch, capsys):
    store, parent, calls = admission

    @contextlib.contextmanager
    def lock(_):
        quarantine(store)
        yield (object(),)

    monkeypatch.setattr(gpu, "gpu_locks", lock)
    assert gpu.run(store.root, "t", parent, ["must-not-run"]) == 2
    assert calls["probes"] == calls["launches"] == 0
    rejected(store, capsys)


def test_quarantine_during_probe_still_fails_real_claim(admission, monkeypatch, capsys):
    store, parent, calls = admission
    monkeypatch.setattr(gpu, "gpu_group_idle", lambda _: quarantine(store))
    assert gpu.run(store.root, "t", parent, ["must-not-run"]) == 2
    assert calls["launches"] == 0
    job = store.one("SELECT * FROM jobs WHERE kind='gpu'")
    assert job["started"] is None  # Real Store.claim refused after the advisory checks.
    rejected(store, capsys)


def test_quarantine_after_claim_prevents_launch(admission, monkeypatch, capsys):
    store, parent, calls = admission
    real_claim = Store.claim

    def claim(self, *args):
        claimed = real_claim(self, *args)
        if claimed:
            quarantine(store)
        return claimed

    monkeypatch.setattr(Store, "claim", claim)
    assert gpu.run(store.root, "t", parent, ["must-not-run"]) == 2
    assert calls["launches"] == 0
    assert store.one("SELECT started FROM jobs WHERE kind='gpu'")["started"] is not None
    rejected(store, capsys)
