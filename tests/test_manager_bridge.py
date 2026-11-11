"""Foreground model handoffs use real subprocess receipts, never bridge liveness."""

import json
import os
import signal
import sys
import time

import pytest
from test_manager import launch, replace_command, wait_for

from kernel_envolve.manager import bridge
from kernel_envolve.manager.process import alive, exclusive, identity, lock_path

pytest_plugins = ["test_manager"]


def start(managed, idle=120, budget=None):
    store, _, work = managed
    root = store.root.parent / "bridge"
    replace_command(
        store,
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "manage",
            "bridge",
            "--root",
            str(root),
            "--idle-seconds",
            str(idle),
        ],
        host_driven=True,
        round_timeout_seconds=30,
        gpu_budget_seconds=budget,
    )
    assert store.schedule() == []
    bridge.open_host(store, "toy", "foreground", "claim1")
    job = store.schedule()[0]
    assert bridge.open_host(store, "toy", "foreground", "claim1")["round"] == job
    proc = launch(store, job)
    directory = bridge.folder(root, "toy", job)
    wait_for(lambda: (directory / "context.json").exists())
    return store, root, job, proc, directory


def send(store, root, job, rid, value=None, wait=5):
    return bridge.request(store, root, "toy", job, "foreground", rid, value, wait)


def consume(store, root, job, rid, receipt):
    return bridge.consume(
        store,
        root,
        "toy",
        job,
        "foreground",
        rid,
        receipt["receipt_sha256"],
        "Observed subprocess result",
        "Run the next distinct hypothesis",
    )


def cpu(script, seconds=3):
    return {"kind": "cpu", "argv": [sys.executable, "-c", script], "expected_seconds": seconds}


def test_result_consume_next_request_and_finish(managed):
    store, root, job, proc, directory = start(managed)
    payload = cpu("print('first-result')")
    receipt = send(store, root, job, "first", payload)
    assert receipt["status"] == "succeeded" and receipt["stdout"] == "first-result\n"
    assert send(store, root, job, "first", payload) == receipt
    with pytest.raises(ValueError, match="different content"):
        send(store, root, job, "first", cpu("print('changed')"))
    with pytest.raises(ValueError, match="Consume"):
        send(store, root, job, "second", cpu("print('second')"))
    with pytest.raises(ValueError, match="match the completed"):
        bridge.consume(store, root, "toy", job, "foreground", "first", "wrong-hash", "read", "next")
    assert consume(store, root, job, "first", receipt) == consume(
        store, root, job, "first", receipt
    )
    assert store.task("toy")["report"]["outcome"] == "Observed subprocess result"
    # Exercise the managed GPU path on the CPU fixture, without allocating a GPU.
    receipt = send(
        store, root, job, "second", cpu("print('managed-result')", 10) | {"kind": "gpu"}, 15
    )
    assert receipt["status"] == "succeeded" and "managed-result" in receipt["stdout"]
    assert len(store.rows("SELECT id FROM jobs WHERE kind='gpu'")) == 1
    consume(store, root, job, "second", receipt)
    receipt = send(
        store,
        root,
        job,
        "finish",
        {
            "kind": "finish",
            "report": {"outcome": "two results consumed", "next_action": "implement new tile size"},
        },
    )
    assert receipt["status"] == "succeeded"
    proc.communicate(timeout=10)
    assert store.job(job)["status"] == "succeeded"
    assert store.task("toy")["report"]["next_action"] == "implement new tile size"
    store.db.execute("UPDATE tasks SET next_at=0")
    assert store.schedule() == []
    bridge.open_host(store, "toy", "foreground", "claim1")
    assert store.schedule() == []  # Used claim cannot open empty new rounds.
    assert len(list((directory / "claims").glob("*.json"))) == 3


def test_idle_exit_does_not_auto_reopen(managed):
    store, root, job, proc, directory = start(managed, idle=1)
    proc.communicate(timeout=8)
    assert store.job(job)["status"] == "failed"
    assert json.loads((directory / "context.json").read_text())["status"] == "closed"
    store.db.execute("UPDATE tasks SET next_at=0")
    assert store.schedule() == []
    assert store.task("toy")["desired"] == "active"
    assert any(r["kind"] == "host_disconnected" for r in store.rows("SELECT kind FROM events"))


def test_command_failure_is_consumable_then_repaired(managed):
    store, root, job, proc, _ = start(managed)
    receipt = send(
        store,
        root,
        job,
        "missing",
        {"kind": "cpu", "argv": ["/no/such/executable"], "expected_seconds": 1},
    )
    assert receipt["status"] == "failed" and "error" in receipt
    consume(store, root, job, "missing", receipt)
    receipt = send(store, root, job, "repair", cpu("print('repaired')"))
    assert receipt["status"] == "succeeded"
    consume(store, root, job, "repair", receipt)
    send(
        store,
        root,
        job,
        "done",
        {"kind": "finish", "report": {"next_action": "continue from repaired environment"}},
    )
    proc.communicate(timeout=8)


@pytest.mark.parametrize("stop", ["timeout", "pause", "crash"])
def test_interruption_never_replays_claimed_request(managed, stop):
    store, root, job, proc, directory = start(managed)
    seconds = 0.25 if stop == "timeout" else 20
    send(store, root, job, "once", cpu("import time; time.sleep(20)", seconds), wait=0)
    wait_for(lambda: (directory / "claims/once.json").exists())
    if stop == "pause":
        store.control("toy", "pause")
    elif stop == "crash":
        # Kill the bridge child; its supervisor must retain uncertainty rather than replay.
        import subprocess

        children = subprocess.check_output(["pgrep", "-P", str(proc.pid)], text=True).split()
        os.kill(int(children[0]), signal.SIGKILL)
    proc.communicate(timeout=10)
    receipt = send(store, root, job, "once", wait=0)
    assert receipt["status"] in ("failed", "interrupted_without_receipt")
    old_jobs = len(store.rows("SELECT id FROM jobs"))
    send(store, root, job, "once", cpu("import time; time.sleep(20)", seconds), wait=0)
    store.db.execute("UPDATE tasks SET next_at=0")
    assert not store.schedule()
    assert len(store.rows("SELECT id FROM jobs")) == old_jobs
    if stop == "pause":
        with pytest.raises(ValueError, match="active host_driven"):
            bridge.open_host(store, "toy", "foreground", "new-claim")
        assert store.task("toy")["desired"] == "paused"


def test_duration_and_ownership_checks(managed):
    store, root, job, proc, _ = start(managed, budget=2)
    with pytest.raises(ValueError, match="owner"):
        bridge.request(store, root, "toy", job, "other", "x", cpu("pass"))
    with pytest.raises(ValueError, match="expected_seconds"):
        send(store, root, job, "x", {"kind": "gpu", "argv": [sys.executable, "-c", "pass"]})
    receipt = send(store, root, job, "too-long", cpu("print('MUST_NOT_RUN')", 5) | {"kind": "gpu"})
    assert receipt["status"] == "failed" and "gpu_duration_admission" in receipt["stdout"]
    assert "MUST_NOT_RUN" not in receipt["stdout"]
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")
    store.control("toy", "pause")
    proc.communicate(timeout=10)
    before = store.task("toy")["wall_seconds"]
    consume(store, root, job, "too-long", receipt)
    assert store.task("toy")["desired"] == "paused" and store.task("toy")["wall_seconds"] == before


def test_legacy_registration_and_terminal_admission(managed):
    store, deployment, _ = managed
    spec = store.task("toy")["spec"]
    spec.pop("host_driven")
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(spec),))
    store.register(deployment)  # Old saved specs remain idempotent on upgrade.
    spec.update(host_driven=True)
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(spec),))
    for desired in ("paused", "budget", "superseded"):
        store.db.execute("UPDATE tasks SET desired=?", (desired,))
        with pytest.raises(ValueError, match="active host_driven"):
            bridge.open_host(store, "toy", "foreground", desired)
        assert not store.schedule()
        assert store.task("toy")["desired"] == desired
    store.db.execute("UPDATE tasks SET desired='active',gpu_seconds=10")
    spec["gpu_budget_seconds"] = 10
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(spec),))
    with pytest.raises(ValueError, match="Budget exhausted"):
        bridge.open_host(store, "toy", "foreground", "exhausted")


def test_expired_or_conflicting_claim_never_schedules(managed, monkeypatch):
    store, _, _ = managed
    replace_command(store, [sys.executable, "-c", "pass"], host_driven=True)
    first = bridge.open_host(store, "toy", "foreground", "first")
    with pytest.raises(ValueError, match="another host"):
        bridge.open_host(store, "toy", "other", "first")
    with pytest.raises(ValueError, match="unconsumed"):
        bridge.open_host(store, "toy", "foreground", "second")
    monkeypatch.setattr("kernel_envolve.manager.bridge.time.time", lambda: first["expires_at"] + 1)
    assert not store.schedule()
    bridge.open_host(store, "toy", "foreground", "first")
    assert not store.schedule()
    bridge.open_host(store, "toy", "foreground", "second")
    assert len(store.schedule()) == 1


def test_gpu_queue_wait_is_not_an_execution_timeout(managed):
    store, root, job, proc, _ = start(managed)
    with exclusive(lock_path("gpu", "cpu")) as lock:
        assert lock is not None
        send(
            store, root, job, "queued", cpu("print('ran-after-queue')", 1) | {"kind": "gpu"}, wait=0
        )
        wait_for(lambda: store.one("SELECT id FROM jobs WHERE kind='gpu' AND status='queued'"))
        time.sleep(1.3)
        assert store.job(job)["status"] == "running"
    receipt = send(store, root, job, "queued")
    assert receipt["status"] == "succeeded" and "ran-after-queue" in receipt["stdout"]
    consume(store, root, job, "queued", receipt)
    send(
        store,
        root,
        job,
        "done",
        {"kind": "finish", "report": {"next_action": "next distinct kernel"}},
    )
    proc.communicate(timeout=8)


@pytest.mark.parametrize("stop", ["timeout", "crash"])
def test_cpu_descendants_are_reaped_without_touching_unrelated_process(managed, stop):
    import subprocess

    store, root, job, proc, directory = start(managed)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    marker = directory / "descendant.pid"
    script = f"import subprocess,sys,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); Path({str(marker)!r}).write_text(str(child.pid)); time.sleep(30)"
    try:
        send(store, root, job, "fork", cpu(script, 0.5 if stop == "timeout" else 20), wait=0)
        wait_for(marker.exists)
        descendant = int(marker.read_text())
        birth = identity(descendant)
        if stop == "crash":
            os.kill(store.job(job)["child_pid"], signal.SIGKILL)
        proc.communicate(timeout=10)
        wait_for(lambda: not alive(descendant, birth))
        assert unrelated.poll() is None
        assert send(store, root, job, "fork", wait=0)["status"] in (
            "failed",
            "interrupted_without_receipt",
        )
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_bridge_artifacts_cannot_count_as_implementation_changes(managed):
    store, _, work = managed
    root = work / "bridge"
    replace_command(
        store,
        [sys.executable, "-m", "kernel_envolve.cli", "manage", "bridge", "--root", str(root)],
        host_driven=True,
    )
    bridge.open_host(store, "toy", "foreground", "inside-workspace")
    job = store.schedule()[0]
    proc = launch(store, job)
    proc.communicate(timeout=8)
    assert store.job(job)["status"] == "failed"
    assert not root.exists()
    assert "outside task workspaces" in (store.root / "jobs" / job / "stderr.log").read_text()
