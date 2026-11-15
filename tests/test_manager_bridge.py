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
    assert store.job(job)["status"] == "yielded"
    assert store.task("toy")["failures"] == 0
    assert proc.returncode == 0
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


@pytest.mark.parametrize(
    "fault",
    ["none", "bridge_after_leader_exit", "cpu_supervisor", "cpu_supervisor_after_leader_exit"],
)
def test_cpu_success_cleans_descendants_before_receipt(managed, fault):
    import subprocess

    store, root, job, proc, directory = start(managed)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    marker = directory / "ordinary-child.pid"
    script = f"import subprocess,sys,time; from pathlib import Path; child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); Path({str(marker)!r}).write_text(str(child.pid)); time.sleep(0.5)"
    try:
        send(store, root, job, "fork-exit", cpu(script, 5), wait=0)
        wait_for(marker.exists)
        descendant = int(marker.read_text())
        birth = identity(descendant)
        registered = store.rows("SELECT * FROM processes WHERE job=?", (job,))
        assert len(registered) == 1
        if fault == "bridge_after_leader_exit":
            bridge_pid = store.job(job)["child_pid"]
            os.kill(bridge_pid, signal.SIGSTOP)
            wait_for(lambda: (directory / "logs/fork-exit.exit.json").exists())
            assert alive(descendant, birth)
            assert alive(registered[0]["pid"], registered[0]["identity"])
            os.kill(bridge_pid, signal.SIGKILL)
            proc.communicate(timeout=10)
            receipt = send(store, root, job, "fork-exit", wait=0)
            assert receipt["status"] == "interrupted_without_receipt"
        else:
            if fault == "cpu_supervisor_after_leader_exit":
                bridge_pid = store.job(job)["child_pid"]
                os.kill(bridge_pid, signal.SIGSTOP)
                wait_for(lambda: (directory / "logs/fork-exit.exit.json").exists())
                os.kill(registered[0]["pid"], signal.SIGKILL)
                wait_for(lambda: not alive(registered[0]["pid"], registered[0]["identity"]))
                os.kill(bridge_pid, signal.SIGCONT)
            elif fault == "cpu_supervisor":
                os.kill(registered[0]["pid"], signal.SIGKILL)
            receipt = send(store, root, job, "fork-exit")
            assert receipt["status"] == ("succeeded" if fault == "none" else "failed")
            assert not alive(descendant, birth)  # Clean before receipt, not at round end.
            consume(store, root, job, "fork-exit", receipt)
            following = send(store, root, job, "next", cpu("print('next command')"))
            assert following["status"] == "succeeded"
            assert not alive(descendant, birth)
            store.control("toy", "pause")
            proc.communicate(timeout=10)
        wait_for(lambda: not alive(descendant, birth))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_wait_renews_short_host_lease_without_disconnect(managed):
    store, root, job, proc, directory = start(managed, idle=1)
    receipt = send(
        store, root, job, "slow", cpu("import time;time.sleep(1.5);print('result')"), wait=4
    )
    assert receipt["status"] == "succeeded"
    assert json.loads((directory / "context.json").read_text())["idle_seconds"] == 1
    consume(store, root, job, "slow", receipt)
    send(
        store,
        root,
        job,
        "done",
        {"kind": "finish", "report": {"next_action": "next distinct input"}},
    )
    proc.communicate(timeout=8)
    assert store.job(job)["status"] == "succeeded"


def test_sixty_second_wait_throttles_durable_host_writes(tmp_path, monkeypatch):
    directory = bridge.folder(tmp_path, "task", "round")
    for name in ("requests", "results"):
        (directory / name).mkdir(parents=True)
    bridge.atomic(
        directory / "context.json", {"owner": "host", "status": "ready", "idle_seconds": 120}
    )
    bridge.atomic(directory / "requests/request.json", {})
    writes = []
    original = bridge.atomic

    def counted(path, value):
        if path.name == "host.json":
            writes.append(value)
        original(path, value)

    clock = [0.0]
    with monkeypatch.context() as patch:
        patch.setattr(bridge, "active", lambda *_: True)
        patch.setattr(bridge, "atomic", counted)
        patch.setattr(bridge.time, "monotonic", lambda: clock[0])
        patch.setattr(
            bridge.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        )
        receipt = bridge.request(
            None, tmp_path, "task", "round", "host", "request", wait_seconds=60
        )
        assert receipt["status"] == "pending"
        assert 11 <= len(writes) <= 13
        original(directory / "results/request.json", {"status": "succeeded"})
        writes.clear()
        assert (
            bridge.request(None, tmp_path, "task", "round", "host", "request", wait_seconds=60)[
                "status"
            ]
            == "succeeded"
        )
        assert not writes  # Reading a completed result is not a host keepalive loop.


@pytest.mark.parametrize("desired", ["active", "paused", "budget", "superseded"])
def test_closed_round_consumption_is_visible_without_restarting_task(managed, desired):
    from kernel_envolve.manager.views import detail, overview

    store, root, job, proc, _ = start(managed, idle=1)
    receipt = send(store, root, job, "actual-result", cpu("print('actual experiment output')"))
    proc.communicate(timeout=8)
    assert store.job(job)["status"] == "yielded"
    store.db.execute("UPDATE tasks SET desired=? WHERE id='toy'", (desired,))
    protected = ("desired", "next_at", "failures", "wall_seconds", "gpu_seconds")
    before = {name: store.task("toy")[name] for name in protected}
    consume(store, root, job, "actual-result", receipt)
    view = overview(store)["tasks"][0]
    assert view["research_record"]["outcome"] == "Observed subprocess result"
    archived = next(row for row in detail(store, "toy")["rounds"] if row["id"] == job)
    assert archived["report"]["report"]["outcome"] == "Observed subprocess result"
    assert {name: store.task("toy")[name] for name in protected} == before
    events = len(store.rows("SELECT seq FROM events WHERE kind='report'"))
    consume(store, root, job, "actual-result", receipt)
    assert len(store.rows("SELECT seq FROM events WHERE kind='report'")) == events
    assert not store.schedule()


def test_late_old_round_never_overwrites_newer_round_report(managed):
    from kernel_envolve.manager.views import detail

    store, root, old_job, proc, _ = start(managed, idle=1)
    receipt = send(store, root, old_job, "old-result", cpu("print('old')"))
    proc.communicate(timeout=8)
    bridge.open_host(store, "toy", "foreground", "new-round")
    new_job = store.schedule()[0]
    assert store.claim(new_job, os.getpid(), identity(os.getpid()))
    store.report(
        "toy",
        new_job,
        {"phase": "new experiment", "outcome": "newer evidence", "next_action": "new plan"},
    )
    consume(store, root, old_job, "old-result", receipt)
    assert store.task("toy")["report"]["outcome"] == "newer evidence"
    archived = next(row for row in detail(store, "toy")["rounds"] if row["id"] == old_job)
    assert archived["report"]["report"]["outcome"] == "Observed subprocess result"
    assert store.job(new_job)["status"] == "running"
    store.finish(new_job, "cancelled")


def test_replaying_legacy_consumption_repairs_projection_once(managed):
    store, root, job, proc, directory = start(managed, idle=1)
    receipt = send(store, root, job, "legacy", cpu("print('legacy')"))
    proc.communicate(timeout=8)
    # The old release recorded idle expiry as an execution failure. Keep that evidence.
    store.db.execute(
        "UPDATE jobs SET status='failed',exit_code=2,error='legacy idle exit' WHERE id=?", (job,)
    )
    store.db.execute("UPDATE tasks SET failures=3,next_at=? WHERE id='toy'", (time.time() + 300,))
    before = store.task("toy")
    record = {
        "request": "legacy",
        "receipt_sha256": receipt["receipt_sha256"],
        "owner": "foreground",
        "outcome": "Observed subprocess result",
        "next_action": "Run the next distinct hypothesis",
    }
    bridge.atomic(directory / "consumed/legacy.json", record)
    store.event("toy", "host_consumed_result", record | {"round": job})
    original_at = store.one("SELECT MAX(at) AS at FROM events WHERE kind='host_consumed_result'")[
        "at"
    ]
    assert not store.task("toy")["report"].get("outcome")
    consume(store, root, job, "legacy", receipt)
    assert store.task("toy")["report"]["outcome"] == record["outcome"]
    saved = store.one("SELECT data FROM metadata WHERE key=?", ("report_progress:toy",))
    assert json.loads(saved["data"]) == original_at
    assert len(store.rows("SELECT seq FROM events WHERE kind='host_consumed_result'")) == 1
    assert store.job(job)["status"] == "failed" and store.job(job)["exit_code"] == 2
    assert store.task("toy")["failures"] == 3
    assert store.task("toy")["next_at"] == before["next_at"]


@pytest.mark.parametrize("code", [2, bridge.IDLE_EXIT_CODE])
def test_exit_code_alone_cannot_exempt_real_failure(managed, code):
    store, _, _ = managed
    replace_command(store, [sys.executable, "-c", f"raise SystemExit({code})"], host_driven=True)
    bridge.open_host(store, "toy", "foreground", "not-a-bridge")
    job = store.schedule()[0]
    proc = launch(store, job)
    proc.communicate(timeout=8)
    assert store.job(job)["status"] == "failed"
    assert store.task("toy")["failures"] == 1
    assert store.task("toy")["next_at"] > time.time()


def test_repeated_idle_yields_do_not_accumulate_failures_or_reopen(managed):
    store, root, job, proc, _ = start(managed, idle=1)
    for index in range(3):
        proc.communicate(timeout=8)
        assert store.job(job)["status"] == "yielded"
        assert store.task("toy")["failures"] == 0
        assert not store.schedule()
        if index < 2:
            bridge.open_host(store, "toy", "foreground", f"fresh-{index}")
            job = store.schedule()[0]
            proc = launch(store, job)
    assert store.task("toy")["desired"] == "active"
    assert store.task("toy")["next_at"] <= time.time()


def test_idle_preserves_prior_real_failures_and_gpu_error_receipts(managed):
    store, root, job, proc, _ = start(managed, idle=1)
    store.db.execute("UPDATE tasks SET failures=2 WHERE id='toy'")
    receipt = send(store, root, job, "failed-test", cpu("raise SystemExit(2)", 5) | {"kind": "gpu"})
    assert receipt["status"] == "failed"
    gpu = store.one("SELECT status,exit_code FROM jobs WHERE kind='gpu'")
    assert gpu == {"status": "failed", "exit_code": 2}
    proc.communicate(timeout=8)
    assert store.job(job)["status"] == "yielded"
    assert store.task("toy")["failures"] == 2


def test_same_analysis_preserves_phase_and_progress_clock(managed):
    store, root, job, proc, _ = start(managed)
    store.report(
        "toy",
        job,
        {
            "phase": "diagnosis",
            "outcome": "Observed subprocess result",
            "next_action": "Run the next distinct hypothesis",
        },
    )
    before = store.one("SELECT data FROM metadata WHERE key='report_progress:toy'")["data"]
    receipt = send(store, root, job, "more-evidence", cpu("print('another result')"))
    consume(store, root, job, "more-evidence", receipt)
    assert store.task("toy")["report"]["phase"] == "diagnosis"
    assert store.one("SELECT data FROM metadata WHERE key='report_progress:toy'")["data"] == before
    store.control("toy", "pause")
    proc.communicate(timeout=8)


def test_cli_bridge_wait_default_and_explicit_nonblocking():
    from kernel_envolve.manager.cli import parser

    argv = [
        "bridge-request",
        "--root",
        "/tmp/bridge",
        "--task",
        "task",
        "--round",
        "round",
        "--owner",
        "host",
        "--id",
        "request",
    ]
    assert parser().parse_args(argv).wait_seconds == 30
    assert parser().parse_args([*argv, "--wait-seconds", "0"]).wait_seconds == 0


def test_wait_returns_on_completion_without_sleeping_remaining_budget(tmp_path, monkeypatch):
    directory = bridge.folder(tmp_path, "task", "round")
    bridge.atomic(directory / "context.json", {"owner": "host", "status": "ready"})
    bridge.atomic(directory / "requests/request.json", {})
    clock = [0.0]
    monkeypatch.setattr(bridge, "active", lambda *_: True)
    monkeypatch.setattr(bridge.time, "monotonic", lambda: clock[0])

    def advance(seconds):
        clock[0] += seconds
        if clock[0] >= 0.3:
            bridge.atomic(directory / "results/request.json", {"status": "succeeded"})

    monkeypatch.setattr(bridge.time, "sleep", advance)
    receipt = bridge.request(None, tmp_path, "task", "round", "host", "request", wait_seconds=30)
    assert receipt["status"] == "succeeded"
    assert 0.3 <= clock[0] < 0.5


def test_cli_consumes_and_submits_once_even_after_lost_reply(managed, tmp_path, capsys):
    from kernel_envolve.manager.cli import main

    store, root, job, proc, directory = start(managed)
    receipt = send(store, root, job, "first", cpu("print('evidence')"))
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "request": "first",
                "receipt_sha256": receipt["receipt_sha256"],
                "outcome": "Observed evidence",
                "next_action": "Measure second hypothesis",
            }
        )
    )
    marker = tmp_path / "executed"
    payload = tmp_path / "next.json"
    payload.write_text(
        json.dumps(
            cpu(
                f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')"
            )
        )
    )
    argv = [
        "--state",
        str(store.root),
        "bridge-request",
        "--root",
        str(root),
        "--task",
        "toy",
        "--round",
        job,
        "--owner",
        "foreground",
        "--id",
        "second",
        "--file",
        str(payload),
        "--consume-file",
        str(analysis),
    ]
    main(argv)
    first = json.loads(capsys.readouterr().out)
    main(argv)  # A lost transport response is safe to reconcile with the same IDs.
    second = json.loads(capsys.readouterr().out)
    assert first == second and first["status"] == "succeeded"
    assert marker.read_text() == "x"
    assert len(list((directory / "claims").glob("second.json"))) == 1
    assert len(store.rows("SELECT seq FROM events WHERE kind='host_consumed_result'")) == 1
    assert store.task("toy")["report"]["outcome"] == "Observed evidence"
    store.control("toy", "pause")
    proc.communicate(timeout=8)


def test_cli_combined_request_preserves_checks_and_partial_consumption(managed, tmp_path):
    from kernel_envolve.manager.cli import main

    store, root, job, proc, directory = start(managed)
    receipt = send(store, root, job, "first", cpu("print('evidence')"))
    analysis = tmp_path / "analysis.json"
    previous = {
        "request": "first",
        "receipt_sha256": "wrong",
        "outcome": "Observed evidence",
        "next_action": "Measure next hypothesis",
    }
    analysis.write_text(json.dumps(previous))
    payload = tmp_path / "next.json"
    payload.write_text(json.dumps({"kind": "invalid"}))
    argv = [
        "--state",
        str(store.root),
        "bridge-request",
        "--root",
        str(root),
        "--task",
        "toy",
        "--round",
        job,
        "--owner",
        "foreground",
        "--id",
        "second",
        "--file",
        str(payload),
        "--consume-file",
        str(analysis),
    ]
    with pytest.raises(ValueError, match="match the completed"):
        main(argv)
    assert not (directory / "consumed/first.json").exists()
    assert not (directory / "requests/second.json").exists()
    previous["receipt_sha256"] = receipt["receipt_sha256"]
    analysis.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="kind must"):
        main(argv)
    assert (directory / "consumed/first.json").exists()
    assert not (directory / "requests/second.json").exists()
    payload.write_text(json.dumps(cpu("print('repaired request')")))
    main(argv)
    assert len(store.rows("SELECT seq FROM events WHERE kind='host_consumed_result'")) == 1
    previous["outcome"] = "Changed analysis must not be silently accepted"
    analysis.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="different analysis"):
        main(argv)
    store.control("toy", "pause")
    proc.communicate(timeout=8)
