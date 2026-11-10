"""Lifecycle tests exercise independent processes, contention and durable recovery."""

import concurrent.futures
import copy
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.evidence import refresh
from kernel_envolve.manager.process import alive, exclusive, identity, lock_path, terminate
from kernel_envolve.manager.service import reconcile
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import detail, overview
from kernel_envolve.manager.web import server
from kernel_envolve.storage import config, contract


@pytest.fixture
def managed(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "task.json").write_text(
        json.dumps(
            {
                "name": "toy",
                "objective": "fixture CPU latency",
                "device": "cpu",
                "cases": [{"id": "small", "args": {"n": 16}, "weight": 1}],
                "quick_trials": 1,
                "full_trials": 3,
                "warmup": 1,
            }
        )
    )
    (work / "task.py").write_text(
        "import numpy as np\nimport time\ndef make_inputs(case,seed): return (np.ones(case['n']),)\n"
        "def reference(x): return x+1\ndef baseline(x):\n time.sleep(.003)\n return x+1\n"
    )
    (work / "candidate.py").write_text("def run(x): return x+1\n")
    spec = {
        "id": "toy",
        "name": "Toy operator",
        "workspace": str(work),
        "task_dir": str(work),
        "kind": "generic",
        "resource": "local",
        "command": [sys.executable, "-c", "pass"],
        "round_timeout_seconds": 10,
        "cooldown_seconds": 0.1,
        "baseline": {
            "repository": "https://example.test/repo",
            "revision": "fixture-v1",
            "entrypoint": "task.py:baseline",
            "introduced_on": None,
            "introduction_url": None,
            "frozen_on": "2026-11-10",
            "hardware": "CPU",
            "contract": contract(work, config(work)),
        },
    }
    deployment = validate(
        {
            "version": 1,
            "resources": [{"id": "local", "device": "cpu", "sessions": 2}],
            "tasks": [spec],
        },
        tmp_path,
    )
    store = Store(tmp_path / "state")
    store.register(deployment)
    yield store, deployment, work
    # Stop only this fixture's processes, never other tests or host processes.
    for job in store.rows("SELECT * FROM jobs WHERE status='running'"):
        if job["pid"] != os.getpid() and alive(job["pid"], job["identity"]):
            os.kill(job["pid"], signal.SIGTERM)
    store.close()


def wait_for(check, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = check()
        if result:
            return result
        time.sleep(0.03)
    raise AssertionError("Timed out waiting for a verified process/state transition")


def test_live_but_unproductive_rounds_are_visible(managed, monkeypatch):
    store, _, _ = managed
    clock = [time.time()]
    monkeypatch.setattr("kernel_envolve.manager.store.time.time", lambda: clock[0])
    job = store.schedule()[0]
    assert store.claim(job, os.getpid(), identity(os.getpid()))
    clock[0] += 301
    store.heartbeat(job, 301)
    view = overview(store)["tasks"][0]
    assert not view["stale"]  # Supervisor is healthy; research progress is a separate clock.
    assert view["progress_stale"] and view["phase"] == "awaiting_progress"
    original_progress = view["progress_as_of"]
    store.finish(job, "succeeded")
    clock[0] += 1
    next_job = store.schedule()[0]
    assert store.claim(next_job, os.getpid(), identity(os.getpid()))
    view = overview(store)["tasks"][0]
    assert view["progress_stale"] and view["progress_as_of"] == original_progress


def test_detail_reads_workspace_notes_when_task_directory_is_separate(managed):
    store, _, work = managed
    task_dir = work / "task"
    task_dir.mkdir()
    spec = store.task("toy")["spec"]
    spec["task_dir"] = str(task_dir)
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(spec),))
    assert detail(store, "toy")["notes"] == ""
    (work / "notes.md").write_text("workspace research checkpoint")
    assert detail(store, "toy")["notes"] == "workspace research checkpoint"
    (task_dir / "notes.md").write_text("task-specific checkpoint")
    assert detail(store, "toy")["notes"] == "task-specific checkpoint"
    (task_dir / "notes.md").write_text("x" * 40000)
    assert len(detail(store, "toy")["notes"]) == 32000


def test_first_session_has_grace_after_waiting_for_capacity(managed, monkeypatch):
    store, _, _ = managed
    clock = [time.time() + 600]
    monkeypatch.setattr("kernel_envolve.manager.store.time.time", lambda: clock[0])
    job = store.schedule()[0]
    assert store.claim(job, os.getpid(), identity(os.getpid()))
    assert not overview(store)["tasks"][0]["progress_stale"]
    clock[0] += 301
    assert overview(store)["tasks"][0]["progress_stale"]


def test_only_changed_reports_refresh_research_progress(managed, monkeypatch):
    store, _, _ = managed
    clock = [time.time()]
    monkeypatch.setattr("kernel_envolve.manager.store.time.time", lambda: clock[0])
    job = store.schedule()[0]
    assert store.claim(job, os.getpid(), identity(os.getpid()))
    report = {"phase": "inspect precision", "next_action": "run fixed seed diagnosis"}
    store.report("toy", job, report)
    reported_at = overview(store)["tasks"][0]["progress_as_of"]
    clock[0] += 301
    store.report("toy", job, report)
    store.report("toy", job, {})
    view = overview(store)["tasks"][0]
    assert view["progress_stale"] and view["progress_as_of"] == reported_at
    store.report("toy", job, {"outcome": "identified an FP32 reduction discrepancy"})
    view = overview(store)["tasks"][0]
    assert not view["progress_stale"] and view["phase"] == "agent_running"
    assert view["progress_as_of"] == clock[0]


def test_gpu_work_and_pause_do_not_report_stalled_agent(managed, monkeypatch):
    store, _, _ = managed
    clock = [time.time()]
    monkeypatch.setattr("kernel_envolve.manager.store.time.time", lambda: clock[0])
    parent = store.schedule()[0]
    assert store.claim(parent, os.getpid(), identity(os.getpid()))
    gpu = store.enqueue(store.task("toy"), "gpu", ["unused"], parent)
    clock[0] += 600
    assert overview(store)["tasks"][0]["phase"] == "waiting_gpu"
    assert not overview(store)["tasks"][0]["progress_stale"]
    assert store.claim(gpu, os.getpid(), identity(os.getpid()))
    clock[0] += 600
    assert overview(store)["tasks"][0]["phase"] == "gpu_running"
    assert not overview(store)["tasks"][0]["progress_stale"]
    store.finish(gpu, "succeeded")
    assert overview(store)["tasks"][0]["progress_as_of"] == clock[0]
    clock[0] += 301
    store.control("toy", "pause")
    assert not overview(store)["tasks"][0]["progress_stale"]


def test_legacy_report_clock_survives_upgrade_and_repeated_report(managed, monkeypatch):
    store, _, _ = managed
    clock = [time.time()]
    monkeypatch.setattr("kernel_envolve.manager.store.time.time", lambda: clock[0])
    job = store.schedule()[0]
    assert store.claim(job, os.getpid(), identity(os.getpid()))
    report = {"phase": "baseline preparation"}
    store.report("toy", job, report)
    store.db.execute("DELETE FROM metadata WHERE key='report_progress:toy'")
    reported_at = clock[0]
    clock[0] += 301
    assert overview(store)["tasks"][0]["progress_as_of"] == reported_at
    store.report("toy", job, report)
    assert overview(store)["tasks"][0]["progress_as_of"] == reported_at
    assert overview(store)["tasks"][0]["progress_stale"]


def launch(store, key):
    return subprocess.Popen(
        [sys.executable, "-m", "kernel_envolve.manager.worker", str(store.root), key],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def replace_command(store, command, **settings):
    # Fixture setup only: operator reconfiguration deliberately cannot mutate task identity/commands.
    task = store.task("toy")
    task["spec"].update(command=command, **settings)
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(task["spec"]),))


def test_config_rejects_ambiguous_identity_and_duplicate_workspaces(managed):
    _, cfg, _ = managed
    for mutate in (
        lambda c: c["resources"][0].update(device="cuda:0"),
        lambda c: c["resources"][0].update(sessions=3),
        lambda c: c["tasks"][0]["baseline"].update(introduced_on="2026-01-01"),
        lambda c: c["tasks"].append(c["tasks"][0] | {"id": "other"}),
    ):
        bad = copy.deepcopy(cfg)
        bad["tasks"][0].pop("identity")
        mutate(bad)
        with pytest.raises(ValueError):
            validate(bad, ".")


def test_registration_atomic_idempotent_and_baseline_immutable(managed):
    store, cfg, _ = managed
    store.register(cfg)
    assert len(store.rows("SELECT * FROM events")) == 1
    changed = copy.deepcopy(cfg)
    changed["tasks"][0]["baseline"]["revision"] = "new-baseline"
    changed["resources"].append({"id": "extra", "device": "GPU-123", "sessions": 1})
    with pytest.raises(ValueError):
        store.register(changed)
    assert not store.one("SELECT * FROM resources WHERE id='extra'")
    assert store.task("toy")["spec"]["baseline"]["revision"] == "fixture-v1"


def test_transactional_admission_under_competing_coordinators(managed):
    store, _, _ = managed

    def schedule():
        local = Store(store.root)
        try:
            return local.schedule()
        finally:
            local.close()

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        scheduled = list(pool.map(lambda _: schedule(), range(8)))
    assert sum(map(len, scheduled)) == 1
    assert len(store.rows("SELECT * FROM jobs")) == 1


def test_failure_backoff_budget_and_pause_survive_reopen(managed):
    store, _, _ = managed
    for attempt in range(3):
        store.db.execute("UPDATE tasks SET next_at=0")
        key = store.schedule()[0]
        store.finish(key, "failed", "provider unavailable")
        assert not store.schedule()
    assert store.task("toy")["desired"] == "blocked"
    store.control("toy", "pause")
    reopened = Store(store.root)
    assert reopened.task("toy")["desired"] == "paused"
    assert not reopened.schedule()
    reopened.close()
    store.control("toy", "configure", {"wall_budget_seconds": 1})
    store.control("toy", "resume")
    key = store.schedule()[0]
    assert store.claim(key, os.getpid(), identity(os.getpid()))
    store.heartbeat(key, 2)
    store.heartbeat(key, 1)  # Out-of-order heartbeats cannot refund elapsed work.
    store.finish(key, "succeeded")
    assert store.task("toy")["wall_seconds"] == 2
    assert store.task("toy")["desired"] == "budget"
    assert not store.schedule()


def test_actual_round_checkpoint_and_stale_report_refusal(managed):
    store, _, _ = managed
    replace_command(
        store,
        [
            sys.executable,
            "-c",
            "import os,json,pathlib,sys; p=sys.stdin.read(); assert 'Fixed baseline' in p; "
            "pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'hypothesis':'fuse loads','next_action':'test padded shape'}))",
        ],
    )
    key = store.schedule()[0]
    process = launch(store, key)
    out, err = process.communicate(timeout=10)
    assert process.returncode == 0, (out, err)
    assert store.job(key)["status"] == "succeeded"
    assert store.task("toy")["report"]["next_action"] == "test padded shape"
    assert store.task("toy")["desired"] == "active"
    with pytest.raises(ValueError):
        store.report("toy", key, {"hypothesis": "stale"})
    with pytest.raises(ValueError):
        store.report("toy", key, {"speedup": "2x"})


def test_live_supervisor_survives_reconciliation_and_pause_reaps_child(managed):
    store, _, _ = managed
    replace_command(store, [sys.executable, "-c", "import time; time.sleep(30)"])
    key = store.schedule()[0]
    process = launch(store, key)
    try:
        job = wait_for(lambda: j if (j := store.job(key))["child_pid"] else None)
        store.db.execute("UPDATE jobs SET heartbeat=0 WHERE id=?", (key,))
        reconcile(store)
        assert store.job(key)["status"] == "running"  # An expired heartbeat is not death.
        assert not store.schedule()
        store.control("toy", "pause")
        process.communicate(timeout=10)
        assert not alive(job["child_pid"], job["child_identity"])
        assert store.job(key)["status"] == "cancelled"
        assert store.task("toy")["desired"] == "paused"
    finally:
        terminate(process)


def test_killed_supervisor_recovers_owned_child_without_replay(managed):
    store, _, _ = managed
    replace_command(store, [sys.executable, "-c", "import time; time.sleep(30)"])
    key = store.schedule()[0]
    process = launch(store, key)
    job = wait_for(lambda: j if (j := store.job(key))["child_pid"] else None)
    process.kill()
    process.communicate(timeout=5)
    reconcile(store)
    assert store.job(key)["status"] == "interrupted"
    assert not alive(job["child_pid"], job["child_identity"])
    assert store.task("toy")["wall_seconds"] > 0
    assert not store.schedule()  # Recovery backoff, no immediate restart storm.


def test_unknown_process_probe_does_not_replace_worker(managed, monkeypatch):
    store, _, _ = managed
    key = store.schedule()[0]
    store.claim(key, 123, "unknown")
    monkeypatch.setattr(
        "kernel_envolve.manager.service.alive",
        lambda *a: (_ for _ in ()).throw(OSError("probe unavailable")),
    )
    with pytest.raises(OSError):
        reconcile(store)
    assert store.job(key)["status"] == "running"


def test_gpu_fifo_and_exclusion_are_transactional(managed):
    store, _, _ = managed
    parent = store.schedule()[0]
    store.claim(parent, os.getpid(), identity(os.getpid()))
    with store.transaction():
        one = store.enqueue(store.task("toy"), "gpu", ["true"], parent)
        two = store.enqueue(store.task("toy"), "gpu", ["true"], parent)
    assert not store.claim(two, os.getpid(), identity(os.getpid()))
    assert store.claim(one, os.getpid(), identity(os.getpid()))
    assert not store.claim(two, os.getpid(), identity(os.getpid()))
    store.finish(one, "succeeded")
    assert store.claim(two, os.getpid(), identity(os.getpid()))
    store.finish(two, "succeeded")
    store.finish(parent, "succeeded")


def test_host_lock_shared_across_state_directories(managed):
    _, _, work = managed
    path = lock_path("task", str(work))
    with exclusive(path) as handle:
        assert handle
        code = "from kernel_envolve.manager.process import exclusive; import sys;\nwith exclusive(sys.argv[1]) as h: sys.exit(1 if h else 0)"
        assert subprocess.run([sys.executable, "-c", code, str(path)], timeout=5).returncode == 0


def test_real_gpu_wrapper_serializes_two_commands_on_cpu_resource(managed):
    store, _, work = managed
    trace = work / "trace.jsonl"
    child = "import time,json,sys; p=sys.argv[1]; f=open(p,'a'); f.write(json.dumps(['start',time.time()])+'\\n'); f.flush(); time.sleep(.15); f.write(json.dumps(['end',time.time()])+'\\n'); f.close()"
    script = work / "agent.py"
    script.write_text(
        "import subprocess,sys\n"
        f"cmd=[sys.executable,'-m','kernel_envolve.cli','manage','gpu-run','--',sys.executable,'-c',{child!r},{str(trace)!r}]\n"
        "jobs=[subprocess.Popen(cmd) for _ in range(2)]\n"
        "assert all(p.wait()==0 for p in jobs)\n"
        "import os,json,pathlib; pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'next_action':'inspect next hypothesis'}))\n"
    )
    replace_command(store, [sys.executable, str(script)])
    key = store.schedule()[0]
    process = launch(store, key)
    out, err = process.communicate(timeout=15)
    assert process.returncode == 0, (out, err)
    marks = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [m[0] for m in marks] == ["start", "end", "start", "end"]
    assert len(store.rows("SELECT id FROM jobs WHERE kind='gpu' AND status='succeeded'")) == 2
    assert store.task("toy")["gpu_seconds"] > 0.3


def test_real_evidence_history_and_corruption_hide_best(managed):
    store, _, work = managed
    quick, _ = evaluate(work, "quick")
    refresh(store)
    assert overview(store)["tasks"][0]["best"] is None
    run, result = evaluate(work, "full")
    assert result["decision"]["eligible_for_best"], result["decision"]
    promote(run)
    refresh(store)
    task = detail(store, "toy")
    assert task["best"]["baseline_speedup"] > 1
    assert len(task["history"]) == 1
    assert task["history"][0]["promoted_at"]
    assert len(task["runs"]) == 2
    (run / "source" / "candidate.py").write_text("def run(x): return x\n")
    refresh(store)
    task = detail(store, "toy")
    assert task["best"] is None
    assert task["history"][0]["evidence_error"]
    assert task["evidence_error"]


def test_http_origin_controls_and_packaged_assets(managed):
    store, _, _ = managed
    http = server(store.root, 0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{http.server_port}"
    try:
        data = json.load(urllib.request.urlopen(base + "/api/state"))
        assert data["tasks"][0]["spec"]["baseline"]["revision"] == "fixture-v1"
        for asset in ("/", "/app.js", "/sync.js", "/style.css"):
            assert urllib.request.urlopen(base + asset).status == 200
        for headers in ({}, {"X-Manager-Token": data["token"], "Origin": "https://evil.test"}):
            req = urllib.request.Request(
                base + "/api/control",
                data=b'{"task":"toy","action":"pause"}',
                headers={"Content-Type": "application/json"} | headers,
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req)
            assert exc.value.code == 403
        req = urllib.request.Request(
            base + "/api/control",
            data=b'{"task":"toy","action":"pause"}',
            headers={"Content-Type": "application/json", "X-Manager-Token": data["token"]},
        )
        assert json.load(urllib.request.urlopen(req))["ok"]
        assert store.task("toy")["desired"] == "paused"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_state_is_host_bound_and_outside_workspaces(managed, monkeypatch):
    store, cfg, work = managed
    inside = Store(work / "state")
    try:
        with pytest.raises(ValueError, match="outside"):
            inside.register(cfg)
    finally:
        inside.close()
    monkeypatch.setattr("kernel_envolve.manager.store.socket.gethostname", lambda: "different-host")
    with pytest.raises(ValueError, match="another host"):
        Store(store.root)


def test_queued_launch_crash_and_late_claim(managed):
    store, _, _ = managed
    key = store.schedule()[0]
    store.db.execute("UPDATE jobs SET created=created-60 WHERE id=?", (key,))
    reconcile(store)
    assert store.job(key)["status"] == "interrupted"
    assert not store.claim(key, os.getpid(), identity(os.getpid()))
    assert not store.schedule()


def test_budget_terminates_actual_round_without_reset(managed):
    store, _, _ = managed
    replace_command(
        store, [sys.executable, "-c", "import time; time.sleep(10)"], wall_budget_seconds=0.2
    )
    key = store.schedule()[0]
    process = launch(store, key)
    process.communicate(timeout=5)
    assert store.task("toy")["desired"] == "budget"
    assert store.task("toy")["wall_seconds"] >= 0.2
    assert store.job(key)["status"] == "cancelled"
    store.control("toy", "resume")
    assert not store.schedule()
    assert store.task("toy")["desired"] == "budget"


def test_agent_capacity_and_resource_reassignment_preserve_usage(managed, tmp_path):
    store, cfg, work = managed
    other = tmp_path / "other"
    other.mkdir()
    extra = copy.deepcopy(cfg)
    extra["tasks"][0].update(id="other", workspace=str(other), task_dir=str(other))
    store.register(extra)
    scheduled = store.schedule()
    assert len(scheduled) == 2
    store.claim(scheduled[0], os.getpid(), identity(os.getpid()))
    with pytest.raises(ValueError):
        store.control("toy", "configure", {"wall_budget_seconds": 100})
    store.control("toy", "pause")
    with pytest.raises(ValueError):
        store.control("toy", "configure", {"wall_budget_seconds": 100})
    for job in store.rows("SELECT * FROM jobs"):
        store.finish(job["id"], "cancelled")
    store.control("toy", "configure", {"wall_budget_seconds": 100})
    assert store.task("toy")["desired"] == "paused"


def test_registered_separate_group_is_reaped_after_gpu_supervisor_kill(managed):
    store, _, work = managed
    parent = store.schedule()[0]
    store.claim(parent, os.getpid(), identity(os.getpid()))
    script = work / "nested.py"
    script.write_text(
        "from kernel_envolve.execution import spawn\nimport sys,time\n"
        "p=spawn([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True)\n"
        "time.sleep(30)\n"
    )
    env = os.environ | {
        "KERNEL_ENVOLVE_STATE": str(store.root),
        "KERNEL_ENVOLVE_TASK": "toy",
        "KERNEL_ENVOLVE_ROUND": parent,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "manage",
            "gpu-run",
            "--",
            sys.executable,
            str(script),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        child = wait_for(lambda: store.one("SELECT * FROM processes"))
        job = store.job(child["job"])
        process.kill()
        process.communicate(timeout=5)
        reconcile(store)
        assert not alive(child["pid"], child["identity"])
        assert not alive(job["child_pid"], job["child_identity"])
        assert store.job(job["id"])["status"] == "interrupted"
        with exclusive(lock_path("gpu", "cpu")) as handle:
            assert handle is not None
    finally:
        terminate(process)
        store.finish(parent, "cancelled")


def test_gpu_probe_failures_refuse_work_and_quarantine_requires_probe(managed, monkeypatch):
    from kernel_envolve.manager.process import gpu_idle
    from kernel_envolve.manager.service import clear_resource

    def probe(args, **kwargs):
        if "--query-gpu=uuid" in args:
            return subprocess.CompletedProcess(args, 0, "GPU-123\n", "")
        return subprocess.CompletedProcess(args, 0, "999\n", "")

    monkeypatch.setattr("kernel_envolve.manager.process.subprocess.run", probe)
    with pytest.raises(ValueError, match="existing compute"):
        gpu_idle("GPU-123")
    with pytest.raises(ValueError, match="identity"):
        gpu_idle("GPU-456")
    store, _, _ = managed
    store.db.execute(
        "UPDATE resources SET spec=?,problem='orphan'",
        (json.dumps({"id": "local", "device": "GPU-123", "sessions": 2}),),
    )
    assert not store.schedule()
    with pytest.raises(ValueError):
        clear_resource(store, "local")
    assert store.resource("local")["problem"] == "orphan"


def test_multiple_promotions_between_refreshes_and_uncommitted_receipt(managed):
    store, _, work = managed
    (work / "candidate.py").write_text("import time\ndef run(x):\n time.sleep(.001)\n return x+1\n")
    first, result = evaluate(work, "full")
    assert result["decision"]["eligible_for_best"], result["decision"]
    promote(first)
    (work / "candidate.py").write_text("def run(x): return x+1\n")
    second, result = evaluate(work, "full")
    assert result["decision"]["eligible_for_best"], result["decision"]
    promote(second)
    # No indexing occurred between promotions; the reachable chain retains both.
    refresh(store)
    assert len(detail(store, "toy")["history"]) == 2
    pointer = json.loads((second / "promotion.json").read_text())
    (work / ".kernel_envolve/best.json").write_text(json.dumps(pointer["previous_pointer"]))
    # The dashboard keeps the last atomically indexed snapshot until the next scan.
    assert overview(store)["tasks"][0]["best"]["run"] == str(second)
    refresh(store)
    assert overview(store)["tasks"][0]["best"]["run"] == str(first)
    pointer["previous_pointer"]["result_hash"] = "broken"
    (second / "promotion.json").write_text(json.dumps(pointer))
    (work / ".kernel_envolve/best.json").write_text(json.dumps(pointer["pointer"]))
    refresh(store)
    assert "predecessor" in store.task("toy")["evidence_error"]


def test_changed_contract_and_missing_evidence_are_visible(managed):
    store, _, work = managed
    run, result = evaluate(work, "full")
    promote(run)
    refresh(store)
    spec = store.task("toy")["spec"]
    spec["baseline"]["contract"] = "wrong"
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(spec),))
    refresh(store, force=True)
    assert "contract" in store.task("toy")["evidence_error"]
    assert detail(store, "toy")["best"] is None


def test_dashboard_user_text_cannot_execute_or_claim_performance(managed):
    store, _, _ = managed
    key = store.schedule()[0]
    store.claim(key, os.getpid(), identity(os.getpid()))
    store.report("toy", key, {"hypothesis": "<script>alert(1)</script>"})
    assert overview(store)["tasks"][0]["report"]["hypothesis"].startswith("<script>")
    assert overview(store)["tasks"][0]["best"] is None
    store.finish(key, "succeeded")


def test_deleted_receipt_invalidates_previously_verified_best(managed):
    store, _, work = managed
    run, _ = evaluate(work, "full")
    promote(run)
    refresh(store)
    assert detail(store, "toy")["best"]
    receipt = (run / "receipt.json").read_bytes()
    (run / "receipt.json").unlink()
    refresh(store)
    task = detail(store, "toy")
    assert task["best"] is None
    assert "receipt.json" in task["evidence_error"]
    assert task["history"][0]["evidence_error"]
    (run / "receipt.json").write_bytes(receipt)
    refresh(store)
    assert detail(store, "toy")["best"]


@pytest.mark.parametrize("next_action", [None, 123, [], " "])
def test_invalid_checkpoint_finishes_round_without_crashing_supervisor(managed, next_action):
    store, _, _ = managed
    payload = json.dumps({"next_action": next_action})
    replace_command(
        store,
        [
            sys.executable,
            "-c",
            "import os; from pathlib import Path; "
            f"Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text({payload!r})",
        ],
    )
    key = store.schedule()[0]
    process = launch(store, key)
    _, stderr = process.communicate(timeout=5)
    assert b"Traceback" not in stderr
    assert store.job(key)["status"] == "failed"
    assert "next_action" in store.job(key)["error"]


def test_pause_is_durable_even_when_immediately_resumed(managed):
    store, _, _ = managed
    replace_command(store, [sys.executable, "-c", "import time; time.sleep(30)"])
    key = store.schedule()[0]
    process = launch(store, key)
    try:
        wait_for(lambda: store.job(key)["child_pid"])
        store.control("toy", "pause")
        store.control("toy", "resume")
        process.communicate(timeout=5)
        assert store.job(key)["status"] == "cancelled"
        assert store.task("toy")["desired"] == "active"
    finally:
        terminate(process)


def test_pause_cancels_unlaunched_intent_without_waiting_for_timeout(managed):
    store, _, _ = managed
    key = store.schedule()[0]
    store.control("toy", "pause")
    assert store.job(key)["status"] == "cancelled"
    assert not store.claim(key, os.getpid(), identity(os.getpid()))
    store.control("toy", "configure", {"gpu_budget_seconds": 10})


def test_success_without_checkpoint_is_failed_not_infinite_progress(managed):
    store, _, _ = managed
    key = store.schedule()[0]
    process = launch(store, key)
    process.communicate(timeout=5)
    assert store.job(key)["status"] == "failed"
    assert "checkpoint" in store.job(key)["error"]


def test_native_task_cannot_accidentally_override_managed_device(monkeypatch):
    from kernel_envolve.native_config import environment

    monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "job")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-123")
    with pytest.raises(ValueError, match="cannot be overridden"):
        environment({"env": {"CUDA_VISIBLE_DEVICES": "GPU-456"}}, {})


def test_dashboard_snapshot_is_consistent_and_does_not_block_writer(managed, monkeypatch):
    from kernel_envolve.manager import views

    store, _, _ = managed
    reader = Store(store.root, readonly=True)
    original = views.overview

    def change_after_overview(*args, **kwargs):
        data = original(*args, **kwargs)
        store.control("toy", "pause")  # A separate connection commits while the WAL reader is open.
        return data

    monkeypatch.setattr(views, "overview", change_after_overview)
    try:
        data = views.snapshot(reader, "toy")
        assert data["tasks"][0]["desired"] == data["detail"]["desired"] == "active"
        assert store.task("toy")["desired"] == "paused"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.db.execute("DELETE FROM tasks")
    finally:
        reader.close()


def test_evidence_publication_is_atomic_and_file_scan_has_no_writer_lock(managed, monkeypatch):
    from kernel_envolve.manager import evidence

    store, _, work = managed
    run, _ = evaluate(work, "full")
    promote(run)
    original = evidence.inspect_run

    def inspect(*args):
        writer = Store(store.root)
        try:
            writer.control("toy", "pause")  # Verification must not own a database writer lock.
        finally:
            writer.close()
        return original(*args)

    monkeypatch.setattr(evidence, "inspect_run", inspect)
    refresh(store)
    before = detail(store, "toy")
    (run / "receipt.json").unlink()
    store.db.execute(
        "CREATE TRIGGER abort_projection BEFORE UPDATE ON tasks BEGIN SELECT RAISE(ABORT, 'publish failed'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="publish failed"):
        refresh(store)
    after = detail(store, "toy")
    assert after["best"] == before["best"]
    assert after["evidence_as_of"] == before["evidence_as_of"]
    assert not after["runs"][0]["error"]
    store.db.execute("DROP TRIGGER abort_projection")
    refresh(store)
    after = detail(store, "toy")
    assert after["best"] is None and after["evidence_error"]
    assert after["history"][0]["evidence_error"]


def test_evidence_freshness_is_independent_of_http_snapshot_time(managed):
    from kernel_envolve.manager.views import snapshot

    store, _, _ = managed
    refresh(store)
    original = snapshot(store, "toy")
    observed = original["detail"]["evidence_as_of"]
    assert not original["detail"]["evidence_stale"]
    with exclusive(store.root / "evidence.lock"):
        assert refresh(store) is False  # Concurrent scans cannot overwrite newer publications.
    later = snapshot(store, "toy")
    assert later["as_of"] >= original["as_of"]
    assert later["detail"]["evidence_as_of"] == observed
    store.db.execute(
        "UPDATE metadata SET data=? WHERE key='evidence:toy'",
        (json.dumps({"observed_at": observed - 60}),),
    )
    assert snapshot(store, "toy")["detail"]["evidence_stale"]


def test_scan_progress_distinguishes_queue_age_from_scan_failure(managed, monkeypatch):
    from kernel_envolve.manager import evidence
    from kernel_envolve.manager.views import task_view

    store, _, _ = managed
    refresh(store)
    observed = task_view(store, "toy")["evidence_as_of"]
    now = observed + 60
    assert task_view(store, "toy", now)["evidence_status"] == "pending"

    def scan_task(*args):
        scan = json.loads(store.one("SELECT data FROM metadata WHERE key='evidence_scan'")["data"])
        assert scan["running"] and scan["task"] == "toy"
        view = task_view(store, "toy", now)
        assert view["evidence_stale"]  # Observation age alone is not a failure.
        assert view["evidence_status"] == "verifying"
        scan["task"] = "another-task"
        store.db.execute(
            "UPDATE metadata SET data=? WHERE key='evidence_scan'", (json.dumps(scan),)
        )
        assert task_view(store, "toy", now)["evidence_status"] == "queued"
        assert task_view(store, "toy", scan["updated_at"] + 301)["evidence_status"] == "slow"
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(evidence, "refresh_task", scan_task)
    with pytest.raises(OSError, match="filesystem unavailable"):
        refresh(store)
    view = task_view(store, "toy")
    assert view["evidence_status"] == "error"
    assert view["evidence_scan_error"] == "filesystem unavailable"
    scan = json.loads(store.one("SELECT data FROM metadata WHERE key='evidence_scan'")["data"])
    assert not scan["running"] and scan["finished_at"] >= scan["started_at"]
    monkeypatch.undo()
    refresh(store)
    assert task_view(store, "toy")["evidence_status"] == "current"


def test_latest_evaluation_visible_without_promotion_and_invalidated_on_tamper(managed):
    store, _, work = managed
    run, result = evaluate(work, "quick")
    refresh(store)
    task = detail(store, "toy")
    assert task["best"] is None
    latest = task["latest_evaluation"]
    assert latest["run"] == str(run)
    assert latest["correctness"] == "pass"
    assert latest["passed_checks"] == len(result["checks"])
    assert "source" not in latest and "decision" not in latest

    # A later failed candidate must replace the summary, not leave the older PASS visible.
    (work / "candidate.py").write_text("def run(x): return x\n")
    failed, _ = evaluate(work, "quick")
    refresh(store)
    latest = detail(store, "toy")["latest_evaluation"]
    assert latest["run"] == str(failed) and latest["correctness"] == "fail"
    assert not latest["evidence_error"]  # A verified failure is valid evidence.
    (failed / "source" / "candidate.py").write_text("def run(x): return x+1\n")
    refresh(store)
    latest = detail(store, "toy")["latest_evaluation"]
    assert latest["run"] == str(failed)
    assert latest["evidence_error"] and latest["correctness"] is None


def test_old_cached_results_receive_latest_projection_once(managed, monkeypatch):
    from kernel_envolve.manager import evidence

    store, _, work = managed
    run, _ = evaluate(work, "quick")
    refresh(store)
    row = store.one("SELECT data FROM runs WHERE path=?", (str(run),))
    legacy = json.loads(row["data"])
    legacy.pop("recorded_at")
    store.db.execute("UPDATE runs SET data=? WHERE path=?", (json.dumps(legacy), str(run)))
    original = evidence.inspect_run
    calls = []

    def inspect(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(evidence, "inspect_run", inspect)
    refresh(store)
    refresh(store)
    assert calls == [run]
    assert detail(store, "toy")["latest_evaluation"]["recorded_at"] > 0


def test_http_selected_detail_and_compact_summary(managed):
    store, _, work = managed
    run, _ = evaluate(work, "full")
    promote(run)
    refresh(store)
    http = server(store.root, 0)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        data = json.load(
            urllib.request.urlopen(f"http://127.0.0.1:{http.server_port}/api/state?task=toy")
        )
        assert data["detail"]["id"] == "toy"
        assert data["detail"]["best"]["decision"]["cases"]
        assert "cases" not in data["tasks"][0]["best"]["decision"]
        assert "source" not in data["tasks"][0]["best"]
        assert "decision" not in data["detail"]["history"][0]
        assert "cases" not in data["detail"]["runs"][0]["data"]["decision"]
        assert data["detail"]["runs"][0]["data"]["source"]
        assert data["detail"]["evidence_as_of"] == data["tasks"][0]["evidence_as_of"]
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_dashboard_polling_races_backoff_and_visibility():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for browser-independent polling tests")
    subprocess.run(
        [node, "--test", str(Path(__file__).with_name("manager_sync.mjs"))], check=True, timeout=15
    )


@pytest.mark.parametrize("command", ["evaluate", "native-evaluate"])
@pytest.mark.parametrize("descriptor", [None, "invalid", "-1"])
def test_managed_lock_rejection_precedes_source_snapshot(
    tmp_path, monkeypatch, capsys, command, descriptor
):
    from kernel_envolve.cli import main

    monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "managed-fixture")
    if descriptor is None:
        monkeypatch.delenv("KERNEL_ENVOLVE_LOCK_FD", raising=False)
    else:
        monkeypatch.setenv("KERNEL_ENVOLVE_LOCK_FD", descriptor)
    # An absent task must not be read or created before diagnosing the broken adapter.
    task = tmp_path / "not-created"
    assert main([command, "--task", str(task)]) == 2
    error = capsys.readouterr().err
    assert "KERNEL_ENVOLVE_LOCK_FD is missing or invalid" in error
    assert "os.execvpe" in error and "pass_fds" in error
    assert not task.exists()


def test_worker_spawn_rejects_closed_managed_lock(monkeypatch, tmp_path):
    from kernel_envolve.execution import spawn

    monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "managed-fixture")
    monkeypatch.setenv("KERNEL_ENVOLVE_LOCK_FD", "-1")
    marker = tmp_path / "worker-started"
    with pytest.raises(ValueError, match="pass_fds"):
        spawn([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    assert not marker.exists()


@pytest.mark.parametrize("strategy", ["default_subprocess", "exec", "pass_fds"])
def test_real_manager_evaluator_lock_inheritance(managed, strategy):
    store, _, work = managed
    evaluator = [
        sys.executable,
        "-m",
        "kernel_envolve.cli",
        "evaluate",
        "--task",
        str(work),
        "--mode",
        "quick",
    ]
    guard = f"import os,subprocess,sys; argv={evaluator!r}; "
    if strategy == "exec":
        guard += "os.execvpe(argv[0],argv,os.environ)"
    elif strategy == "pass_fds":
        guard += (
            "sys.exit(subprocess.run(argv,pass_fds="
            "(int(os.environ['KERNEL_ENVOLVE_LOCK_FD']),)).returncode)"
        )
    else:
        guard += "sys.exit(subprocess.run(argv).returncode)"
    gpu = [
        sys.executable,
        "-m",
        "kernel_envolve.cli",
        "manage",
        "gpu-run",
        "--",
        sys.executable,
        "-c",
        guard,
    ]
    agent = (
        "import json,os,pathlib,subprocess; "
        f"reply=subprocess.run({gpu!r},capture_output=True,text=True); "
        "pathlib.Path('reply.json').write_text(json.dumps("
        "{'code':reply.returncode,'stdout':reply.stdout,'stderr':reply.stderr})); "
        "pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text("
        "json.dumps({'next_action':'Inspect managed adapter results'}))"
    )
    replace_command(store, [sys.executable, "-c", agent], round_timeout_seconds=30)
    key = store.schedule()[0]
    process = launch(store, key)
    try:
        out, err = process.communicate(timeout=30)
        assert process.returncode == 0, (out, err)
        reply = json.loads((work / "reply.json").read_text())
        jobs = store.rows("SELECT * FROM jobs WHERE kind='gpu' AND parent=?", (key,))
        assert len(jobs) == 1
        if strategy == "default_subprocess":
            assert reply["code"] != 0
            assert "KERNEL_ENVOLVE_LOCK_FD is missing or invalid" in reply["stdout"]
            assert "os.execvpe" in reply["stdout"] and "pass_fds" in reply["stdout"]
            assert not (work / ".kernel_envolve").exists()
            assert jobs[0]["status"] == "failed"
        else:
            assert reply["code"] == 0, reply
            result = json.loads(reply["stdout"])
            assert result["execution"] == "completed" and result["correctness"] == "pass"
            assert jobs[0]["status"] == "succeeded"
    finally:
        terminate(process)
