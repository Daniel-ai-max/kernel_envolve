"""Actual model-host subprocesses resume through the existing coordinator, without GPUs."""

import copy
import json
import os
import subprocess
import sys
import time

import pytest
from test_manager import launch, replace_command, wait_for

from kernel_envolve.manager.config import validate
from kernel_envolve.manager.gpu import DurationAdmissionError, require_duration
from kernel_envolve.manager.process import alive, identity, terminate
from kernel_envolve.manager.recovery import context
from kernel_envolve.manager.store import Store

pytest_plugins = ["test_manager"]


def configure(managed, script, **settings):
    store, _, work = managed
    root = store.root.parent / "bridge"
    agent = work / "host.py"
    agent.write_text(script)
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
            "1",
        ],
        host_driven=True,
        host_command=[sys.executable, str(agent)],
        wall_budget_seconds=60,
        round_timeout_seconds=40,
        **settings,
    )
    return root


HOST = """import os,sys,json
from pathlib import Path
from kernel_envolve.manager import bridge
from kernel_envolve.manager.store import Store
prompt=sys.stdin.read()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
store=Store(os.environ['KERNEL_ENVOLVE_STATE'])
task=os.environ['KERNEL_ENVOLVE_TASK']; round_id=os.environ['KERNEL_ENVOLVE_ROUND']
root=Path(os.environ['KERNEL_ENVOLVE_BRIDGE_ROOT']); owner=os.environ['KERNEL_ENVOLVE_BRIDGE_OWNER']
marker=Path('iteration.txt')
n=int(marker.read_text())+1 if marker.exists() else 1
marker.write_text(str(n))
if n==3:
 print('Unrecoverable fixture provider failure; saved all results',flush=True)
 raise SystemExit(78)
if n==2:
 previous=store.rows("SELECT id FROM jobs WHERE task=? AND kind='agent' AND status='succeeded'",(task,))[0]['id']
 assert previous in prompt
 assert 'hypothesis rejected' in (root/task/previous/'agent.stdout.log').read_text()
 assert (root/task/previous/'results'/'probe.json').exists()
 # Saved candidate remains available without a new task or copied notes.
 assert Path('candidate.py').read_text().endswith('# saved candidate\\n')
result=bridge.request(store,root,task,round_id,owner,'probe',
 {'kind':'gpu' if n==2 else 'cpu','argv':[sys.executable,'-c',"print('new hypothesis result %s')"%n],
  'expected_seconds':10},20)
assert result['status']=='succeeded',result
if n==1:
 with Path('candidate.py').open('a') as f: f.write('# saved candidate\\n')
 print('hypothesis rejected; try distinct combination',flush=True)
else:
 print('next distinct hypothesis tested',flush=True)
# No finish, consume, analysis, report or notes: real process exit is the boundary.
"""


def test_coordinator_relaunches_real_host_and_preserves_work_and_budget(managed):
    store, _, work = managed
    root = configure(managed, HOST)
    coordinator = subprocess.Popen(
        [sys.executable, "-m", "kernel_envolve.cli", "manage", "--state", str(store.root), "run"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for(lambda: store.task("toy")["desired"] == "blocked", timeout=45)
    finally:
        terminate(coordinator)
    jobs = store.rows("SELECT * FROM jobs WHERE kind='agent' ORDER BY created")
    assert [j["status"] for j in jobs] == ["succeeded", "succeeded", "failed"]
    assert jobs[-1]["exit_code"] == 78
    assert "unrecoverable" in store.task("toy")["reason"]
    assert (work / "iteration.txt").read_text() == "3"
    assert not (work / "notes.md").exists()
    assert store.task("toy")["wall_seconds"] == pytest.approx(sum(j["seconds"] for j in jobs))
    assert store.task("toy")["gpu_seconds"] > 0
    assert len(store.rows("SELECT id FROM jobs WHERE kind='gpu'")) == 1
    assert store.task("toy")["report"] == {}
    assert not store.schedule()
    for job in jobs:
        assert not list((root / "toy" / job["id"] / "consumed").glob("*.json"))
        assert (root / "toy" / job["id"] / "agent.exit.json").exists()
    assert all(not alive(p["pid"], p["identity"]) for p in store.rows("SELECT * FROM processes"))
    reopened = Store(store.root)
    try:
        assert not reopened.schedule()
        assert reopened.task("toy")["wall_seconds"] == store.task("toy")["wall_seconds"]
        assert context(reopened, reopened.task("toy"))["recent_sessions"][0]["id"] == jobs[-1]["id"]
    finally:
        reopened.close()


def test_real_host_analysis_does_not_expire_bridge_idle_lease(managed):
    store, _, _ = managed
    configure(managed, 'import time;time.sleep(1.5);print("analysis complete")')
    key = store.schedule()[0]
    process = launch(store, key)
    process.communicate(timeout=10)
    assert store.job(key)["status"] == "succeeded"  # Not yielded at the one-second idle lease.
    assert store.task("toy")["failures"] == 1  # Model output alone is not experiment progress.


def test_empty_hosts_are_bounded_and_do_not_count_as_progress(managed):
    store, _, _ = managed
    configure(managed, 'print("candidate delivered")')
    for _ in range(3):
        store.db.execute("UPDATE tasks SET next_at=0")
        key = store.schedule()[0]
        process = launch(store, key)
        process.communicate(timeout=10)
    assert store.task("toy")["desired"] == "blocked"
    assert not store.schedule()
    assert not context(store, store.task("toy"))["observable_progress"]


@pytest.mark.parametrize("action", ["pause", "deadline", "budget"])
def test_stop_cleans_model_descendants_without_restart(managed, action):
    store, _, _ = managed
    root = configure(
        managed,
        'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c","import time;time.sleep(100)"])\nprint("started",flush=True)\ntime.sleep(100)',
    )
    key = store.schedule()[0]
    process = launch(store, key)
    path = root / "toy" / key / "agent.stdout.log"
    try:
        wait_for(lambda: path.exists() and "started" in path.read_text())
        if action == "pause":
            store.control("toy", "pause")
        else:
            task = store.task("toy")
            task["spec"].update(
                {"deadline_at": time.time() - 1}
                if action == "deadline"
                else {"wall_budget_seconds": 0.001}
            )
            store.db.execute("UPDATE tasks SET spec=?", (json.dumps(task["spec"]),))
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            terminate(process)
    assert store.task("toy")["desired"] == ("paused" if action == "pause" else "budget")
    assert not store.schedule()
    assert all(not alive(p["pid"], p["identity"]) for p in store.rows("SELECT * FROM processes"))


def test_absolute_deadline_counts_cooldown_and_rejects_long_gpu_command(managed):
    store, _, _ = managed
    configure(managed, "pass", deadline_at=time.time() + 5)
    key = store.schedule()[0]
    store.claim(key, os.getpid(), identity(os.getpid()))
    with pytest.raises(DurationAdmissionError, match="task deadline"):
        require_duration(store, store.task("toy"), store.job(key), 10)
    store.finish(key, "succeeded")
    spec = store.task("toy")["spec"] | {"deadline_at": time.time() - 1}
    store.db.execute("UPDATE tasks SET spec=?,next_at=?", (json.dumps(spec), time.time() + 300))
    assert not store.schedule()
    assert store.task("toy")["desired"] == "budget"


@pytest.mark.parametrize(
    "changes",
    [
        {"host_command": ["agent"], "host_driven": False, "wall_budget_seconds": 60},
        {"host_command": [], "host_driven": True, "wall_budget_seconds": 60},
        {"host_command": ["agent"], "host_driven": True},
        {"host_command": ["agent", "\0"], "host_driven": True, "wall_budget_seconds": 60},
        {"deadline_at": float("nan")},
    ],
)
def test_invalid_automatic_host_configuration_rejected(managed, changes):
    store, deployment, _ = managed
    cfg = copy.deepcopy(deployment)
    cfg["tasks"][0].pop("identity")
    cfg["tasks"][0].update(changes)
    with pytest.raises(ValueError):
        validate(cfg, store.root.parent)


def test_valid_host_config_is_idempotent_and_does_not_reset_budget(managed):
    store, _, _ = managed
    configure(managed, "pass")
    cfg = {
        "version": 1,
        "resources": [store.resource("local")["spec"]],
        "tasks": [store.task("toy")["spec"]],
    }
    cfg["tasks"][0].pop("identity")
    validated = validate(cfg, store.root.parent)
    store.db.execute("UPDATE tasks SET wall_seconds=11,gpu_seconds=7")
    store.register(validated)
    assert store.task("toy")["wall_seconds"] == 11
    assert store.task("toy")["gpu_seconds"] == 7


def test_same_command_with_changed_implementation_is_new_execution(managed):
    from kernel_envolve.manager.recovery import begin

    store, _, work = managed
    configure(managed, "pass")
    for i in range(4):
        store.db.execute("UPDATE tasks SET next_at=0")
        key = store.schedule()[0]
        store.claim(key, os.getpid(), identity(os.getpid()))
        begin(store, store.task("toy"), store.job(key))
        (work / "candidate.py").write_text(f"def run(x): return x+{i}\n")
        store.event(
            "toy",
            "bridge_execution",
            {
                "round": key,
                "request": "same-name",
                "command": ["probe"],
                "status": "succeeded",
                "returncode": 0,
                "cleanup_confirmed": True,
            },
        )
        store.finish(key, "succeeded")
        assert context(store, store.task("toy"))["new_cpu_requests"] == ["same-name"]
        assert store.task("toy")["failures"] == 0


def test_coordinator_restart_observes_live_host_without_duplicate_launch(managed):
    store, _, work = managed
    root = configure(
        managed,
        """import time,sys
from pathlib import Path
p=Path('launches.txt'); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n))
print('model started',flush=True)
time.sleep(2)
raise SystemExit(78)
""",
    )

    def once():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kernel_envolve.cli",
                "manage",
                "--state",
                str(store.root),
                "run",
                "--once",
            ],
            check=True,
            timeout=10,
            capture_output=True,
        )

    once()
    job = store.rows("SELECT * FROM jobs WHERE kind='agent'")[0]
    path = root / "toy" / job["id"] / "agent.stdout.log"
    wait_for(lambda: path.exists() and "model started" in path.read_text())
    once()
    assert len(store.rows("SELECT * FROM jobs WHERE kind='agent'")) == 1
    wait_for(lambda: store.task("toy")["desired"] == "blocked")
    once()
    assert (work / "launches.txt").read_text() == "1"
    assert store.task("toy")["wall_seconds"] > 0


def test_enable_existing_bridge_keeps_workspace_and_usage(managed):
    store, _, work = managed
    root = store.root.parent / "bridge"
    replace_command(
        store,
        [sys.executable, "-m", "kernel_envolve.cli", "manage", "bridge", "--root", str(root)],
        host_driven=True,
        wall_budget_seconds=100,
    )
    store.db.execute("UPDATE tasks SET wall_seconds=23,gpu_seconds=11")
    with pytest.raises(ValueError, match="Pause"):
        store.control("toy", "configure", {"host_command": [sys.executable, "-c", "pass"]})
    store.control("toy", "pause")
    store.control(
        "toy",
        "configure",
        {"host_command": [sys.executable, "-c", "pass"], "deadline_at": time.time() + 60},
    )
    store.control("toy", "resume")
    assert store.task("toy")["wall_seconds"] == 23
    assert store.task("toy")["gpu_seconds"] == 11
    assert store.task("toy")["spec"]["workspace"] == str(work)
    assert len(store.schedule()) == 1


def test_automatic_host_rejects_competing_manual_host(managed):
    from kernel_envolve.manager.bridge import open_host

    store, _, _ = managed
    configure(managed, "pass")
    with pytest.raises(ValueError, match="second host"):
        open_host(store, "toy", "manual", "claim")
    jobs = store.schedule()
    assert len(jobs) == 1
    assert not store.schedule()
    permit = json.loads(
        store.one("SELECT data FROM metadata WHERE key=?", ("host-round:" + jobs[0],))["data"]
    )
    assert permit["owner"] == "manager"


def test_same_command_is_not_proof_of_empty_host_session(managed):
    # A script may consume different input files without changing its argv or implementation.
    from kernel_envolve.manager.recovery import begin

    store, _, _ = managed
    configure(managed, "pass")
    for _ in range(4):
        store.db.execute("UPDATE tasks SET next_at=0")
        key = store.schedule()[0]
        store.claim(key, os.getpid(), identity(os.getpid()))
        begin(store, store.task("toy"), store.job(key))
        store.event(
            "toy",
            "bridge_execution",
            {
                "round": key,
                "request": "probe",
                "command": ["probe"],
                "status": "succeeded",
                "returncode": 0,
                "cleanup_confirmed": True,
            },
        )
        store.finish(key, "succeeded")
        assert store.task("toy")["desired"] == "active"
        assert store.task("toy")["failures"] == 0
    assert not context(store, store.task("toy"))["observable_progress"]
