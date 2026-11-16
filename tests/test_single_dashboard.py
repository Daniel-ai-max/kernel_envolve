"""One kernel uses the real manager and its unchanged HTTP/record pipeline."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from test_manager import wait_for

from kernel_envolve.cli import evaluate
from kernel_envolve.manager.evidence import refresh
from kernel_envolve.manager.process import identity, terminate
from kernel_envolve.manager.single import parser, prepare
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import task_view

pytest_plugins = ["test_manager", "test_native"]


def args(task, *extra):
    return parser().parse_args(["--task", str(task), *extra])


def test_observation_never_launches_work_and_imports_existing_evidence(task, tmp_path):
    run, result = evaluate(task)
    with_store = Store(tmp_path.parent / (tmp_path.name + "-state"))
    try:
        key = prepare(with_store, args(task))
        assert not with_store.schedule()
        assert prepare(with_store, args(task)) == key
        assert len(with_store.rows("SELECT * FROM tasks")) == 1
        assert refresh(with_store)
        view = task_view(with_store, key)
        assert view["phase"] == "observing"
        assert view["latest_evaluation"]["run"] == str(run)
        assert view["recent_execution"]["kind"] == "evaluation"
        assert view["best"] is None
        assert view["allocation_usage"] is None  # No invented zero card-hours.
        assert view["report"] == {}
        assert not (task / "notes.md").exists()
    finally:
        with_store.close()


def test_observer_upgrades_to_one_persistent_agent_without_reset(task, tmp_path):
    store = Store(tmp_path.parent / (tmp_path.name + "-state"))
    try:
        key = prepare(store, args(task))
        store.db.execute("UPDATE tasks SET wall_seconds=3,gpu_seconds=2")
        command = [sys.executable, "-c", "pass"]
        key2 = prepare(store, args(task, "--seconds", "60", "--", *command))
        assert key2 == key
        spec = store.task(key)["spec"]
        assert spec["host_command"] == command and spec["host_driven"]
        assert not spec["observe_only"] and spec["wall_budget_seconds"] == 63
        assert store.task(key)["gpu_seconds"] == 2
        deadline = spec["deadline_at"]
        assert prepare(store, args(task)) == key
        assert store.task(key)["spec"]["deadline_at"] == deadline
        assert len(store.schedule()) == 1
        assert not store.schedule()
        store.control(key, "pause")
        prepare(store, args(task))
        assert store.task(key)["desired"] == "paused"
    finally:
        store.close()


def test_existing_multitask_state_retains_pr_usage_and_controls(managed):
    store, _, work = managed
    store.db.execute(
        "INSERT INTO metadata VALUES('publication:toy',?)", (json.dumps({"fixture": "retained"}),)
    )
    store.db.execute("UPDATE tasks SET wall_seconds=9,gpu_seconds=5")
    store.control("toy", "pause")
    assert prepare(store, args(work)) == "toy"
    assert store.task("toy")["desired"] == "paused"
    assert store.task("toy")["wall_seconds"] == 9
    assert store.one("SELECT data FROM metadata WHERE key='publication:toy'")
    assert len(store.rows("SELECT * FROM tasks")) == 1


def test_prose_and_heartbeats_cannot_mask_absent_experiments(managed):
    store, _, _ = managed
    key = store.schedule()[0]
    store.claim(key, os.getpid(), identity(os.getpid()))
    store.db.execute("UPDATE tasks SET created=created-600")
    store.db.execute("UPDATE jobs SET started=started-600,heartbeat=?", (time.time(),))
    store.report("toy", key, {"outcome": "still working, purported 20% signal"})
    view = task_view(store, "toy")
    assert view["phase"] == "awaiting_progress" and view["experiment_stale"]
    assert view["recent_execution"] is None and view["best"] is None


def test_cpu_activity_is_automatic_and_cannot_survive_terminal_parent(managed):
    store, _, _ = managed
    key = store.schedule()[0]
    store.claim(key, os.getpid(), identity(os.getpid()))
    store.db.execute(
        "INSERT INTO metadata VALUES(?,?)",
        (
            "cpu_activity:toy",
            json.dumps(
                {
                    "task": "toy",
                    "round": key,
                    "kind": "cpu_request",
                    "request": "probe",
                    "command": ["python", "diagnose.py"],
                    "status": "running",
                    "recorded_at": time.time(),
                }
            ),
        ),
    )
    view = task_view(store, "toy")
    assert view["phase"] == "cpu_running" and view["current_work"] == "python diagnose.py"
    assert view["current_work_argv"] == ["python", "diagnose.py"]
    assert view["recent_execution"]["kind"] == "cpu_request"
    store.finish(key, "failed", "host lost")
    view = task_view(store, "toy")
    assert view["phase"] != "cpu_running"
    assert view["recent_cpu_request"]["status"] == "interrupted"
    assert view["current_work_argv"] is None


def test_single_command_http_dashboard_refreshes_without_reports(task, tmp_path):
    state = tmp_path.parent / (tmp_path.name + "-state")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "dashboard",
            "--task",
            str(task),
            "--state",
            str(state),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        line = process.stdout.readline()
        assert line, process.stderr.read()
        info = json.loads(line)

        def snapshot():
            return json.load(urllib.request.urlopen(info["url"] + "/api/state", timeout=5))

        view = snapshot()
        assert len(view["tasks"]) == 1
        assert view["tasks"][0]["phase"] == "observing"
        html = urllib.request.urlopen(info["url"], timeout=5).read().decode()
        assert "<th>最新实现</th>" in html and "<th>消耗卡时</th>" in html
        run, _ = evaluate(task)
        wait_for(lambda: snapshot()["tasks"][0]["latest_evaluation"], timeout=15)
        view = snapshot()["tasks"][0]
        assert view["latest_evaluation"]["run"] == str(run)
        assert not any(view["report"].values()) and view["best"] is None
        assert len(view["jobs"]) == 0
    finally:
        terminate(process)
        process.communicate(timeout=10)


def test_gpu_monitoring_does_not_probe_or_allocate_hardware(task, tmp_path, monkeypatch):
    cfg = json.loads((task / "task.json").read_text())
    cfg["device"] = "cuda:0"
    (task / "task.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(
        "kernel_envolve.manager.single.device_for",
        lambda *a: pytest.fail("observer must not probe GPU"),
    )
    store = Store(tmp_path.parent / (tmp_path.name + "-state"))
    try:
        key = prepare(store, args(task))
        assert store.resource(store.task(key)["spec"]["resource"])["spec"]["device"] == "unassigned"
        assert not store.schedule()
    finally:
        store.close()


def test_observer_service_does_not_dispatch_or_reconcile_existing_tasks(managed, monkeypatch):
    from kernel_envolve.manager.service import run

    store, _, _ = managed
    monkeypatch.setattr(
        "kernel_envolve.manager.service.reconcile",
        lambda *a: pytest.fail("observer must not recover processes"),
    )
    monkeypatch.setattr(store, "schedule", lambda: pytest.fail("observer must not schedule tasks"))
    run(store, once=True, observe=True)
    assert store.one("SELECT data FROM metadata WHERE key='observer'")
    assert not store.one("SELECT data FROM metadata WHERE key='coordinator'")
    assert not store.rows("SELECT * FROM jobs")


def test_quarantined_gpu_still_allows_real_cpu_diagnosis(managed):
    from test_manager_continuation import configure

    from kernel_envolve.manager.gpu import ResourceAdmissionError, require_resource
    from kernel_envolve.manager.recovery import context

    store, _, _ = managed
    configure(managed, "pass")
    store.db.execute("UPDATE resources SET problem='cleanup uncertain'")
    key = store.schedule()[0]
    assert store.claim(key, os.getpid(), identity(os.getpid()))
    assert task_view(store, "toy")["phase"] == "recovering"
    assert context(store, store.task("toy"))["resource_problem"] == "cleanup uncertain"
    with pytest.raises(ResourceAdmissionError):
        require_resource(store, "local")


def test_native_single_task_uses_candidate_workspace(native_task, tmp_path):
    task, candidate = native_task
    store = Store(tmp_path.parent / (tmp_path.name + "-state"))
    try:
        key = prepare(store, args(task))
        spec = store.task(key)["spec"]
        assert spec["kind"] == "native"
        assert spec["workspace"] == str(candidate.parent.resolve())
        assert spec["baseline"]["entrypoint"] == "adapter.py"
        assert not store.schedule()
    finally:
        store.close()


def test_occupied_port_does_not_register_or_launch_task(task, tmp_path):
    state = tmp_path.parent / (tmp_path.name + "-state")
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kernel_envolve.cli",
                "dashboard",
                "--task",
                str(task),
                "--state",
                str(state),
                "--port",
                str(occupied.getsockname()[1]),
                "--",
                sys.executable,
                "-c",
                "raise AssertionError('must not start')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert result.returncode != 0
    store = Store(state)
    try:
        assert not store.rows("SELECT * FROM tasks")
        assert not store.rows("SELECT * FROM jobs")
    finally:
        store.close()


def test_single_launcher_really_resumes_host_and_updates_http(task, tmp_path):
    from test_manager_continuation import HOST

    state = tmp_path.parent / (tmp_path.name + "-state")
    host = task / "host.py"
    host.write_text(HOST)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "dashboard",
            "--task",
            str(task),
            "--state",
            str(state),
            "--port",
            "0",
            "--seconds",
            "90",
            "--",
            sys.executable,
            str(host),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    store = None
    try:
        line = process.stdout.readline()
        assert line, process.stderr.read()
        info = json.loads(line)
        store = Store(state)
        key = info["task"]
        wait_for(lambda: store.task(key)["desired"] == "blocked", timeout=60)
        view = json.load(urllib.request.urlopen(info["url"] + "/api/state", timeout=5))["tasks"][0]
        assert view["phase"] == "blocked" and view["best"] is None
        jobs = store.rows("SELECT * FROM jobs WHERE kind='agent' ORDER BY created")
        assert [j["status"] for j in jobs] == ["succeeded", "succeeded", "failed"]
        assert jobs[-1]["exit_code"] == 78
        assert store.task(key)["wall_seconds"] == pytest.approx(sum(j["seconds"] for j in jobs))
        assert (task / "iteration.txt").read_text() == "3"
        assert not (task / "notes.md").exists()
        assert not any(view["report"].values())
        activity = json.loads(
            store.one("SELECT data FROM metadata WHERE key=?", ("cpu_activity:" + key,))["data"]
        )
        assert activity["request"] == "probe" and activity["status"] == "succeeded"
    finally:
        if store:
            # Only this fixture's supervisor; preserve normal launcher restart semantics.
            store.control(info["task"], "pause")
            store.close()
        terminate(process)
        process.communicate(timeout=10)


@pytest.mark.parametrize(
    "visible,expected",
    [(None, None), ("1", "GPU-bbbb"), ("GPU-aaaa", "GPU-aaaa"), ("", None), ("-1", None)],
)
def test_auto_device_respects_visible_inventory(monkeypatch, visible, expected):
    from kernel_envolve.manager.single import device_for

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "0, GPU-aaaa\n1, GPU-bbbb\n")
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if expected:
        assert device_for({"device": "cuda:0"}, None) == expected
    else:
        with pytest.raises(ValueError):
            device_for({"device": "cuda:0"}, None)


def test_archive_cli_restores_without_task_directory_or_writable_state(managed, tmp_path):
    from kernel_envolve.manager.archive import export_archive

    store, _, work = managed
    destination = tmp_path / "archive"
    export_archive(store, destination)
    import shutil

    shutil.rmtree(work)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.cli",
            "dashboard",
            "--archive",
            str(destination),
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = process.stdout.readline().strip()
        assert line.startswith("Kernel Envolve: http://127.0.0.1:")
        url = line.split(": ", 1)[1]
        with urllib.request.urlopen(url + "/api/state", timeout=5) as response:
            data = json.load(response)
        assert data["archived"] and len(data["tasks"]) == 1
        assert data["tasks"][0]["best"] is None
    finally:
        process.terminate()
        process.communicate(timeout=10)
