"""CPU-only multi-card ownership tests use real locks and child processes."""

import copy
import json
import os
import subprocess
import sys
import time
import uuid

import pytest

from kernel_envolve.execution import managed_lock_fds
from kernel_envolve.manager.config import resource_devices, validate
from kernel_envolve.manager.gpu import DurationAdmissionError, require_duration
from kernel_envolve.manager.process import (
    alive,
    exclusive,
    gpu_group_idle,
    gpu_locks,
    identity,
    lock_path,
    terminate,
)
from kernel_envolve.manager.service import clear_resource, reconcile
from kernel_envolve.manager.store import Store
from kernel_envolve.manager.views import overview
from kernel_envolve.manager.worker import supervise

pytest_plugins = ["test_manager"]


@pytest.fixture
def devices():
    return [f"GPU-{uuid.uuid4()}", f"GPU-{uuid.uuid4()}"]


def group_resource(store, devices):
    spec = {"id": "local", "devices": devices, "sessions": 2}
    store.db.execute("UPDATE resources SET spec=? WHERE id='local'", (json.dumps(spec),))


def running(store, devices):
    group_resource(store, devices)
    parent = store.schedule()[0]
    assert store.claim(parent, os.getpid(), identity(os.getpid()))
    return parent


def test_group_configuration_preserves_rank_order(managed, devices):
    _, deployment, work = managed
    changed = copy.deepcopy(deployment)
    changed["tasks"][0].pop("identity")
    changed["resources"] = [{"id": "local", "devices": devices[::-1]}]
    result = validate(changed, work)
    assert resource_devices(result["resources"][0]) == devices[::-1]
    assert resource_devices(deployment["resources"][0]) == ["cpu"]
    changed["resources"] = [{"id": "local", "device": devices[0]}]
    assert resource_devices(validate(changed, work)["resources"][0]) == devices[:1]


@pytest.mark.parametrize(
    "settings",
    [
        {"devices": []},
        {"devices": "GPU-123"},
        {"devices": ["GPU-123", "GPU-123"]},
        {"devices": ["cpu"]},
        {"devices": ["GPU-123", "cuda:0"]},
        {"devices": ["MIG-123"]},
        {"devices": [1]},
        {"devices": ["GPU-123"], "device": "GPU-123"},
        {},
    ],
)
def test_invalid_group_configuration(managed, settings):
    _, deployment, work = managed
    changed = copy.deepcopy(deployment)
    changed["tasks"][0].pop("identity")
    changed["resources"] = [{"id": "local", **settings}]
    with pytest.raises(ValueError):
        validate(changed, work)


def test_group_overlap_rejected_within_deployment_and_state(managed, devices):
    store, deployment, work = managed
    changed = copy.deepcopy(deployment)
    changed["tasks"][0].pop("identity")
    changed["resources"] = [
        {"id": "local", "devices": devices},
        {"id": "overlap", "device": devices[0]},
    ]
    with pytest.raises(ValueError, match="unique"):
        validate(changed, work)
    group_resource(store, devices)
    with pytest.raises(ValueError, match="already registered"):
        store.register({"resources": [{"id": "overlap", "device": devices[0]}], "tasks": []})


def test_partial_group_lock_is_released_before_retry(devices):
    first, second = sorted(devices)
    # Open-file-description flock conflicts even within this process.
    with exclusive(lock_path("gpu", second)) as single:
        assert single is not None
        with gpu_locks(devices[::-1]) as group:
            assert group is None
            with exclusive(lock_path("gpu", first)) as released:
                assert released is not None
    with gpu_locks(devices) as group:
        assert len(group) == 2
        for device in devices:
            with exclusive(lock_path("gpu", device)) as single:
                assert single is None
    with gpu_locks(devices[::-1]) as group:
        assert len(group) == 2


@pytest.mark.parametrize("held_group", [False, True])
def test_real_process_single_and_group_contention(devices, held_group):
    held = devices[::-1] if held_group else devices[1:]
    script = (
        "import sys; from kernel_envolve.manager.process import gpu_locks; "
        f"devices={held!r}\n"
        "with gpu_locks(devices) as handles:\n"
        " assert handles is not None\n"
        " print('locked',flush=True)\n"
        " sys.stdin.readline()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        with gpu_locks(devices) as handles:
            assert handles is None
        if held_group:
            for device in devices:
                with gpu_locks([device]) as handles:
                    assert handles is None
        _, err = process.communicate("release\n", timeout=5)
        assert process.returncode == 0, err
        with gpu_locks(devices[::-1]) as handles:
            assert len(handles) == 2
    finally:
        terminate(process)


def test_group_lock_exception_releases_all_devices(devices):
    with pytest.raises(RuntimeError):
        with gpu_locks(devices) as handles:
            assert len(handles) == 2
            raise RuntimeError("launch failed")
    for device in devices:
        with gpu_locks([device]) as handles:
            assert handles is not None


def test_job_device_snapshot_and_card_seconds_survive_reassignment(managed, devices):
    store, _, _ = managed
    parent = running(store, devices[::-1])
    task = store.task("toy")
    store.db.execute(
        "INSERT INTO resources(id,spec) VALUES('other',?)",
        (json.dumps({"id": "other", "device": "GPU-ffff", "sessions": 1}),),
    )
    # Exercise frozen historical data even if task configuration later moves elsewhere.
    task["spec"]["resource"] = "other"
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(task["spec"]),))
    child = store.enqueue(store.task("toy"), "gpu", [sys.executable, "-c", "pass"], parent)
    assert store.job(child)["resource"] == "local"
    assert store.job_devices(store.job(child)) == devices[::-1]
    assert store.job(child)["gpu_count"] == 2
    assert store.claim(child, os.getpid(), identity(os.getpid()))
    store.heartbeat(child, 3)
    store.heartbeat(child, 3)
    store.heartbeat(parent, 3)
    # Historical accounting must not be recalculated from mutable resource metadata.
    group_resource(store, devices[:1])
    store.heartbeat(child, 4)
    assert store.task("toy")["gpu_seconds"] == 8
    assert store.task("toy")["wall_seconds"] == 3
    assert store.job(child)["seconds"] == 4
    assert overview(store)["resources"][0]["gpu_seconds"] == 8
    store.finish(child, "succeeded")
    store.finish(parent, "cancelled")


def test_legacy_migration_and_readonly_totals_remain_single_card(managed):
    store, _, _ = managed
    parent = store.schedule()[0]
    assert store.claim(parent, os.getpid(), identity(os.getpid()))
    child = store.enqueue(store.task("toy"), "gpu", ["unused"], parent)
    assert store.claim(child, os.getpid(), identity(os.getpid()))
    store.heartbeat(child, 5)
    store.db.execute("ALTER TABLE jobs DROP COLUMN devices")
    store.db.execute("ALTER TABLE jobs DROP COLUMN gpu_count")
    store.db.execute("PRAGMA user_version=1")
    readonly = Store(store.root, readonly=True)
    try:
        assert overview(readonly)["resources"][0]["gpu_seconds"] == 5
        assert readonly.db.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        readonly.close()
    migrated = Store(store.root)
    try:
        assert migrated.db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert migrated.job_devices(migrated.job(child)) == ["cpu"]
        assert migrated.job(child)["gpu_count"] == 1
        migrated.heartbeat(child, 7)
        assert migrated.task("toy")["gpu_seconds"] == 7
        assert overview(migrated)["resources"][0]["gpu_seconds"] == 7
        migrated.finish(child, "succeeded")
        migrated.finish(parent, "cancelled")
    finally:
        migrated.close()


def test_admission_converts_card_budget_and_unreported_usage(managed, devices, monkeypatch):
    store, _, _ = managed
    clock = time.time()
    monkeypatch.setattr("kernel_envolve.manager.store.time.time", lambda: clock)
    parent = running(store, devices)
    spec = store.task("toy")["spec"]
    spec["gpu_budget_seconds"] = 6
    spec["round_timeout_seconds"] = 100
    store.db.execute("UPDATE tasks SET spec=? WHERE id='toy'", (json.dumps(spec),))
    require_duration(store, store.task("toy"), store.job(parent), 3)
    with pytest.raises(DurationAdmissionError, match="cumulative gpu budget"):
        require_duration(store, store.task("toy"), store.job(parent), 3.1)
    child = store.enqueue(store.task("toy"), "gpu", ["unused"], parent)
    assert store.claim(child, os.getpid(), identity(os.getpid()))
    store.heartbeat(child, 1)
    clock += 2
    # Two card-seconds already reported + two still unreported = 2 card-seconds left.
    require_duration(store, store.task("toy"), store.job(parent), 1)
    with pytest.raises(DurationAdmissionError, match="cumulative gpu budget"):
        require_duration(store, store.task("toy"), store.job(parent), 1.1)
    store.finish(child, "cancelled")
    store.finish(parent, "cancelled")


def test_probe_checks_every_uuid_and_reports_all_failures(devices, monkeypatch):
    probed = []

    def probe(device):
        probed.append(device)
        raise ValueError("occupied")

    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probe)
    with pytest.raises(ValueError) as error:
        gpu_group_idle(devices)
    assert probed == devices
    assert all(device in str(error.value) for device in devices)


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_supervisor_inherits_all_locks_and_checks_every_card(
    managed, devices, monkeypatch, fail_cleanup
):
    store, _, work = managed
    parent = running(store, devices[::-1])
    script = (
        "import json,os,sys; from pathlib import Path; "
        "from kernel_envolve.execution import managed_lock_fds,spawn; "
        "fds=managed_lock_fds(); "
        "Path('group-env.json').write_text(json.dumps("
        "{'devices':os.environ['CUDA_VISIBLE_DEVICES'],'locks':len(fds)})); "
        "p=spawn([sys.executable,'-c',"
        "'from kernel_envolve.execution import managed_lock_fds; '"
        "'assert len(managed_lock_fds()) == 2'],start_new_session=True); "
        "assert p.wait(timeout=5)==0"
    )
    child = store.enqueue(store.task("toy"), "gpu", [sys.executable, "-c", script], parent)
    assert store.claim(child, os.getpid(), identity(os.getpid()))
    probed = []

    def probe(device):
        probed.append(device)
        if fail_cleanup:
            raise ValueError("busy after command")

    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probe)
    with gpu_locks(devices) as handles:
        result = supervise(store, child, handles)
    assert result == (2 if fail_cleanup else 0)
    assert json.loads((work / "group-env.json").read_text()) == {
        "devices": ",".join(devices[::-1]),
        "locks": 2,
    }
    assert probed == devices[::-1]
    assert store.task("toy")["gpu_seconds"] == pytest.approx(2 * store.job(child)["seconds"])
    assert bool(store.resource("local")["problem"]) == fail_cleanup
    if fail_cleanup:
        assert all(device in store.job(child)["error"] for device in devices)
    with gpu_locks(devices) as handles:
        assert len(handles) == 2
    store.finish(parent, "cancelled")


def test_cleanup_probe_continues_after_child_reap_error(managed, devices, monkeypatch):
    store, _, _ = managed
    parent = running(store, devices)
    child = store.enqueue(store.task("toy"), "gpu", [sys.executable, "-c", "pass"], parent)
    assert store.claim(child, os.getpid(), identity(os.getpid()))
    probed = []
    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probed.append)

    def fail(*_):
        raise OSError("cleanup failed")

    monkeypatch.setattr("kernel_envolve.manager.worker.reap_registered", fail)
    with gpu_locks(devices) as handles:
        assert supervise(store, child, handles) == 2
    assert probed == devices
    assert "cleanup failed" in store.resource("local")["problem"]
    store.finish(parent, "cancelled")


def test_reconcile_charges_frozen_group_and_quarantines_all_failures(managed, devices, monkeypatch):
    store, _, _ = managed
    parent = running(store, devices)
    child = store.enqueue(store.task("toy"), "gpu", ["unused"], parent)
    assert store.claim(child, 99999999, "absent")
    store.db.execute("UPDATE jobs SET started=? WHERE id=?", (time.time() - 3, child))
    # Deliberately mutate resource metadata to prove recovery uses the job snapshot.
    group_resource(store, ["GPU-ffff"])
    probed = []

    def probe(device):
        probed.append(device)
        raise ValueError("still occupied")

    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probe)
    reconcile(store)
    assert probed == devices
    assert store.job(child)["status"] == "interrupted"
    assert store.task("toy")["gpu_seconds"] >= 6
    assert all(device in store.resource("local")["problem"] for device in devices)
    store.finish(parent, "cancelled")


def test_group_quarantine_clear_checks_every_card_and_lock(managed, devices, monkeypatch):
    store, _, _ = managed
    group_resource(store, devices)
    store.db.execute("UPDATE resources SET problem='orphan' WHERE id='local'")
    probed = []
    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probed.append)
    with gpu_locks(devices[1:]) as handles:
        assert handles is not None
        with pytest.raises(ValueError, match="lock"):
            clear_resource(store, "local")
        assert not probed
    clear_resource(store, "local")
    assert probed == devices
    assert store.resource("local")["problem"] is None


@pytest.mark.parametrize("value", ["[]", "{}", "[true]", "null", "not-json", "[-1]"])
def test_invalid_inherited_lock_group_is_rejected(monkeypatch, value, tmp_path):
    with (tmp_path / "lock").open("w") as handle:
        monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "group-test")
        monkeypatch.setenv("KERNEL_ENVOLVE_LOCK_FD", str(handle.fileno()))
        monkeypatch.setenv("KERNEL_ENVOLVE_LOCK_FDS", value)
        with pytest.raises(ValueError, match="pass_fds"):
            managed_lock_fds()


def test_crashed_group_supervisor_keeps_every_lock_until_children_reaped(
    managed, devices, monkeypatch
):
    from test_manager import wait_for

    store, _, work = managed
    parent = running(store, devices)
    # Only the NVIDIA probe is doubled; CLI, locks, subprocesses and SQLite are real.
    fake = work / "bin"
    fake.mkdir()
    smi = fake / "nvidia-smi"
    smi.write_text(
        f"#!{sys.executable}\nimport sys\nif '--query-gpu=uuid' in sys.argv: print(sys.argv[-1])\n"
    )
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake) + os.pathsep + os.environ["PATH"])
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
        for device in devices:
            with gpu_locks([device]) as handles:
                assert handles is None
        reconcile(store)
        assert not alive(child["pid"], child["identity"])
        assert not alive(job["child_pid"], job["child_identity"])
        assert store.job(job["id"])["status"] == "interrupted"
        assert store.task("toy")["gpu_seconds"] == pytest.approx(
            2 * store.job(job["id"])["seconds"]
        )
        with gpu_locks(devices) as handles:
            assert len(handles) == 2
    finally:
        terminate(process)
        reconcile(store)
        store.finish(parent, "cancelled")


@pytest.mark.parametrize("bad_index", [0, 1])
def test_any_busy_card_prevents_entire_command_launch(managed, devices, monkeypatch, bad_index):
    from kernel_envolve.manager.gpu import run

    store, _, work = managed
    parent = running(store, devices)
    probed = []

    def probe(device):
        probed.append(device)
        if device == devices[bad_index]:
            # End this fixture's wait after a failed admission, without launching anything.
            store.control("toy", "pause")
            raise ValueError("device busy")

    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probe)
    marker = work / "must-not-launch"
    assert (
        run(
            store.root,
            "toy",
            parent,
            [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
            expected_seconds=1,
        )
        == 2
    )
    assert probed == devices
    assert not marker.exists()
    job = store.one("SELECT * FROM jobs WHERE kind='gpu'")
    assert job["status"] == "cancelled" and job["child_pid"] is None
    assert store.task("toy")["gpu_seconds"] == 0
    with gpu_locks(devices) as handles:
        assert len(handles) == 2
    store.finish(parent, "cancelled")


def test_closed_second_lock_is_rejected_even_when_first_is_valid(monkeypatch, tmp_path):
    with (tmp_path / "first").open("w") as first:
        with (tmp_path / "second").open("w") as second:
            descriptors = [first.fileno(), second.fileno()]
        monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "group-test")
        monkeypatch.setenv("KERNEL_ENVOLVE_LOCK_FD", str(first.fileno()))
        monkeypatch.setenv("KERNEL_ENVOLVE_LOCK_FDS", json.dumps(descriptors))
        with pytest.raises(ValueError, match="pass_fds"):
            managed_lock_fds()


@pytest.mark.parametrize("missing_group", [True, False])
def test_multi_card_adapter_cannot_fall_back_to_first_lock(devices, tmp_path, missing_group):
    marker = tmp_path / "must-not-launch"
    child = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    script = (
        "import sys; from kernel_envolve.execution import spawn; "
        f"spawn([sys.executable,'-c',{child!r}])"
    )
    with gpu_locks(devices) as handles:
        descriptors = [handle.fileno() for handle in handles]
        env = os.environ | {
            "KERNEL_ENVOLVE_GPU_JOB": "group-test",
            "KERNEL_ENVOLVE_LOCK_FD": str(descriptors[0]),
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
        }
        if missing_group:
            env.pop("KERNEL_ENVOLVE_LOCK_FDS", None)
        else:
            env["KERNEL_ENVOLVE_LOCK_FDS"] = json.dumps(descriptors[:1])
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            pass_fds=tuple(descriptors),
            capture_output=True,
            text=True,
            timeout=5,
        )
    assert result.returncode != 0 and "Device group count" in result.stderr
    assert not marker.exists()


def test_cleanup_errors_preserve_locks_owned_by_surviving_descendant(managed, devices, monkeypatch):
    from kernel_envolve.manager.process import stop_recorded

    store, _, _ = managed
    parent = running(store, devices)
    script = (
        "import sys; from kernel_envolve.execution import spawn; "
        "spawn([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)"
    )
    child = store.enqueue(store.task("toy"), "gpu", [sys.executable, "-c", script], parent)
    assert store.claim(child, os.getpid(), identity(os.getpid()))
    probed = []
    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probed.append)

    def fail_terminate(*_):
        raise OSError("injected terminate failure")

    def fail_reap(*_):
        raise OSError("injected descendant cleanup failure")

    monkeypatch.setattr("kernel_envolve.manager.worker.terminate", fail_terminate)
    monkeypatch.setattr("kernel_envolve.manager.worker.reap_registered", fail_reap)
    descendant = None
    try:
        with gpu_locks(devices) as handles:
            assert supervise(store, child, handles) == 2
        descendant = store.one("SELECT * FROM processes WHERE job=?", (child,))
        assert alive(descendant["pid"], descendant["identity"])
        assert probed == devices
        job = store.job(child)
        assert job["status"] == "failed"
        assert "terminate failure" in job["error"] and "descendant cleanup failure" in job["error"]
        assert store.resource("local")["problem"] == job["error"]
        # A separate resource/state cannot acquire any card after supervisor exits.
        for device in devices:
            with gpu_locks([device]) as handles:
                assert handles is None
    finally:
        descendant = descendant or store.one("SELECT * FROM processes WHERE job=?", (child,))
        if descendant:
            stop_recorded(descendant["pid"], descendant["identity"])
        store.finish(parent, "cancelled")
    with gpu_locks(devices) as handles:
        assert len(handles) == 2


def test_reconcile_keeps_probing_after_first_cleanup_error(managed, devices, monkeypatch):
    store, _, _ = managed
    parent = running(store, devices)
    child = store.enqueue(store.task("toy"), "gpu", ["unused"], parent)
    assert store.claim(child, 99999999, "absent")
    probed = []
    reaped = []
    monkeypatch.setattr("kernel_envolve.manager.process.gpu_idle", probed.append)
    monkeypatch.setattr(
        "kernel_envolve.manager.service.reap_registered", lambda *args: reaped.append(args[1])
    )

    def fail_stop(*_):
        raise OSError("injected stop failure")

    monkeypatch.setattr("kernel_envolve.manager.service.stop_recorded", fail_stop)
    reconcile(store)
    assert reaped == [child] and probed == devices
    assert store.job(child)["status"] == "interrupted"
    assert "stop failure" in store.job(child)["error"]
    assert "stop failure" in store.resource("local")["problem"]
    store.finish(parent, "cancelled")


def test_registered_child_cleanup_continues_after_one_process_error(managed, monkeypatch):
    from kernel_envolve.manager import process as processes

    store, _, _ = managed
    job = store.schedule()[0]
    children = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
        )
        for _ in range(2)
    ]
    births = [identity(child.pid) for child in children]
    for child, birth in zip(children, births, strict=True):
        store.db.execute("INSERT INTO processes VALUES(?,?,?)", (job, child.pid, birth))
    first = min(children, key=lambda child: child.pid)
    stop = processes.stop_recorded
    attempted = []

    def fail_one(pid, birth):
        attempted.append(pid)
        if pid == first.pid:
            raise OSError("injected process error")
        stop(pid, birth)

    monkeypatch.setattr(processes, "stop_recorded", fail_one)
    try:
        with pytest.raises(OSError, match="injected process error"):
            processes.reap_registered(store, job)
        assert set(attempted) == {child.pid for child in children}
        assert alive(first.pid, births[children.index(first)])
        for child, birth in zip(children, births, strict=True):
            if child is not first:
                assert not alive(child.pid, birth)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        store.finish(job, "cancelled")


@pytest.mark.parametrize("complete_fds", [False, True])
def test_native_device_subset_uses_frozen_group_for_lock_validation(managed, devices, complete_fds):
    store, _, _ = managed
    parent = running(store, devices)
    child = store.enqueue(store.task("toy"), "gpu", ["unused"], parent)
    with gpu_locks(devices) as handles:
        descriptors = [handle.fileno() for handle in handles]
        # Native checks legitimately narrow visible devices but retain the whole reservation.
        env = os.environ | {
            "KERNEL_ENVOLVE_STATE": str(store.root),
            "KERNEL_ENVOLVE_GPU_JOB": child,
            "KERNEL_ENVOLVE_LOCK_FD": str(descriptors[0]),
            "CUDA_VISIBLE_DEVICES": devices[0],
        }
        if complete_fds:
            env["KERNEL_ENVOLVE_LOCK_FDS"] = json.dumps(descriptors)
        else:
            env.pop("KERNEL_ENVOLVE_LOCK_FDS", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from kernel_envolve.execution import managed_lock_fds; assert len(managed_lock_fds()) == 2",
            ],
            env=env,
            pass_fds=tuple(descriptors),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if complete_fds:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0 and "Device group count" in result.stderr
    store.finish(child, "cancelled")
    store.finish(parent, "cancelled")
