"""Duration declarations reject work before child side effects, including after queuing."""

import json
import os
import sys
import time

import pytest
from test_manager import launch, replace_command, wait_for

from kernel_envolve.manager.gpu import run
from kernel_envolve.manager.process import exclusive, identity, lock_path

pytest_plugins = ["test_manager"]


def agent(store, work, expected, returncode=0, **settings):
    marker = work / "child-ran"
    command = [
        sys.executable,
        "-m",
        "kernel_envolve.cli",
        "manage",
        "gpu-run",
        "--expected-seconds",
        str(expected),
        "--",
        sys.executable,
        "-c",
        f"from pathlib import Path;Path({str(marker)!r}).write_text('ran')",
    ]
    script = work / "agent.py"
    script.write_text(
        "import subprocess,os,json,pathlib\n"
        f"p=subprocess.run({command!r},capture_output=True,text=True)\n"
        f"pathlib.Path({str(work / 'response.json')!r}).write_text(json.dumps(dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)))\n"
        f"assert p.returncode == {returncode}, (p.returncode,p.stdout,p.stderr)\n"
        "pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'next_action':'inspect completed evidence before scheduling another command'}))\n"
    )
    replace_command(store, [sys.executable, str(script)], **settings)
    key = store.schedule()[0]
    return key, launch(store, key), marker


def finish(process):
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, (stdout, stderr)


@pytest.mark.parametrize(
    "expected,settings,reason",
    [
        (20, {}, "parent round"),
        (3, {"wall_budget_seconds": 2}, "cumulative wall"),
        (1, {"gpu_budget_seconds": 0.5}, "cumulative gpu"),
    ],
)
def test_real_round_rejects_before_enqueue_without_child_effects(
    managed, expected, settings, reason
):
    store, _, work = managed
    _, process, marker = agent(store, work, expected, returncode=2, **settings)
    finish(process)
    assert not marker.exists()
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")
    assert store.task("toy")["gpu_seconds"] == 0
    response = json.loads((work / "response.json").read_text())
    assert reason in response["stdout"] + response["stderr"]


def test_real_queue_wait_rechecks_parent_time_before_child_start(managed):
    store, _, work = managed
    with exclusive(lock_path("gpu", "cpu")) as handle:
        assert handle is not None
        _, process, marker = agent(store, work, 8, returncode=2)
        wait_for(lambda: store.rows("SELECT id FROM jobs WHERE kind='gpu' AND status='queued'"))
        time.sleep(2.4)  # 10s round accepted 8s, but no longer fits after waiting.
        assert not marker.exists()
    finish(process)
    job = store.one("SELECT * FROM jobs WHERE kind='gpu'")
    assert job["status"] == "cancelled"
    assert "duration admission rejected" in job["error"]
    assert "parent round" in job["error"]
    assert job["child_pid"] is None
    assert not marker.exists()


def test_sufficient_duration_runs_and_bills_actual_usage(managed):
    store, _, work = managed
    _, process, marker = agent(store, work, 3)
    finish(process)
    assert marker.read_text() == "ran"
    assert store.one("SELECT status FROM jobs WHERE kind='gpu'")["status"] == "succeeded"
    assert 0 < store.task("toy")["gpu_seconds"] < 3
    event = store.one("SELECT data FROM events WHERE kind='gpu_duration_declared'")
    assert json.loads(event["data"])["seconds"] == 3


def test_new_round_never_renews_cumulative_budget(managed):
    store, _, work = managed
    store.db.execute("UPDATE tasks SET gpu_seconds=4 WHERE id='toy'")
    for _ in range(2):
        # Exercise the next admitted round without waiting for recovery backoff.
        # Only the test clock gate moves; original usage and limits stay intact.
        store.db.execute("UPDATE tasks SET next_at=0 WHERE id='toy'")
        _, process, marker = agent(store, work, 2, returncode=2, gpu_budget_seconds=5)
        finish(process)
        assert not marker.exists()
        assert store.task("toy")["gpu_seconds"] == 4
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")


def test_unreported_parent_elapsed_time_counts_against_wall_budget(managed):
    store, _, work = managed
    replace_command(store, [sys.executable, "-c", "pass"], wall_budget_seconds=6)
    parent = store.schedule()[0]
    store.claim(parent, os.getpid(), identity(os.getpid()))
    store.db.execute("UPDATE jobs SET started=started-5 WHERE id=?", (parent,))
    with pytest.raises(ValueError, match="cumulative wall"):
        run(store.root, "toy", parent, [sys.executable, "-c", "pass"], expected_seconds=2)
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")
    store.finish(parent, "succeeded")


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan"), True])
def test_invalid_duration_never_enqueues(managed, duration):
    store, _, _ = managed
    with pytest.raises(ValueError, match="finite and positive"):
        run(store.root, "toy", "unused", ["unused"], expected_seconds=duration)
    assert not store.rows("SELECT id FROM jobs WHERE kind='gpu'")
