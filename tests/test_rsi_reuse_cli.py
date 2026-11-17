"""Real command entrypoints exercise operator workflows without model/GPU calls."""

import json
import os
import subprocess
import sys
from pathlib import Path

from test_rsi_reuse_memory import INITIAL, make_task, trial

from kernel_envolve.rsi import reuse_memory
from kernel_envolve.storage import digest

pytest_plugins = ["test_rsi_reuse_memory"]


def cli(root, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "kernel_envolve.rsi.reuse_cli",
            "--state",
            str(root),
            *map(str, args),
        ],
        cwd=root.parent,
        env={**os.environ, "PYTHONPATH": str(Path(reuse_memory.__file__).resolve().parents[2])},
        text=True,
        capture_output=True,
        timeout=30,
    )


def result(reply):
    assert reply.returncode == 0, reply.stderr
    return json.loads(reply.stdout)


def test_cli_prepare_observe_disable_and_restart(setup, tmp_path):
    source, _, scope, _, record = setup
    root = tmp_path / "cli-memory"
    assert result(cli(root, "init", "--archive", source.archive))["status"] == "initialized"
    assert result(cli(root, "register", record))["entry"]
    task = make_task(tmp_path / "cli-target")
    out = tmp_path / "cli-out"
    offer = result(cli(root, "prepare", "--scope", scope, "--task", task, "--artifacts", out))
    _, final, _ = trial(task, out, snapshot=offer["snapshot"])
    feedback = result(cli(root, "observe", offer["ticket"], "--final", final))
    assert feedback["status"] == "helpful"
    assert result(cli(root, "observe", offer["ticket"], "--final", final)) == feedback
    assert result(cli(root, "disable")) == {"enabled": False}
    (task / "candidate.py").write_text(INITIAL)
    assert (
        result(
            cli(root, "prepare", "--scope", scope, "--task", task, "--artifacts", tmp_path / "off")
        )["reason"]
        == "disabled"
    )
    assert result(cli(root, "enable")) == {"enabled": True}
    status = result(cli(root, "status"))
    assert status["code_compatible"] and status["trial_counts"] == {"helpful": 1}


def test_cli_stale_version_can_be_stopped_and_inspected(setup, tmp_path):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    offer = memory.prepare(scope, task, {}, tmp_path / "out")
    changed = memory.config | {"implementation": "0" * 64}
    memory.db.execute("UPDATE meta SET value=? WHERE key='config'", (json.dumps(changed),))
    memory.db.execute(
        "UPDATE meta SET value=? WHERE key='config_hash'", (json.dumps(digest(changed)),)
    )
    before = (memory.root / "reuse.sqlite").read_bytes()
    status = result(cli(memory.root, "status"))
    assert not status["code_compatible"] and status["pending"][0]["ticket"] == offer["ticket"]
    assert (memory.root / "reuse.sqlite").read_bytes() == before
    assert result(cli(memory.root, "disable")) == {"enabled": False}
    reply = cli(memory.root, "enable")
    assert reply.returncode == 2 and json.loads(reply.stderr)["error_type"] == "ValueError"
    assert not reply.stdout
    closed = result(
        cli(memory.root, "abandon", offer["ticket"], "--reason", "Stopped after code upgrade")
    )
    assert closed["status"] == "incomplete"
    assert result(cli(memory.root, "status"))["trial_counts"] == {"incomplete": 1}


def test_cli_missing_state_does_not_create_an_empty_database(tmp_path):
    root = tmp_path / "missing"
    reply = cli(root, "status")
    assert reply.returncode == 2
    assert json.loads(reply.stderr)["error_type"] == "OperationalError"
    assert not root.exists()


def test_cli_rejects_corrupt_profile_without_reserving_work(setup, tmp_path):
    memory, _, scope, _, _ = setup
    task = make_task(tmp_path / "target")
    profile = tmp_path / "profile.json"
    profile.write_text("{broken")
    reply = cli(
        memory.root,
        "prepare",
        "--scope",
        scope,
        "--task",
        task,
        "--artifacts",
        tmp_path / "out",
        "--profile",
        profile,
    )
    assert reply.returncode == 2 and not reply.stdout
    assert json.loads(reply.stderr)["error_type"] == "JSONDecodeError"
    assert result(cli(memory.root, "status"))["trial_counts"] == {}
