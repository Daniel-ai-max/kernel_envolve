"""Real SIGKILL/owner loss tests for external model attempts, without GPUs or APIs."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernel_envolve.manager.execution.host_attempt import HostAttempt, model_result, read_events
from kernel_envolve.manager.execution.host_process import ModelHost
from kernel_envolve.manager.execution.host_recovery import HostRecovery
from kernel_envolve.manager.execution.process import alive, identity


def wait_for(check, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if value := check():
            return value
        time.sleep(0.03)
    raise AssertionError("Process transition did not finish")


def test_linux_identity_handles_exit_during_proc_read_without_hiding_permissions(monkeypatch):
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: True if str(p) == "/proc" else original_is_dir(p))

    def vanished(_path, *args, **kwargs):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(Path, "read_text", vanished)
    assert identity(1234) is None

    def denied(_path, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        identity(1234)


def start(path, script, *, seconds=30):
    return HostAttempt.start(
        path,
        argv=[sys.executable, "-c", script],
        prompt="continue",
        cwd=path.parent,
        deadline=time.time() + seconds,
        metadata={"model": "fixture", "session": "original"},
    )


STARTED = "import json,time; print(json.dumps({'type':'thread.started','thread_id':'original'}),flush=True); "


def test_killed_model_recovers_original_session_only_after_durable_exit(tmp_path):
    first = start(tmp_path / "first", STARTED + "time.sleep(60)")
    try:
        record = wait_for(
            lambda: (
                json.loads((first.directory / "process.json").read_text())
                if (first.directory / "process.json").exists()
                else None
            )
        )
        wait_for(lambda: read_events(first.directory / "trace.jsonl"))
        assert HostAttempt(first.directory).outcome() is None
        os.kill(record["pid"], signal.SIGKILL)
        receipt = wait_for(first.outcome)
        assert receipt["returncode"] == -signal.SIGKILL and receipt["cleanup_confirmed"]
        ledger = HostRecovery(
            tmp_path / "recovery.json",
            model="fixture",
            session="original",
            round_id="round",
            deadline=time.time() + 600,
        )
        ledger.failed(
            read_events(first.directory / "trace.jsonl"),
            receipt["returncode"],
            trace_sha256=hashlib.sha256((first.directory / "trace.jsonl").read_bytes()).hexdigest(),
            now=receipt["finished_at"],
            process_exited=receipt["cleanup_confirmed"],
        )
        assert (
            ledger.admit(
                now=receipt["finished_at"] + 6,
                active=True,
                reconciled=False,
                model="fixture",
                session="original",
            )
            == "wait"
        )
        assert (
            ledger.admit(
                now=receipt["finished_at"] + 6,
                active=True,
                reconciled=True,
                model="fixture",
                session="original",
            )
            == "resume"
        )
        second = start(
            tmp_path / "second", STARTED + "print(json.dumps({'type':'turn.completed'}),flush=True)"
        )
        try:
            assert wait_for(second.outcome)["returncode"] == 0
            assert read_events(second.directory / "trace.jsonl")[0]["thread_id"] == "original"
        finally:
            second.stop()
    finally:
        first.stop()


def test_broker_sigkill_preserves_exit_receipt_for_new_controller(tmp_path):
    directory = tmp_path / "attempt"
    launcher = tmp_path / "controller.py"
    launcher.write_text(
        "import sys,time\nfrom kernel_envolve.manager.execution.host_attempt import HostAttempt\n"
        f"a=HostAttempt.start({str(directory)!r}, argv={[sys.executable, '-c', STARTED + 'time.sleep(60)']!r}, "
        f"prompt='resume',cwd={str(tmp_path)!r},deadline=time.time()+30,metadata={{'session':'original'}})\n"
        "print('ready',flush=True)\ntime.sleep(60)\n"
    )
    owner = subprocess.Popen([sys.executable, str(launcher)], stdout=subprocess.PIPE, text=True)
    try:
        assert owner.stdout.readline().strip() == "ready"
        wait_for(lambda: read_events(directory / "trace.jsonl"))
        model = json.loads((directory / "process.json").read_text())
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=5)
        receipt = wait_for(HostAttempt(directory).outcome)
        assert receipt["reason"] == "owner_lost"
        assert receipt["cleanup_confirmed"]
        assert not alive(model["pid"], model["identity"])
        assert read_events(directory / "trace.jsonl")[0]["thread_id"] == "original"
        with pytest.raises(FileExistsError):
            start(directory, "pass")  # Restart may inspect/reconcile, never replay this attempt.
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        owner.stdout.close()


def test_leader_exit_cleans_descendant_holding_stdout(tmp_path):
    pid_file = tmp_path / "descendant"
    script = (
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        f"p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(p.pid))\ntime.sleep(.2)\n"
    )
    attempt = start(tmp_path / "attempt", script)
    try:
        wait_for(pid_file.exists)
        pid = int(pid_file.read_text())
        birth = identity(pid)
        receipt = wait_for(attempt.outcome)
        assert receipt["cleanup_confirmed"] and receipt["returncode"] == 0
        assert not alive(pid, birth)
    finally:
        attempt.stop()


def test_supervisor_sigkill_does_not_authorize_blind_resume(tmp_path):
    attempt = start(tmp_path / "attempt", STARTED + "time.sleep(60)")
    try:
        wait_for(lambda: read_events(attempt.directory / "trace.jsonl"))
        record = json.loads((attempt.directory / "process.json").read_text())
        os.kill(attempt.process.pid, signal.SIGKILL)
        attempt.process.wait(timeout=5)
        receipt = wait_for(attempt.outcome)
        assert not receipt["cleanup_confirmed"]
        assert receipt["reason"] == "supervisor_receipt_missing"
        assert not alive(record["pid"], record["identity"])
    finally:
        attempt.stop()


def test_frozen_live_model_cannot_hold_lease_until_full_budget(tmp_path):
    attempt = HostAttempt.start(
        tmp_path / "attempt",
        argv=[sys.executable, "-c", STARTED + "time.sleep(60)"],
        prompt="continue",
        cwd=tmp_path,
        deadline=time.time() + 30,
        idle_seconds=0.3,
        metadata={"session": "original"},
    )
    try:
        wait_for(lambda: read_events(attempt.directory / "trace.jsonl"))
        pid = json.loads((attempt.directory / "process.json").read_text())["pid"]
        os.kill(pid, signal.SIGSTOP)
        result = wait_for(lambda: model_result(attempt.directory))
        assert result["termination_reason"] == "model_stalled"
        assert result["status"] == "failed" and result["process_exited"]
    finally:
        attempt.stop()


def test_absolute_deadline_and_missing_session_do_not_become_success(tmp_path):
    attempt = start(tmp_path / "attempt", "import time; time.sleep(60)", seconds=0.5)
    try:
        receipt = wait_for(attempt.outcome)
        assert receipt["reason"] == "deadline" and receipt["cleanup_confirmed"]
        assert not read_events(attempt.directory / "trace.jsonl")
        assert model_result(attempt.directory)["status"] == "cancelled"
    finally:
        attempt.stop()


@pytest.mark.parametrize(
    "script,category",
    [
        ("pass", None),
        (
            "import sys; sys.stderr.write('Error: thread/resume: failed to read session metadata: rollout is empty'); sys.exit(1)",
            "model_resume_state_unavailable",
        ),
        (
            STARTED
            + "print(json.dumps({'type':'item.started','item':{'id':'tool','type':'command_execution'}})); print(json.dumps({'type':'turn.completed'}))",
            "host_tool_cleanup_unconfirmed",
        ),
        (
            "import json; print(json.dumps({'type':'thread.started','thread_id':'other'}))",
            "host_session_unavailable",
        ),
        (
            STARTED
            + "print(json.dumps({'type':'turn.failed','error':{'code':'invalid_api_key'}}))",
            "model_access_denied",
        ),
    ],
)
def test_only_verified_model_completion_is_successful(tmp_path, script, category):
    attempt = start(tmp_path / "attempt", script)
    try:
        result = wait_for(lambda: model_result(attempt.directory))
        assert result["status"] == "failed" and result["exit_code"] != 0
        assert model_result(attempt.directory) == result  # Immutable replay.
        if category:
            assert result["model_failure"]["category"] == category
    finally:
        attempt.stop()


def test_model_adapter_restores_original_attempt_and_resumes_exact_session(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    codex = tmp_path / "fake-codex"
    codex.write_text(
        "#!/usr/bin/env python3\n"
        + """import json,sys,time,os
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
prompt=sys.stdin.read()
print(json.dumps({'type':'thread.started','thread_id':'original'}),flush=True)
Path('argv.json').write_text(json.dumps(sys.argv))
if 'hang' in prompt:time.sleep(60)
else:
 assert 'resume' in sys.argv and 'original' in sys.argv
 Path('continued').write_text('same-session')
 print(json.dumps({'type':'turn.completed'}),flush=True)
"""
    )
    codex.chmod(0o700)
    root = tmp_path / "host"
    launcher = tmp_path / "owner.py"
    launcher.write_text(
        "import time\nfrom kernel_envolve.manager.execution.host_process import ModelHost\n"
        f"m=ModelHost({str(root)!r},{str(tmp_path)!r},codex={str(codex)!r},model_options={{'model':'fixture'}})\n"
        "m.start({'id':'attempt-0'},'hang',time.time()+60)\ntime.sleep(60)\n"
    )
    owner = subprocess.Popen([sys.executable, str(launcher)])
    host = ModelHost(root, tmp_path, codex=str(codex), model_options={"model": "fixture"})
    try:
        wait_for(lambda: (tmp_path / "argv.json").exists())
        owner.kill()
        owner.wait(timeout=5)
        intent = host.restore("attempt-0")
        result = wait_for(host.collect)
        assert result["process_exited"] and result["termination_reason"] == "owner_lost"
        assert result["session_id"] == "original"
        host.start({"id": "attempt-1"}, "continue", intent["deadline"])
        result = wait_for(host.collect)
        assert result["status"] == "succeeded"
        assert (tmp_path / "continued").read_text() == "same-session"
        argv = json.loads((tmp_path / "argv.json").read_text())
        host.restore("attempt-1")
        assert host.collect() == result
        assert json.loads((tmp_path / "argv.json").read_text()) == argv
        other = ModelHost(root, tmp_path, model_options={"model": "other"})
        with pytest.raises(ValueError, match="different research model"):
            other.restore("attempt-1")
    finally:
        host.terminate()
        if owner.poll() is None:
            owner.kill()
            owner.wait()


def test_completed_model_racing_owner_loss_is_replayed_not_retried(tmp_path):
    attempt = start(tmp_path / "attempt", STARTED + "print(json.dumps({'type':'turn.completed'}))")
    try:
        receipt = wait_for(attempt.outcome)
        assert receipt["returncode"] == 0 and receipt["cleanup_confirmed"]
        # EOF and model exit can be observed in the same selector iteration.
        receipt["reason"] = "owner_lost"
        (attempt.directory / "exit.json").write_text(json.dumps(receipt))
        result = model_result(attempt.directory)
        assert result["status"] == "succeeded"
        assert model_result(attempt.directory) == result
    finally:
        attempt.stop()


def test_model_host_cannot_nest_inside_gpu_job(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_ENVOLVE_GPU_JOB", "owned-gpu-job")
    host = ModelHost(tmp_path / "host", tmp_path, model_options={"model": "fixture"})
    with pytest.raises(ValueError, match="inside a managed GPU job"):
        host.start({"id": "attempt"}, "continue", time.time() + 30)
    assert not (tmp_path / "host/turns").exists()


def test_bridge_owner_eof_cleans_a_still_running_model(tmp_path):
    pid_file = tmp_path / "model.pid"
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kernel_envolve.manager.execution.bridge",
            str(tmp_path / "exit.json"),
        ],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        command = [
            sys.executable,
            "-c",
            f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)",
        ]
        supervisor.stdin.write((json.dumps(command) + "\n").encode())
        supervisor.stdin.flush()
        wait_for(pid_file.exists)
        pid = int(pid_file.read_text())
        birth = identity(pid)
        supervisor.stdin.close()
        supervisor.wait(timeout=5)
        wait_for(lambda: not alive(pid, birth))
    finally:
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGKILL)
            supervisor.wait()
