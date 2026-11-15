"""Owned CPU subprocesses exercise inheritance without GPU or host sysctl changes."""

import json
import os
import resource
import subprocess
import sys

import pytest

from kernel_envolve.manager import worker
from kernel_envolve.manager.process import gpu_locks, identity

pytest_plugins = ["test_manager"]


def run_python(code):
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10, check=True
    )
    return json.loads(result.stdout)


def test_limit_reaches_exec_and_grandchild_without_changing_caller():
    before = resource.getrlimit(resource.RLIMIT_CORE)
    code = """
import json,resource,subprocess,sys
from kernel_envolve.manager import worker
worker.sys.platform='linux'
previous=resource.getrlimit(resource.RLIMIT_CORE)
if previous[1] == 0:
    print(json.dumps({'skip':True}))
    raise SystemExit
probe="import json,resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_CORE)))"
child="import subprocess,sys; subprocess.run([sys.executable,'-c',"+repr(probe)+"],check=True)"
with worker.inherited_core_limit('gpu') as warning:
    observed=json.loads(subprocess.check_output([sys.executable,'-c',child],text=True))
print(json.dumps(dict(previous=previous,restored=resource.getrlimit(resource.RLIMIT_CORE),
                     observed=observed,warning=warning)))
"""
    result = run_python(code)
    assert resource.getrlimit(resource.RLIMIT_CORE) == before
    if result.get("skip"):
        pytest.skip("The inherited hard limit is already zero")
    assert result["observed"] == [1, result["previous"][1]]
    assert result["restored"] == result["previous"] and result["warning"] is None


def test_hard_zero_remains_zero_and_command_still_runs():
    result = run_python("""
import json,resource,subprocess,sys
from kernel_envolve.manager import worker
worker.sys.platform='linux'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
with worker.inherited_core_limit('gpu') as warning:
    child=subprocess.check_output([sys.executable,'-c',
      'import json,resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_CORE)))'],text=True)
print(json.dumps(dict(child=json.loads(child),warning=warning,
                     restored=resource.getrlimit(resource.RLIMIT_CORE))))
""")
    assert result["child"] == result["restored"] == [0, 0]
    assert "hard limit is 0" in result["warning"] and "may still run" in result["warning"]


@pytest.mark.parametrize("kind,platform", [("agent", "linux"), ("gpu", "darwin")])
def test_agent_and_non_linux_limits_are_not_touched(monkeypatch, kind, platform):
    monkeypatch.setattr(worker.sys, "platform", platform)

    def forbidden(*_):
        raise AssertionError("Unrelated process limits were inspected or changed")

    monkeypatch.setattr(worker.resource, "getrlimit", forbidden)
    monkeypatch.setattr(worker.resource, "setrlimit", forbidden)
    with worker.inherited_core_limit(kind) as warning:
        assert warning is None


def test_rejected_limit_preserves_execution_and_reports_limitation(monkeypatch):
    monkeypatch.setattr(worker.sys, "platform", "linux")
    monkeypatch.setattr(worker.resource, "getrlimit", lambda _: (0, 1024))
    changes = []

    def denied(kind, value):
        changes.append((kind, value))
        raise PermissionError("fixture permission denied")

    monkeypatch.setattr(worker.resource, "setrlimit", denied)
    with worker.inherited_core_limit("gpu") as warning:
        assert "fixture permission denied" in warning
    assert changes == [(resource.RLIMIT_CORE, (1, 1024))]


def test_launch_failure_restores_worker_limit():
    result = run_python("""
import json,resource
from kernel_envolve.manager import worker
worker.sys.platform='linux'
before=resource.getrlimit(resource.RLIMIT_CORE)
try:
    with worker.inherited_core_limit('gpu'):
        raise OSError('fixture spawn failure')
except OSError:
    pass
print(json.dumps(dict(before=before,after=resource.getrlimit(resource.RLIMIT_CORE))))
""")
    assert result["before"] == result["after"]


def test_gpu_supervisor_child_inherits_limit_and_original_locks(managed, monkeypatch):
    store, _, work = managed
    before = resource.getrlimit(resource.RLIMIT_CORE)
    if before[1] == 0:
        pytest.skip("The inherited hard limit is already zero")
    monkeypatch.setattr(worker.sys, "platform", "linux")
    parent = store.schedule()[0]
    assert store.claim(parent, os.getpid(), identity(os.getpid()))
    command = (
        "import json,resource; from pathlib import Path; "
        "from kernel_envolve.execution import managed_lock_fds; "
        "Path('limits.json').write_text(json.dumps(dict("
        "limits=resource.getrlimit(resource.RLIMIT_CORE),locks=len(managed_lock_fds()))))"
    )
    job = store.enqueue(store.task("toy"), "gpu", [sys.executable, "-c", command], parent)
    assert store.claim(job, os.getpid(), identity(os.getpid()))
    with gpu_locks(["cpu"]) as handles:
        assert worker.supervise(store, job, handles) == 0
    assert json.loads((work / "limits.json").read_text()) == {"limits": [1, before[1]], "locks": 1}
    assert resource.getrlimit(resource.RLIMIT_CORE) == before
    store.finish(parent, "cancelled")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux kernel core-dump sentinel")
def test_linux_abort_after_exec_has_no_core_dump_flag():
    # Uses the existing core_pattern, whether file or pipe. Never changes host sysctls.
    result = run_python("""
import json,os,resource,signal,sys,time
from pathlib import Path
from kernel_envolve.manager.worker import inherited_core_limit
if resource.getrlimit(resource.RLIMIT_CORE)[1] == 0:
    print(json.dumps({'skip':True}))
    raise SystemExit
started=time.monotonic()
with inherited_core_limit('gpu') as warning:
    assert warning is None
    pid=os.fork()
    if pid==0:
        os.execv(sys.executable,[sys.executable,'-c',
            'import os,resource; assert resource.getrlimit(resource.RLIMIT_CORE)[0]==1; os.abort()'])
_,status=os.waitpid(pid,0)
print(json.dumps(dict(signal=os.WTERMSIG(status),core=os.WCOREDUMP(status),
                     seconds=time.monotonic()-started,
                     core_pattern=Path('/proc/sys/kernel/core_pattern').read_text().strip())))
""")
    if result.get("skip"):
        pytest.skip("The inherited hard limit is already zero")
    assert result["signal"] == 6 and result["core"] is False
    assert result["seconds"] < 5
