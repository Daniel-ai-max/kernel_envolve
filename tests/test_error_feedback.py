"""Compiler failures must provide actionable feedback through the normal CLI."""

import json
import subprocess
import sys

import pytest

from kernel_envolve.cli import evaluate, promote
from kernel_envolve.worker import error_details


def test_captured_compiler_error_is_visible_in_cli(task):
    command = (
        "import sys; sys.stdout.buffer.write(b'compiler output'); "
        "sys.stderr.buffer.write(b'x' * 9000 + b'\\xff missing CUDA header'); sys.exit(2)"
    )
    (task / "candidate.py").write_text(
        "import subprocess, sys\n"
        f"subprocess.run([sys.executable, '-c', {command!r}], check=True, capture_output=True)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "kernel_envolve.cli", "evaluate", "--task", str(task)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 2, proc.stderr
    feedback = json.loads(proc.stdout)
    error = feedback["error"]
    assert error["phase"] == "setup"
    assert error["returncode"] == 2
    assert error["stdout_tail"] == "compiler output"
    assert error["stdout_truncated"] is False
    assert error["stderr_tail"].endswith("\ufffd missing CUDA header")
    assert len(error["stderr_tail"]) == 8192
    assert error["stderr_truncated"] is True
    assert feedback["eligible_for_best"] is False
    with pytest.raises(ValueError, match="Not eligible"):
        promote(feedback["run"])


def test_runtime_subprocess_failure_has_correctness_phase(task):
    (task / "candidate.py").write_text(
        "import subprocess, sys\n"
        "def run(x,y):\n"
        " subprocess.run([sys.executable,'-c',\"import sys; print('launch failed',file=sys.stderr); sys.exit(3)\"],check=True,capture_output=True,text=True)\n"
    )
    _, result = evaluate(task)
    assert result["error"]["phase"] == "correctness"
    assert result["error"]["stderr_tail"] == "launch failed\n"
    assert not result["samples"]
    assert not result["decision"]["eligible_for_best"]


def test_subprocess_timeout_preserves_partial_output():
    error = error_details(
        subprocess.TimeoutExpired("compiler", 1, output=b"partial output", stderr=b"partial error"),
        "setup",
    )
    assert error["stdout_tail"] == "partial output"
    assert error["stderr_tail"] == "partial error"
    assert error["phase"] == "setup"
    assert "returncode" not in error
