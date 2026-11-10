"""Exercise real process groups, without GPU allocation or external services."""

import os
import signal
import subprocess
import sys
import time

import pytest

from kernel_envolve.execution import stop


@pytest.mark.parametrize("parent_exits", [True, False])
@pytest.mark.parametrize("ignore_term", [True, False])
def test_cleanup_reaps_descendants_after_supervisor_exits(tmp_path, parent_exits, ignore_term):
    ready = tmp_path / "ready"
    handler = "signal.SIG_IGN" if ignore_term else "signal.SIG_DFL"
    child = (
        "import signal,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM,{handler}); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f'subprocess.Popen([sys.executable,"-c",{child!r}]); '
        f"\nwhile not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
    )
    if not parent_exits:
        parent += "time.sleep(60)\n"
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        if parent_exits:
            process.wait(timeout=5)
        stop(process)
        # Cleanup may use SIGKILL after a grace period; either outcome must stop the group.
        for _ in range(50):
            result = subprocess.run(
                ["ps", "-axo", "pgid=,stat="], capture_output=True, text=True, check=True
            )
            live = [
                line
                for line in result.stdout.splitlines()
                if line.split()[0] == str(process.pid) and not line.split()[1].startswith("Z")
            ]
            if not live:
                break
            time.sleep(0.02)
        assert not live, "Child outlived its completed evaluator process"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
