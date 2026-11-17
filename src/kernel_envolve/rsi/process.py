"""Bounded operator-supplied adapters. This is supervision, not an OS sandbox."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from ..execution import stop


def invoke(command, directory, packet, timeout, *, device):
    env = {k: v for k, v in os.environ.items() if not k.startswith("KERNEL_ENVOLVE_")}
    # Keep evaluator subprocesses on this exact installation, even when the caller's
    # PYTHONPATH contains relative paths and the adapter changes working directory.
    source = str(Path(__file__).resolve().parents[2])
    inherited = [str(Path(p).resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([source, *inherited]))
    env.update(
        KERNEL_ENVOLVE_AUTO_UPDATE="0",
        KERNEL_ENVOLVE_UPDATE_SESSION="1",
        CUDA_VISIBLE_DEVICES="" if device == "cpu" else device,
    )
    (directory / "request.json").write_text(json.dumps(packet, ensure_ascii=False))
    previous = {}

    def interrupted(signum, frame):
        handler = previous.get(signum)
        if callable(handler):
            handler(signum, frame)
        raise InterruptedError("RSI adapter interrupted")

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, interrupted)
    process = None
    started = time.monotonic()
    try:
        with (
            (directory / "request.json").open("rb") as source,
            (directory / "stdout.log").open("wb") as out,
            (directory / "stderr.log").open("wb") as err,
        ):
            process = subprocess.Popen(
                command,
                cwd=directory,
                stdin=source,
                stdout=out,
                stderr=err,
                env=env,
                start_new_session=True,
            )
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("RSI adapter exceeded its fixed time budget")
                if any(
                    (directory / name).stat().st_size > 1024 * 1024
                    for name in ("stdout.log", "stderr.log")
                ):
                    raise ValueError("RSI adapter log limit exceeded")
                time.sleep(0.025)
            if process.returncode:
                raise ValueError(f"RSI adapter exited {process.returncode}")
    finally:
        if process is not None:
            stop(process)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return time.monotonic() - started
