"""Shared exclusive locks and cleanup of evaluator-owned process groups."""

import contextlib
import fcntl
import os
import signal
import subprocess


@contextlib.contextmanager
def lock(path):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another operation holds {path}; inspect it before retrying") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stop(process, grace_seconds=2):
    """Stop an owned start_new_session process group, including surviving children."""

    def send(sig):
        # Reap an already-dead leader before signalling its remaining group (macOS).
        process.poll()
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may have exited while a compiler/launcher child is still alive.
        send(signal.SIGKILL)
        process.wait()
