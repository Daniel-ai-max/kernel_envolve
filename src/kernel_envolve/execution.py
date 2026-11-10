"""Shared exclusive locks and cleanup of evaluator-owned process groups."""

import contextlib
import fcntl
import os
import signal
import subprocess


def managed_lock_fd():
    """Validate inherited ownership without acquiring or replacing the device lock."""
    if not os.environ.get("KERNEL_ENVOLVE_GPU_JOB"):
        return None
    try:
        descriptor = int(os.environ["KERNEL_ENVOLVE_LOCK_FD"])
        os.fstat(descriptor)
    except (KeyError, ValueError, OverflowError, OSError) as exc:
        raise ValueError(
            "Managed GPU lock descriptor KERNEL_ENVOLVE_LOCK_FD is missing or invalid. "
            "Launch the evaluator directly, transfer control with os.execvpe, or preserve "
            "this descriptor with pass_fds in intermediary subprocesses. Do not remove "
            "the manager environment or acquire a replacement lock."
        ) from exc
    return descriptor


def spawn(*args, **kwargs):
    """Register managed evaluator children so supervisor recovery can reap separate groups."""
    descriptor = managed_lock_fd()
    if descriptor is not None:
        # All evaluator process groups retain the device lock through supervisor crashes,
        # even while importing/compiling before they have a visible CUDA context.
        kwargs["pass_fds"] = tuple(set(kwargs.get("pass_fds", ())) | {descriptor})
    process = subprocess.Popen(*args, **kwargs)
    if os.environ.get("KERNEL_ENVOLVE_GPU_JOB"):
        try:
            from .manager.process import identity
            from .manager.store import Store

            store = Store(os.environ["KERNEL_ENVOLVE_STATE"])
            try:
                birth = identity(process.pid)
                if birth:
                    store.db.execute(
                        "INSERT OR IGNORE INTO processes(job,pid,identity) VALUES(?,?,?)",
                        (os.environ["KERNEL_ENVOLVE_GPU_JOB"], process.pid, birth),
                    )
            finally:
                store.close()
        except BaseException:
            stop(process)
            raise
    return process


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
