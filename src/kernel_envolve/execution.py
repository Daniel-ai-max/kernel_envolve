"""Shared exclusive locks and cleanup of evaluator-owned process groups."""

import contextlib
import fcntl
import json
import os
import signal
import subprocess


def managed_lock_fds():
    """Validate every inherited device lock without acquiring replacements."""
    if not os.environ.get("KERNEL_ENVOLVE_GPU_JOB"):
        return ()
    try:
        first = int(os.environ["KERNEL_ENVOLVE_LOCK_FD"])
        descriptors = json.loads(os.environ.get("KERNEL_ENVOLVE_LOCK_FDS", f"[{first}]"))
        if (
            not isinstance(descriptors, list)
            or not descriptors
            or descriptors[0] != first
            or any(type(fd) is not int for fd in descriptors)
            or len(set(descriptors)) != len(descriptors)
        ):
            raise ValueError("Invalid device lock set")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device_count = len(visible.split(",")) if visible else 1
        if len(descriptors) != device_count:
            raise ValueError("Visible device count does not match inherited lock count")
        for descriptor in descriptors:
            os.fstat(descriptor)
    except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
        raise ValueError(
            "Managed GPU lock descriptor KERNEL_ENVOLVE_LOCK_FD is missing or invalid. "
            "All KERNEL_ENVOLVE_LOCK_FDS must also remain inherited. "
            "Launch the evaluator directly, transfer control with os.execvpe, or preserve "
            "these descriptors with pass_fds in intermediary subprocesses. Do not remove "
            "the manager environment or acquire replacement locks."
        ) from exc
    return tuple(descriptors)


def managed_lock_fd():
    """Backward-compatible first descriptor, after validating the entire group."""
    descriptors = managed_lock_fds()
    return descriptors[0] if descriptors else None


def spawn(*args, **kwargs):
    """Register managed evaluator children so supervisor recovery can reap separate groups."""
    descriptors = managed_lock_fds()
    if descriptors:
        # All evaluator process groups retain the device lock through supervisor crashes,
        # even while importing/compiling before they have a visible CUDA context.
        kwargs["pass_fds"] = tuple(set(kwargs.get("pass_fds", ())) | set(descriptors))
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
