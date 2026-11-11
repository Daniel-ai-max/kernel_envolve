"""POSIX process identity and host-wide locks; an expired heartbeat is never a lease release."""

import contextlib
import fcntl
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path


def identity(pid):
    """Return a birth identity, None for absence, and raise on an unavailable probe."""
    if Path("/proc").is_dir():
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] == "Z":
                return None
            return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
        except FileNotFoundError:
            return None
    reply = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat="],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if reply.returncode == 1 and not reply.stdout.strip():
        return None
    if reply.returncode:
        raise OSError(f"Process probe failed: {reply.stderr.strip()}")
    fields = reply.stdout.split()
    return None if not fields or fields[-1].startswith("Z") else " ".join(fields[:-1])


def alive(pid, birth):
    return bool(pid and birth and identity(pid) == birth)


def lock_path(kind, name):
    root = Path.home() / ".cache" / "kernel-envolve" / "locks"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(name.encode()).hexdigest()
    return root / f"{kind}-{key}.lock"


@contextlib.contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def gpu_locks(devices):
    """Acquire every existing per-device lock, or release the partial set immediately."""
    with contextlib.ExitStack() as stack:
        handles = []
        for device in sorted(devices):
            handle = stack.enter_context(exclusive(lock_path("gpu", device)))
            if handle is None:
                stack.close()
                yield None
                return
            handles.append(handle)
        yield tuple(handles)


def gpu_idle(device):
    if device == "cpu":
        return
    found = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", device],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    if found.stdout.strip() != device:
        raise ValueError("Physical GPU identity does not match the configured UUID")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits", "-i", device],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    if processes.stdout.strip():
        raise ValueError(f"GPU has existing compute processes: {processes.stdout.strip()}")


def gpu_group_idle(devices):
    """Probe every physical device, retaining all failures for cleanup diagnosis."""
    errors = []
    for device in devices:
        try:
            gpu_idle(device)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"{device}: {exc}")
    if errors:
        raise ValueError("; ".join(errors))


def terminate(process, grace=3):
    """The caller owns this Popen and group; finish descendants even if the leader exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Give children a chance to exit after their parent (and its cleanup handlers).
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def stop_recorded(pid, birth):
    """Never signal an absent or recycled leader. A remaining device process blocks reuse."""
    if not alive(pid, birth):
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 3
    while alive(pid, birth) and time.monotonic() < deadline:
        time.sleep(0.1)
    current = identity(pid)
    if current is None or current == birth:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def reap_registered(store, job):
    for child in store.rows("SELECT * FROM processes WHERE job=?", (job,)):
        stop_recorded(child["pid"], child["identity"])
