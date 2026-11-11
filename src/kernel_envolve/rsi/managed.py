"""Explicit per-measurement Manager admission for the experimental research loop.

Only the evaluator runs under the device lock. Queue waiting and execution share
one local deadline, in addition to the Manager's cumulative and round budgets.
This POSIX adapter is not a sandbox, a publisher, or an independent final gate.
"""

import os
import signal
import sys
import time
from pathlib import Path

from ..execution import managed_lock_fd
from ..manager.evidence import inspect_run
from ..manager.gpu import run as gpu_run
from ..manager.store import Store
from ..storage import config, contract, digest, file_hash, read_json, write_json
from .model import keys, number, read


class _DeadlineExpired(Exception):
    """Do not let device-probe OSError recovery swallow the local deadline."""


def binding(task, *, worker=False):
    """Bind the request to the current registered generic task and live round."""
    if bool(os.environ.get("KERNEL_ENVOLVE_GPU_JOB")) != worker:
        raise ValueError("Research decisions must run outside the Manager GPU job")
    names = ("KERNEL_ENVOLVE_STATE", "KERNEL_ENVOLVE_TASK", "KERNEL_ENVOLVE_ROUND")
    if any(not os.environ.get(name) for name in names):
        raise ValueError("Managed evaluation requires the current Manager agent round")
    root, task_id, round_id = (os.environ[name] for name in names)
    store = Store(root, readonly=True)
    try:
        registered = store.task(task_id)
        parent = store.job(round_id)
        spec = registered["spec"]
        if (
            spec["kind"] != "generic"
            or Path(spec["task_dir"]).resolve() != task
            or parent["task"] != task_id
            or parent["kind"] != "agent"
            or parent["status"] != "running"
            or parent["cancel_requested"]
            or registered["desired"] != "active"
            or contract(task, config(task)) != spec["baseline"]["contract"]
        ):
            raise ValueError("Managed evaluation task, contract, or active round mismatch")
        cfg = config(task)
        device = store.resource(parent["resource"])["spec"]["device"]
        if cfg["candidate_files"] != ["candidate.py"] or (cfg["device"] == "cpu") != (
            device == "cpu"
        ):
            raise ValueError("Managed research requires one candidate.py and a matching device")
        if worker:
            managed_lock_fd()
            job = store.job(os.environ["KERNEL_ENVOLVE_GPU_JOB"])
            if job["parent"] != round_id or job["task"] != task_id or job["status"] != "running":
                raise ValueError("Managed evaluator is not in this round's admitted GPU job")
        return Path(root).resolve(), task_id, round_id, spec
    finally:
        store.close()


def evaluate(task, mode, timeout, directory):
    """Return an independently rechecked receipt after releasing device ownership.

    directory is a new, durable evidence directory outside the task. A POSIX timer
    bounds queue waiting too; a caller already using ITIMER_REAL must use an outer
    process adapter instead. Cleanup can add time after the deadline, as with the
    ordinary evaluator. No timeout is converted into a successful observation.
    """
    number(timeout, "managed evaluation timeout")
    if mode not in {"quick", "full"}:
        raise ValueError("Managed research supports only quick or full")
    task, directory = Path(task).resolve(), Path(directory).resolve()
    root, task_id, round_id, spec = binding(task)
    if directory.is_relative_to(task):
        raise ValueError("Managed evaluation evidence must be outside the task")
    if any(signal.getitimer(signal.ITIMER_REAL)):
        raise ValueError("Managed evaluation cannot replace an existing process deadline")
    directory.mkdir(parents=True, exist_ok=False)
    source_hash = file_hash(task / "candidate.py")
    request = {
        "task": str(task),
        "mode": mode,
        "timeout": timeout,
        "source_hash": source_hash,
        "contract": spec["baseline"]["contract"],
    }
    write_json(directory / "request.json", request)
    source = str(Path(__file__).resolve().parents[2])
    inherited = [
        str(Path(p).resolve()) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    pythonpath = os.pathsep.join(dict.fromkeys([source, *inherited]))
    # Manager changes cwd to the registered workspace. Bind this package for both
    # the admitted helper and its evaluator children; preserve Manager ownership.
    launch = (
        "import os,sys; os.environ['PYTHONPATH']=sys.argv.pop(1); "
        "sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep); "
        "from kernel_envolve.rsi.managed import worker; worker(sys.argv[1])"
    )
    old = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()

    def expired(signum, frame):
        raise _DeadlineExpired("Managed evaluation exhausted its queue/execution deadline")

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        code = gpu_run(
            root,
            task_id,
            round_id,
            [sys.executable, "-c", launch, pythonpath, str(directory)],
            expected_seconds=timeout,
        )
        if code:
            raise ValueError("Managed evaluation did not finish successfully; inspect Manager logs")
        locator = read(directory / "result.json", 8192)
        keys(locator, {"run", "request_hash", "result_hash"})
        if locator["request_hash"] != digest(request):
            raise ValueError("Managed evaluation request changed")
        path = Path(locator["run"])
        if path.parent != task / ".kernel_envolve/runs" or path.is_symlink():
            raise ValueError("Managed evaluation result escapes the current task")
        verified = inspect_run(path, spec)
        result = read_json(path / "result.json")
        if (
            verified["mode"] != mode
            or verified["result_hash"] != locator["result_hash"]
            or digest(result) != locator["result_hash"]
            or file_hash(path / "source/candidate.py") != source_hash
        ):
            raise ValueError("Managed evaluation source, mode, or receipt mismatch")
        return path, result
    except _DeadlineExpired as exc:
        # Translate only after GPU admission/supervision has unwound and cleaned
        # up. TimeoutError itself is an OSError and can be retried by device probes.
        raise TimeoutError(str(exc)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        write_json(
            directory / "cost.json", {"queue_and_execution_seconds": time.monotonic() - started}
        )


def worker(directory):
    from ..cli import evaluate as evaluate_direct

    directory = Path(directory).resolve()
    request = read(directory / "request.json", 8192)
    keys(request, {"task", "mode", "timeout", "source_hash", "contract"})
    task = Path(request["task"]).resolve()
    _, _, _, spec = binding(task, worker=True)
    if (
        directory.is_relative_to(task)
        or request["contract"] != spec["baseline"]["contract"]
        or file_hash(task / "candidate.py") != request["source_hash"]
    ):
        raise ValueError("Managed evaluation changed before device admission")
    if request["mode"] not in {"quick", "full"}:
        raise ValueError("Invalid managed evaluation mode")
    number(request["timeout"], "managed evaluation timeout")
    path, result = evaluate_direct(task, request["mode"], timeout=request["timeout"])
    write_json(
        directory / "result.json",
        {"run": str(path), "request_hash": digest(request), "result_hash": digest(result)},
    )


if __name__ == "__main__":
    worker(sys.argv[1])
