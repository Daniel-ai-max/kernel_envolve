"""One-use independent confirmation of a frozen native full evaluation."""

import json
import math
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from .execution import lock
from .native_config import confirmation_config, load_config
from .storage import digest, evaluator_hash, file_hash, read_json, write_json


def _within(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in (path, *path.parents) if p != root.parent
    ):
        raise ValueError("Confirmation evidence path escapes its root")
    return path


def _audit(run, checkpoint=None, *, deep=True):
    from .native import verify_roots

    checkpoint = checkpoint or (lambda: None)
    if run.is_symlink():
        raise ValueError("Native run must not be a symlink")
    result, cfg = read_json(run / "result.json"), read_json(run / "config.json")
    receipt = read_json(run / "receipt.json")
    if (
        digest(result) != receipt["result_hash"]
        or file_hash(run / "config.json") != receipt["config_hash"]
    ):
        raise ValueError("Native result/config receipt changed")
    if not deep:
        return result, cfg
    for name, expected in receipt["evidence"].items():
        checkpoint()
        if file_hash(_within(run, name)) != expected:
            raise ValueError("Native evidence changed")
    for role, identity in result["identities"].items():
        verify_roots(run, identity, role, checkpoint)
    if file_hash(run / "adapter.py") != result["adapter_hash"]:
        raise ValueError("Native adapter changed")
    return result, cfg


def verify_live_sources(result, cfg):
    """Reject drift in every live input root; read hashes without copying trees."""
    from .native import snapshot, source_contents

    for role, roots in cfg["sources"].items():
        actual = {
            name: snapshot(path, None, cfg["artifacts"].get(name, []))
            for name, path in roots.items()
        }
        if source_contents(actual) != source_contents(result["identities"][role]):
            raise ValueError(f"Live confirmation source changed: {role}")


def _binding(run, result):
    return {
        "parent_run": str(run),
        "parent_result_hash": digest(result),
        "parent_receipt_hash": file_hash(run / "receipt.json"),
        "parent_config_hash": file_hash(run / "config.json"),
    }


def verify_confirmation(run, result, cfg, checkpoint=None, *, deep=True):
    """Verify frozen provenance; deep=False loads metadata for an existing best.

    Promotion, publication and manager evidence audits retain the default deep
    verification. Loading best already verifies the source actually executed.
    """
    if "confirmation" not in cfg:
        return None
    from .native import native_decision, source_contents

    run = Path(run)
    folder = _within(run, "confirmation")
    receipt_path = folder / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Independent confirmation receipt is required")
    receipt = read_json(receipt_path)
    binding = _binding(run, result)
    if receipt.get("binding") != binding or receipt.get("status") != "passed":
        raise ValueError("Independent confirmation failed or belongs to another full run")
    state = Path(result["task_root"]) / ".kernel_native"
    attempt = read_json(state / "confirmation-attempt.json")
    if attempt != read_json(folder / "attempt.json") or attempt.get("binding") != binding:
        raise ValueError("Confirmation consumption record changed")
    if digest(attempt) != receipt.get("attempt_hash") or attempt.get("confirmation_hash") != digest(
        cfg["confirmation"]
    ):
        raise ValueError("Confirmation attempt receipt changed")
    task = _within(folder, "task")
    expected = confirmation_config(cfg)
    if load_config(task) != expected or file_hash(task / "adapter.py") != result["adapter_hash"]:
        raise ValueError("Confirmation task/adapter changed")
    child = _within(task / ".kernel_native" / "runs", receipt["run"])
    held, held_cfg = _audit(child, checkpoint, deep=deep)
    if (
        held_cfg != expected
        or Path(held["task_root"]) != task
        or digest(held) != receipt.get("result_hash")
        or file_hash(child / "receipt.json") != receipt.get("receipt_hash")
        or held["evaluator_hash"] != result["evaluator_hash"]
        or held["adapter_hash"] != result["adapter_hash"]
        or held["initial_best_hash"] is not None
        or held["mode"] != "full"
    ):
        raise ValueError("Confirmation provenance changed")
    if any(
        environment != result.get("environments", {}).get(role)
        for role, environment in held["environments"].items()
    ):
        raise ValueError("Confirmation runtime environments differ from the original full run")
    if {role: source_contents(identity) for role, identity in held["identities"].items()} != {
        role: source_contents(identity) for role, identity in result["identities"].items()
    }:
        raise ValueError("Confirmation sources differ from the original full run")
    if not native_decision(held, held_cfg)["eligible_for_best"]:
        raise ValueError("Independent confirmation did not pass full acceptance")
    return {
        "receipt_hash": file_hash(receipt_path),
        "parent_result_hash": digest(result),
        "run": str(child),
        "result_hash": digest(held),
    }


@contextmanager
def _phase(phases, name):
    started = time.monotonic()
    try:
        yield
    finally:
        phases[name] = time.monotonic() - started


def confirm_native(run, timeout=1800):
    """Consume this task's confirmation set once, including failed/interrupted attempts."""
    from .native import _best, evaluate_native, native_decision

    started = time.monotonic()
    phases = {}
    run = Path(run).resolve()
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid confirmation timeout")
    result = read_json(run / "result.json")
    task = Path(result["task_root"])
    state = task / ".kernel_native"
    if not run.is_relative_to((state / "runs").resolve()):
        raise ValueError("Confirmation parent is outside task runs")
    with lock(state / "promote.lock"):
        phases["lock_wait"] = time.monotonic() - started
        with _phase(phases, "parent_audit"):
            result, cfg = _audit(run)
            if "confirmation" not in cfg:
                raise ValueError("Task has no frozen confirmation contract")
            if not native_decision(result, cfg)["eligible_for_best"]:
                raise ValueError("Confirmation requires an eligible full evaluation")
            if evaluator_hash() != result["evaluator_hash"]:
                raise ValueError("Evaluator changed before confirmation")
            if load_config(task) != cfg or file_hash(task / "adapter.py") != result["adapter_hash"]:
                raise ValueError("Live task changed before confirmation")
            best = _best(state)
            if (digest(best[0]) if best else None) != result["initial_best_hash"]:
                raise ValueError("Best changed before confirmation")
        with _phase(phases, "live_sources_before"):
            verify_live_sources(result, cfg)
        binding = _binding(run, result)
        attempt = {
            "binding": binding,
            "started_at": time.time(),
            "confirmation_hash": digest(cfg["confirmation"]),
        }
        # Task-level consumption prevents re-running full or changing the candidate
        # to silently reuse inputs already seen in a previous confirmation.
        claim = state / "confirmation-attempt.json"
        try:
            with claim.open("x") as stream:
                json.dump(attempt, stream, sort_keys=True)
        except FileExistsError as exc:
            raise ValueError(
                "Confirmation set already consumed; create a new task contract"
            ) from exc
        folder = run / "confirmation"
        folder.mkdir()
        write_json(folder / "attempt.json", attempt)
        receipt = {"binding": binding, "attempt_hash": digest(attempt), "status": "failed"}
        try:
            derived = folder / "task"
            derived.mkdir()
            shutil.copy2(run / "adapter.py", derived / "adapter.py")
            write_json(derived / "native.json", confirmation_config(cfg))
            with _phase(phases, "child_full"):
                child, held = evaluate_native(derived, "full", timeout)
            receipt.update(
                run=child.name,
                result_hash=digest(held),
                receipt_hash=file_hash(child / "receipt.json"),
                decision=held["decision"],
                execution=held["execution"],
                correctness=held["correctness"],
            )
            # Child evaluation snapshots live roots; require those bytes to match
            # the parent before accepting its independent measurements.
            from .native import source_contents

            if {r: source_contents(i) for r, i in held["identities"].items()} != {
                r: source_contents(result["identities"][r]) for r in held["identities"]
            }:
                raise ValueError("Confirmation sources changed during snapshot")
            with _phase(phases, "live_sources_after"):
                verify_live_sources(result, cfg)
            if native_decision(held, load_config(derived))["eligible_for_best"]:
                receipt["status"] = "passed"
                write_json(folder / "receipt.json", receipt)
                with _phase(phases, "posterior_audit"):
                    verify_confirmation(run, result, cfg)
        except BaseException as exc:
            receipt["status"] = "failed"
            receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            receipt["seconds"] = time.monotonic() - started
            receipt["phase_wall_seconds"] = phases
            write_json(folder / "receipt.json", receipt)
        return receipt
