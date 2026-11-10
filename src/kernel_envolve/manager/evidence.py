"""Read-only projection of evaluator receipts. Agent prose never grants a speedup."""

import json
import math
import os
import stat
import statistics
import time
from pathlib import Path

from ..storage import config, digest, file_hash, read_json, verify_source
from .process import exclusive
from .store import encode


def state_path(spec):
    return Path(spec["task_dir"]) / (
        ".kernel_native" if spec["kind"] == "native" else ".kernel_envolve"
    )


def within(root, relative):
    original = root / relative
    path = original.resolve()
    if not path.is_relative_to(root.resolve()) or original.is_symlink():
        raise ValueError("Evidence path escapes its root")
    return path


def inspect_run(path, spec):
    if path.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    result = read_json(path / "result.json")
    receipt = read_json(path / "receipt.json")
    if receipt["result_hash"] != digest(result):
        raise ValueError("Result receipt mismatch")
    if Path(result["task_root"]).resolve() != Path(spec["task_dir"]):
        raise ValueError("Result belongs to a different task")
    if spec["kind"] == "native":
        from ..native import native_decision, verify_roots

        cfg = read_json(path / "config.json")
        if receipt["config_hash"] != file_hash(path / "config.json"):
            raise ValueError("Native config receipt mismatch")
        for name, expected in receipt["evidence"].items():
            if file_hash(within(path, name)) != expected:
                raise ValueError(f"Native evidence changed: {name}")
        for role, value in result["identities"].items():
            verify_roots(path, value, role)
        if file_hash(path / "adapter.py") != result["adapter_hash"]:
            raise ValueError("Native adapter changed")
        decision = native_decision(result, cfg)
        contract = result.get("contract")
        source = result["identities"].get("candidate")
    else:
        from ..cli import audit_result

        cfg = config(path / "source")
        verify_source(path, result["identity"])
        decision = audit_result(result, read_json(path / "request.json"), cfg)
        contract = result["identity"]["contract"]
        source = result["identity"]["files"]
    if contract != spec["baseline"]["contract"]:
        raise ValueError("Evidence contract differs from the task's frozen baseline contract")
    fixed = decision.get("fixed_baseline_gain_by_batch", [])
    incremental = decision.get("gain_by_batch", [])
    return {
        "run": str(path),
        "result_hash": receipt["result_hash"],
        "initial_best_hash": result["initial_best_hash"],
        "mode": result["mode"],
        "execution": result["execution"],
        "correctness": result["correctness"],
        "decision": decision,
        "contract": contract,
        "source": source,
        "environment": result.get("environments", result.get("environment")),
        "native_checks": result.get("native_checks", []),
        "memory": result.get("memory", []),
        "error": result.get("error"),
        "recorded_at": (path / "receipt.json").stat().st_mtime,
        "passed_checks": sum(c.get("status") == "pass" for c in result.get("checks", [])),
        "verified_at": time.time(),
        "baseline_speedup": 1 / (1 - statistics.median(fixed)) if fixed else None,
        "incremental_speedup": 1 / (1 - statistics.median(incremental)) if incremental else None,
        "evaluator_hash": result.get(
            "evaluator_hash", result.get("identity", {}).get("evaluator_hash")
        ),
    }


def fingerprint(path):
    """Invalidate cached verification with one metadata read per non-cache entry.

    Metadata only decides whether to repeat inspect_run's content hashes. ctime
    and inode identity also catch equal-sized writes/replacements that preserve
    mtime. Never follow links, including links named like an ignored cache.
    """
    path = Path(path)
    root = path.lstat()
    if not stat.S_ISDIR(root.st_mode):
        raise ValueError("Evidence root must be a real directory")
    pending = [(os.fspath(path), "")]
    files = []
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                name = prefix + entry.name
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(f"Symlink introduced in evidence: {name}")
                if entry.name == "__pycache__":
                    continue
                if stat.S_ISDIR(info.st_mode):
                    pending.append((entry.path, name + os.sep))
                elif stat.S_ISREG(info.st_mode):
                    files.append(
                        (
                            name,
                            info.st_size,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                            info.st_dev,
                            info.st_ino,
                            info.st_mode,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported evidence file type: {name}")
    return digest(sorted(files))


def refresh_task(store, task, force=False):
    spec = task["spec"]
    state = state_path(spec)
    problems = []
    best_projection = None
    started = time.time()
    indexed_runs = {
        r["path"]: r for r in store.rows("SELECT * FROM runs WHERE task=?", (task["id"],))
    }
    updates = {}
    chain = []
    for path in sorted((state / "runs").glob("*")):
        old = indexed_runs.get(str(path))
        if not (path / "receipt.json").exists() and not old:
            continue  # New in-flight results are not final evidence; indexed ones must revalidate.
        try:
            signature = fingerprint(path)
            if (
                old
                and not force
                and "recorded_at" in json.loads(old["data"])
                and json.loads(old["data"]).get("fingerprint") == signature
            ):
                if old["error"]:
                    problems.append(f"{path.name}: {old['error']}")
                continue
            data = inspect_run(path, spec) | {"fingerprint": signature}
            error = None
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            error = str(exc)
            # Retain observation order even if a previously verified receipt disappears.
            recorded_at = json.loads(old["data"]).get("recorded_at", 0) if old else 0
            try:
                recorded_at = (path / "receipt.json").stat().st_mtime
            except OSError:
                pass
            data = {"run": str(path), "recorded_at": recorded_at}
            problems.append(f"{path.name}: {error}")
        updates[str(path)] = {"data": encode(data), "error": error}
        indexed_runs[str(path)] = updates[str(path)]
    # Removed evidence must invalidate the entire published best in the same commit.
    for path, row in indexed_runs.items():
        if not Path(path).exists():
            updates[path] = {"data": row["data"], "error": "Evidence directory missing"}
            indexed_runs[path] = updates[path]
            problems.append(f"{path}: missing")
    try:
        if (state / "best.json").exists():
            pointer = read_json(state / "best.json")
            pointer_hash = digest(pointer)
            visited = set()
            chain = []
            while pointer:
                run = within(state / "runs", str(Path(pointer["run"]).relative_to("runs")))
                if str(run) in visited:
                    raise ValueError("Cycle in promotion history")
                visited.add(str(run))
                indexed = indexed_runs.get(str(run))
                if not indexed or indexed["error"]:
                    raise ValueError(f"Best has missing or invalid evidence: {run.name}")
                data = json.loads(indexed["data"])
                if (
                    pointer["result_hash"] != data["result_hash"]
                    or not data["decision"]["eligible_for_best"]
                ):
                    raise ValueError("Best does not reference an eligible, verified full result")
                history = (
                    read_json(run / "promotion.json") if (run / "promotion.json").exists() else None
                )
                if history and history["pointer"] != pointer:
                    raise ValueError("Promotion receipt mismatch")
                if history:
                    previous = history["previous_pointer"]
                    if (digest(previous) if previous else None) != data["initial_best_hash"]:
                        raise ValueError("Promotion predecessor differs from evaluated best")
                    if pointer["previous"] != (previous["run"] if previous else None):
                        raise ValueError("Promotion chain link mismatch")
                    if (
                        type(history["at"]) not in (int, float)
                        or not math.isfinite(history["at"])
                        or history["at"] <= 0
                    ):
                        raise ValueError("Invalid promotion timestamp")
                # Legacy promotions have unknown time, not a fabricated historical date.
                data["promoted_at"] = history["at"] if history else None
                chain.append(data)
                pointer = history["previous_pointer"] if history else None
            best_projection = {
                "pointer_hash": pointer_hash,
                "run": chain[0]["run"],
                "verified_at": time.time(),
            }
        elif store.one("SELECT run FROM promotions WHERE task=?", (task["id"],)):
            problems.append("Previously observed best pointer is missing")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(str(exc))
    latest = None
    for row in indexed_runs.values():
        value = json.loads(row["data"])
        if latest is None or value.get("recorded_at", 0) > latest["recorded_at"]:
            latest = {
                **{
                    key: value.get(key)
                    for key in (
                        "run",
                        "mode",
                        "execution",
                        "correctness",
                        "passed_checks",
                        "error",
                    )
                },
                "native_checks": [
                    {key: check.get(key) for key in ("id", "scope", "status")}
                    for check in value.get("native_checks", [])
                ],
                "recorded_at": value.get("recorded_at", 0),
                "evidence_error": row["error"],
            }
    # File IO and verification above never hold SQLite's writer lock. Publish each task's
    # runs, best, history, errors and observation time atomically for dashboard readers.
    with store.transaction():
        for path, row in updates.items():
            store.db.execute(
                "INSERT INTO runs(path,task,imported,data,error) VALUES(?,?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET data=excluded.data,error=excluded.error",
                (path, task["id"], time.time(), row["data"], row["error"]),
            )
        if best_projection:
            for data in reversed(chain):
                store.db.execute(
                    "INSERT OR IGNORE INTO promotions(task,run,at,gpu_seconds,data) VALUES(?,?,?,?,?)",
                    (
                        task["id"],
                        data["run"],
                        data["promoted_at"] or time.time(),
                        task["gpu_seconds"],
                        encode(data),
                    ),
                )
        store.db.execute(
            "UPDATE tasks SET evidence_error=? WHERE id=?",
            ("; ".join(problems) or None, task["id"]),
        )
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            (f"best:{task['id']}", encode(best_projection)),
        )
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            (f"latest:{task['id']}", encode(latest)),
        )
        store.db.execute(
            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
            (
                f"evidence:{task['id']}",
                encode({"observed_at": started, "published_at": time.time()}),
            ),
        )


def refresh(store, force=False):
    # A manual reindex and the coordinator must not publish out of order.
    with exclusive(store.root / "evidence.lock") as handle:
        if handle is None:
            return False
        scan = {"started_at": time.time(), "running": True, "task": None, "error": None}

        def publish():
            scan["updated_at"] = time.time()
            store.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES('evidence_scan',?)", (encode(scan),)
            )

        try:
            for row in store.rows("SELECT id FROM tasks"):
                scan["task"] = row["id"]
                publish()
                refresh_task(store, store.task(row["id"]), force)
        except Exception as exc:
            scan["error"] = str(exc)
            raise
        finally:
            scan.update(running=False, task=None, finished_at=time.time())
            publish()
    return True
