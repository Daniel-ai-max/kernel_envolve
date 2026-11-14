"""Native project evaluation with frozen sources, isolated workers and native gates."""

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .execution import lock, spawn, stop
from .native_checks import junit as junit  # Preserve the existing native helper import.
from .native_checks import native_checks
from .native_config import environment, load_config
from .native_harness import prepare_harness, validate_overlay
from .protocol import objective_cases, order, plan, post_seed, timing_seed
from .scoring import score
from .storage import digest, evaluator_hash, file_hash, read_json, write_json


def snapshot(root, dest, artifacts=(), checkpoint=None):
    checkpoint = checkpoint or (lambda: None)
    checkpoint()
    root = Path(root)
    resolved_root = root.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=checkpoint()
    ).strip()
    listed = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            timeout=checkpoint(),
        )
        .decode()
        .split("\0")
    )
    names = set(
        n
        for n in listed
        if n
        and not any(
            part in {".kernel_native", ".kernel_envolve", "__pycache__"} for part in Path(n).parts
        )
    )
    for pattern in artifacts:
        checkpoint()
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            raise ValueError(f"Required compiled artifact missing: {pattern}")
        names.update(str(p.relative_to(root)) for p in matches)
    if dest is not None:
        dest.mkdir(parents=True)
    directories = {dest}
    files, total = {}, 0
    for name in sorted(names):
        checkpoint()
        src = root / name
        if src.is_dir():  # Gitlinks are supplied as separate named roots when needed.
            continue
        if not src.is_file():
            continue  # A tracked local deletion is part of this source revision.
        if src.is_symlink() or not src.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Source symlink/outside-root file: {src}")
        if src.suffix in {".pyc", ".pyo"}:
            continue
        total += src.stat().st_size
        if total > 512 * 1024 * 1024:
            raise ValueError("Source snapshot exceeds 512 MiB")
        if dest is None:
            files[name] = file_hash(src)
        else:
            dst = dest / name
            if dst.parent not in directories:
                dst.parent.mkdir(parents=True, exist_ok=True)
                directories.add(dst.parent)
            shutil.copy2(src, dst)
            files[name] = file_hash(dst)
    if not files:
        raise ValueError("Empty source snapshot")
    checkpoint()
    return {"git_head": head, "files": files}


def verify_roots(run, identities, role, checkpoint=None):
    checkpoint = checkpoint or (lambda: None)
    checkpoint()
    for root, identity in identities.items():
        base = run / "sources" / role / root
        if base.is_symlink():
            raise ValueError("Symlink introduced in snapshot")
        expected, seen = identity["files"], set()
        pending = [(base, "")]
        while pending:
            checkpoint()
            directory, prefix = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    checkpoint()
                    name = prefix + entry.name
                    if entry.is_symlink():
                        raise ValueError("Symlink introduced in snapshot")
                    if entry.is_dir(follow_symlinks=False):
                        if name in expected:
                            raise ValueError(f"Frozen source changed: {role}/{root}/{name}")
                        pending.append((Path(entry.path), name + "/"))
                    elif name in expected:
                        if (
                            not entry.is_file(follow_symlinks=False)
                            or file_hash(Path(entry.path)) != expected[name]
                        ):
                            raise ValueError(f"Frozen source changed: {role}/{root}/{name}")
                        seen.add(name)
                    # Report outputs are permitted; new source or bytecode is not.
                    elif entry.is_file(follow_symlinks=False) and Path(name).suffix in {
                        ".py",
                        ".so",
                        ".cu",
                        ".cuh",
                        ".cpp",
                        ".pyc",
                        ".pyo",
                    }:
                        raise ValueError(f"Unrecorded executable source: {name}")
        checkpoint()
        missing = expected.keys() - seen
        if missing:
            raise ValueError(f"Frozen source changed: {role}/{root}/{min(missing)}")


def source_contents(identities):
    """Compare actual bundled contents; an empty Git commit is not a new kernel."""
    return {name: identity["files"] for name, identity in identities.items()}


class Worker:
    def __init__(self, run, cfg, roots, role, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("Native evaluation time budget exhausted")
        self.folder = run / "workers" / role
        self.folder.mkdir(parents=True)
        self.deadline = deadline
        self.count = 0
        self.log = (self.folder / "worker.log").open("w")
        request = self.folder / "request.json"
        try:
            write_json(
                request,
                {"config": cfg, "roots": roots, "role": role, "adapter": str(run / "adapter.py")},
            )
            self.process = spawn(
                [sys.executable, "-m", "kernel_envolve.native_worker", str(request)],
                stdin=subprocess.PIPE,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=run,
                env=environment(cfg, roots),
                start_new_session=True,
            )
        except BaseException:
            self.log.close()
            raise

    def call(self, kind, case, seed):
        import torch

        if time.monotonic() >= self.deadline:
            raise TimeoutError("Native evaluation time budget exhausted")
        self.count += 1
        base = self.folder / str(self.count)
        reply, output = base.with_suffix(".json"), base.with_suffix(".pt")
        command = {
            "id": self.count,
            "kind": kind,
            "case": case,
            "seed": seed,
            "reply": str(reply),
            "output": str(output),
        }
        write_json(base.with_suffix(".command.json"), command)
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        while not reply.exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.folder.name} worker exited; see {self.folder}/worker.log"
                )
            if time.monotonic() >= self.deadline:
                raise TimeoutError("Native evaluation time budget exhausted")
            time.sleep(0.02)
        result = read_json(reply)
        if result.get("command_id") != self.count or result.get("status") != "pass":
            raise RuntimeError(f"{self.folder.name}: {result.get('error', 'invalid response')}")
        observations = torch.load(output, map_location="cpu", weights_only=True)
        output.unlink()
        return result, observations

    def close(self):
        errors = []
        for cleanup in (lambda: stop(self.process), self.process.stdin.close, self.log.close):
            try:
                cleanup()
            except Exception as exc:
                errors.append(exc)
        if errors:
            for other in errors[1:]:
                errors[0].add_note(f"Additional cleanup error: {type(other).__name__}: {other}")
            raise errors[0]


def compare(actual, expected, cfg):
    from .arrays import assert_close
    from .training import equal

    for path in cfg.get("required_outputs", []):
        for observation in (actual, expected):
            value = observation["output"]
            try:
                for key in path.split("."):
                    value = value[int(key)] if isinstance(value, (tuple, list)) else value[key]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise AssertionError(f"Missing required output/gradient: {path}") from exc
    for key in ("input_metadata", "input_aliases", "mutable", "output_aliases"):
        if actual[key] != expected[key]:
            raise AssertionError(f"Contract mismatch: {key}")
    if actual["inputs"].keys() != expected["inputs"].keys() or any(
        not equal(actual["inputs"][name], tensor) for name, tensor in expected["inputs"].items()
    ):
        raise AssertionError("Input values differ (including signed zero and NaN payloads)")

    def check_meta(a, b):
        if isinstance(b, dict) and isinstance(b.get("dtype"), str):
            keys = ("shape", "dtype", "device")
            if cfg["check_output_layout"]:
                keys += ("stride", "offset", "conjugate", "negative")
            if not isinstance(a, dict) or any(a.get(k) != b[k] for k in keys):
                raise AssertionError("Output shape/dtype/device/layout differs")
        elif type(a) is not type(b) or len(a) != len(b):
            raise AssertionError("Output structure differs")
        elif isinstance(b, dict):
            if a.keys() != b.keys():
                raise AssertionError("Output keys differ")
            for k in b:
                check_meta(a[k], b[k])
        else:
            for x, y in zip(a, b):
                check_meta(x, y)

    def check_bits(a, b):
        if isinstance(b, dict):
            return all(check_bits(a[k], b[k]) for k in b)
        if isinstance(b, (tuple, list)):
            return all(check_bits(x, y) for x, y in zip(a, b))
        return equal(a, b)

    check_meta(actual["output_metadata"], expected["output_metadata"])
    for key in ("output", "state"):
        if key == "state" and not expected[key] and not actual[key]:
            continue
        assert_close(actual[key], expected[key], cfg, key)
        if cfg["bitwise"] and not check_bits(actual[key], expected[key]):
            raise AssertionError(f"Bitwise mismatch: {key}")


def _roots(run, role, identities):
    return {name: str((run / "sources" / role / name).resolve()) for name in identities}


def _best(state):
    if not (state / "best.json").exists():
        return None
    pointer = read_json(state / "best.json")
    run = (state / pointer["run"]).resolve()
    if not run.is_relative_to((state / "runs").resolve()):
        raise ValueError("Best is outside task runs")
    result = read_json(run / "result.json")
    if digest(result) != pointer["result_hash"]:
        raise ValueError("Best result changed")
    verify_roots(run, result["identities"]["candidate"], "candidate")
    from .native_confirmation import verify_confirmation

    cfg = read_json(run / "config.json")
    if pointer.get("confirmation") is not None or "confirmation" in cfg:
        receipt = read_json(run / "receipt.json")
        if (
            receipt["result_hash"] != digest(result)
            or receipt["config_hash"] != file_hash(run / "config.json")
            or "confirmation" not in cfg
        ):
            raise ValueError("Best confirmation config/receipt changed")
        confirmation = verify_confirmation(run, result, cfg, deep=False)
        if pointer.get("confirmation") != confirmation:
            raise ValueError("Best confirmation receipt changed")
    return pointer, run, result


def evaluate_native(task, mode="quick", timeout=None, *, cases=None, seeds=None, check_only=False):
    task = Path(task).resolve()
    if timeout is None:
        timeout = 120 if mode == "quick" else 1800
    if (
        mode not in {"quick", "full"}
        or not math.isfinite(timeout)
        or timeout <= 0
        or sys.flags.optimize
    ):
        raise ValueError("Invalid mode/timeout or optimized Python")
    cfg = load_config(task)
    selection = {}
    if cases is not None:
        if not cases:
            raise ValueError("Selected cases must be nonempty")
        selection["cases"] = cases
    if seeds is not None:
        selection["seeds"] = seeds
    selected, _, _, _ = plan(cfg, mode, selection)
    if seeds is not None and set(seeds) & set(cfg.get("confirmation", {}).get("seeds", [])):
        raise ValueError("Diagnostic seeds must not use reserved confirmation seeds")
    if not selected:
        raise ValueError("No cases selected")
    if not check_only and not objective_cases(selected):
        raise ValueError("No measured objective cases selected; use --check-only")
    state = task / ".kernel_native"
    (state / "runs").mkdir(parents=True, exist_ok=True)
    with lock(state / "evaluate.lock"):
        return _evaluate(task, state, cfg, mode, timeout, selection, check_only)


def _evaluate(task, state, cfg, mode, timeout, selection, check_only):
    run = state / "runs" / uuid.uuid4().hex
    run.mkdir()
    deadline = time.monotonic() + timeout
    start = time.perf_counter()
    shutil.copy2(task / "adapter.py", run / "adapter.py")
    write_json(run / "config.json", cfg)
    result = {
        "schema": 1,
        "task_root": str(task),
        "mode": mode,
        "execution": "error",
        "correctness": "partial",
        "signal_only": check_only,
        "check_only": check_only,
        "selection": selection,
        "checks": [],
        "samples": [],
        "native_checks": [],
        "identities": {},
        "evaluator_hash": evaluator_hash(),
        "adapter_hash": file_hash(run / "adapter.py"),
        "memory": [],
        "environments": {},
        "phase_wall_seconds": {},
    }
    phase, phase_started = "snapshot", start

    def begin_phase(name):
        nonlocal phase, phase_started
        now = time.perf_counter()
        result["phase_wall_seconds"][phase] = now - phase_started
        phase, phase_started = name, now
        return now

    workers = {}

    def checkpoint():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Native evaluation time budget exhausted")
        return remaining

    try:
        for role, roots in cfg["sources"].items():
            result["identities"][role] = {
                name: snapshot(
                    path, run / "sources" / role / name, cfg["artifacts"].get(name, []), checkpoint
                )
                for name, path in roots.items()
            }
        begin_phase("setup")
        contract_cfg = {k: v for k, v in cfg.items() if k != "sources"}
        result["contract"] = digest(
            {
                "config": contract_cfg,
                "adapter": result["adapter_hash"],
                "baseline": result["identities"]["baseline"],
            }
        )
        validate_overlay(cfg, result["identities"])
        best = _best(state)
        result["initial_best_hash"] = digest(best[0]) if best else None
        if best and best[2]["contract"] != result["contract"]:
            raise ValueError("Native task contract or baseline changed; use a new task")
        roots = _roots(run, "candidate", result["identities"]["candidate"])
        result["comparator"] = {"kind": "best" if best else "baseline"}
        result["same_candidate_as_best"] = bool(
            best
            and source_contents(result["identities"]["candidate"])
            == source_contents(best[2]["identities"]["candidate"])
        )
        role_roots = {
            "oracle": _roots(run, "baseline", result["identities"]["baseline"]),
            "candidate": roots,
        }
        if not check_only:
            role_roots["comparator"] = (
                _roots(best[1], "candidate", best[2]["identities"]["candidate"])
                if best
                else role_roots["oracle"]
            )
        if best and mode == "full" and not check_only:
            role_roots["baseline"] = role_roots["oracle"]
        for role, sources in role_roots.items():
            workers[role] = Worker(run, cfg, sources, role, deadline)
        cases, seeds, trials, batches = plan(cfg, mode, selection)
        roles = [r for r in workers if r != "oracle"]

        def call(role, kind, case, seed):
            meta, observed = workers[role].call(kind, case, seed)
            env = {k: meta[k] for k in ("torch_version", "cuda_version", "gpu", "runtime_settings")}
            if result["environments"] and env != next(iter(result["environments"].values())):
                raise ValueError("Role runtime environments differ")
            result["environments"][role] = env
            return meta, observed

        def check(case, seed, kinds):
            _, expected = call("oracle", "check", case, seed)
            for role in roles:
                _, observed = call(role, "check", case, seed)
                if len(observed) != len(expected):
                    raise AssertionError("Incomplete correctness observations")
                for index, kind in enumerate(kinds):
                    compare(observed[index], expected[index], cfg)
                    if kind == "repeat":
                        compare(
                            observed[index],
                            observed[index - 1],
                            cfg | {"rtol": 0, "atol": 0, "bitwise": True},
                        )
                    result["checks"].append(
                        {
                            "case": case["id"],
                            "seed": seed,
                            "kind": kind,
                            "role": role,
                            "status": "pass",
                        }
                    )

        begin_phase("correctness")
        kinds = ["fresh", "reuse", "repeat"] if cfg["deterministic"] else ["fresh", "reuse"]
        for case in cases:
            for seed in seeds:
                check(case, seed, kinds)
        if mode == "full":
            begin_phase("native_checks")
            validation, seconds = prepare_harness(run, cfg, result["identities"], checkpoint)
            result["identities"]["validation"] = validation
            result["harness_seconds"] = seconds
            validation_roots = _roots(run, "validation", validation)
            native_checks(run, cfg, validation_roots, deadline, result["native_checks"])
            scopes = {r["scope"] for r in result["native_checks"]}
            if set(cfg["required_scopes"]) - scopes:
                raise ValueError("Missing required native validation scopes")
        begin_phase("measurement")
        timed = [] if check_only else [c for c in cases if c.get("measure", True)]
        for batch in range(batches):
            for case in timed if batch == 0 else reversed(timed):
                for trial in range(trials):
                    expected_cache, repeats = {}, dict.fromkeys(roles, 0)
                    for role in order(roles, batch, trial):
                        repeat = repeats[role]
                        seed = timing_seed(case["id"], batch, trial, repeat, seeds)
                        if seed not in expected_cache:
                            _, oracle = call("oracle", "timing_oracle", case, seed)
                            if len(oracle) != 1:
                                raise AssertionError("Expected one timing oracle observation")
                            expected_cache[seed] = oracle[0]
                        meta, observed = call(role, "measure", case, seed)
                        compare(observed[0], expected_cache[seed], cfg)
                        result["samples"].append(
                            {
                                "case": case["id"],
                                "seed": seed,
                                "batch": batch,
                                "trial": trial,
                                "role": role,
                                "repeat": repeat,
                                "us": statistics.median(meta["samples_us"]),
                                "raw_us": meta["samples_us"],
                            }
                        )
                        result["memory"].append(
                            {"case": case["id"], "role": role, "peak_bytes": meta["peak_bytes"]}
                        )
                        if batch == batches - 1 and trial == trials - 1 and repeat == 1:
                            result["checks"].append(
                                {
                                    "case": case["id"],
                                    "seed": seed,
                                    "kind": "timed_output",
                                    "role": role,
                                    "status": "pass",
                                }
                            )
                        repeats[role] += 1
        begin_phase("postcheck")
        for case in timed:
            check(case, post_seed(case["id"], seeds), ["post"])
        begin_phase("integrity")
        for role, identity in result["identities"].items():
            verify_roots(run, identity, role, checkpoint)
        if best:
            verify_roots(best[1], best[2]["identities"]["candidate"], "candidate", checkpoint)
        if (
            evaluator_hash() != result["evaluator_hash"]
            or file_hash(run / "adapter.py") != result["adapter_hash"]
        ):
            raise ValueError("Evaluator/adapter changed during evaluation")
        result.update(execution="completed", correctness="pass")
    except Exception as exc:
        result["failed_phase"] = phase
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, AssertionError):
            result.update(execution="completed", correctness="fail")
    finally:
        begin_phase("cleanup")
        cleanup_errors = []
        for role, worker in workers.items():
            try:
                worker.close()
            except Exception as exc:
                cleanup_errors.append(
                    {
                        "role": role,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "notes": getattr(exc, "__notes__", []),
                    }
                )
        if cleanup_errors:
            if "error" in result:
                result["prior_error"] = result["error"]
                result["prior_failed_phase"] = result["failed_phase"]
            result.update(
                execution="error",
                correctness="partial",
                failed_phase="cleanup",
                cleanup_errors=cleanup_errors,
                error={"type": "WorkerCleanupError", "message": "Native worker cleanup failed"},
            )
    result["seconds"] = begin_phase(None) - start
    try:
        result["decision"] = native_decision(result, cfg)
    except (ValueError, KeyError, TypeError) as exc:
        result.update(execution="error", correctness="partial")
        result["error"] = {"type": "EvidenceError", "message": str(exc)}
        result["decision"] = native_decision(result, cfg)
    write_json(run / "result.json", result)
    write_json(
        run / "receipt.json",
        {
            "result_hash": digest(result),
            "config_hash": file_hash(run / "config.json"),
            "evidence": {
                str(p.relative_to(run)): file_hash(p)
                for folder in (run / "native_checks", run / "workers")
                if folder.exists()
                for p in folder.rglob("*")
                if p.is_file() and p.suffix != ".pt"
            },
        },
    )
    return run, result


def native_decision(result, cfg):
    decision = score(result, cfg)
    identities = result["identities"]
    if {"candidate", "baseline"} <= identities.keys() and source_contents(
        identities["candidate"]
    ) == source_contents(identities["baseline"]):
        decision["reasons"].append("Candidate source is identical to fixed baseline")
    if result["mode"] == "full":
        scopes = {s["scope"] for s in result["native_checks"] if s["status"] == "pass"}
        if set(cfg["required_scopes"]) - scopes:
            decision["reasons"].append("Required native checks are incomplete")
        for case in cfg["cases"]:
            if "max_peak_bytes" in case and any(
                r["peak_bytes"] > case["max_peak_bytes"]
                for r in result["memory"]
                if r["case"] == case["id"] and r["role"] == "candidate"
            ):
                decision["reasons"].append(f"{case['id']}: peak memory limit exceeded")
    decision["eligible_for_best"] = decision["eligible_for_best"] and not decision["reasons"]
    return decision


def promote_native(run):
    run = Path(run).resolve()
    result, cfg = read_json(run / "result.json"), read_json(run / "config.json")
    task = Path(result["task_root"])
    state = task / ".kernel_native"
    if not run.is_relative_to((state / "runs").resolve()):
        raise ValueError("Run is outside task")
    with lock(state / "promote.lock"):
        receipt = read_json(run / "receipt.json")
        if (
            digest(result) != receipt["result_hash"]
            or file_hash(run / "config.json") != receipt["config_hash"]
        ):
            raise ValueError("Result/config changed")
        for name, expected in receipt["evidence"].items():
            if file_hash(run / name) != expected:
                raise ValueError("Native evidence changed")
        if (
            evaluator_hash() != result["evaluator_hash"]
            or file_hash(run / "adapter.py") != result["adapter_hash"]
        ):
            raise ValueError("Evaluator/adapter changed")
        if load_config(task) != cfg or file_hash(task / "adapter.py") != result["adapter_hash"]:
            raise ValueError("Live task changed")
        for role, identity in result["identities"].items():
            verify_roots(run, identity, role)
        best = _best(state)
        if (digest(best[0]) if best else None) != result["initial_best_hash"]:
            raise ValueError("Best changed; reevaluate")
        decision = native_decision(result, cfg)
        if not decision["eligible_for_best"]:
            raise ValueError("Not eligible: " + "; ".join(decision["reasons"]))
        from .native_confirmation import verify_confirmation, verify_live_sources

        confirmation = verify_confirmation(run, result, cfg)
        if confirmation is not None:
            verify_live_sources(result, cfg)
        pointer = {
            "run": str(run.relative_to(state)),
            "result_hash": digest(result),
            "previous": best[0]["run"] if best else None,
        }
        if confirmation is not None:
            pointer["confirmation"] = confirmation
        write_json(
            run / "promotion.json",
            {
                "pointer": pointer,
                "previous_pointer": best[0] if best else None,
                "at": time.time(),
            },
        )
        write_json(state / "best.json", pointer)
        return pointer
