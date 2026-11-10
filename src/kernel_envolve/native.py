"""Native project evaluation with frozen sources, isolated workers and native gates."""

import fnmatch
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from .cli import lock
from .protocol import order, plan, post_seed, timing_seed
from .scoring import score
from .storage import digest, evaluator_hash, file_hash, read_json, write_json


def load_config(task):
    raw = read_json(task / "native.json")
    defaults = {
        "device": "cuda:0",
        "quick_seeds": [0],
        "full_seeds": [0, 17, 101],
        "quick_trials": 3,
        "full_trials": 6,
        "warmup": 3,
        "iterations": 5,
        "rtol": 1e-5,
        "atol": 1e-6,
        "equal_nan": False,
        "bitwise": False,
        "deterministic": True,
        "min_gain": 0.01,
        "noise_multiplier": 2.0,
        "max_regression": 0.05,
        "max_baseline_regression": 0.05,
        "check_output_layout": False,
        "threads": 1,
        "env": {},
        "artifacts": {},
        "checks": [],
        "required_scopes": ["operator", "numerical", "integration"],
        "required_outputs": ["output", "gradients"],
    }
    unknown = set(raw) - (set(defaults) | {"name", "objective", "sources", "imports", "cases"})
    if unknown:
        raise ValueError(f"Unknown native settings: {sorted(unknown)}")
    cfg = defaults | raw
    if not cfg["required_outputs"] or any(
        not isinstance(p, str) or not p for p in cfg["required_outputs"]
    ):
        raise ValueError("Declare required output/gradient paths")
    if not cfg.get("name") or not cfg.get("objective") or not cfg.get("imports"):
        raise ValueError("Declare name, objective, and production package imports")
    if not (task / "adapter.py").is_file():
        raise ValueError("Missing adapter.py")
    if set(cfg["sources"]) != {"candidate", "baseline"}:
        raise ValueError("Provide candidate and fixed baseline sources")
    if (
        not cfg["sources"]["candidate"]
        or cfg["sources"]["candidate"].keys() != cfg["sources"]["baseline"].keys()
    ):
        raise ValueError("Source roles must have the same nonempty named roots")
    for roots in cfg["sources"].values():
        for name, path in roots.items():
            if not name.isidentifier():
                raise ValueError("Source root names must be identifiers")
            expanded = os.path.expandvars(path)
            if "$" in expanded:
                raise ValueError(f"Unresolved source environment variable: {path}")
            roots[name] = str((task / expanded).resolve())
    if set(cfg["imports"].values()) - cfg["sources"]["candidate"].keys():
        raise ValueError("Unknown import root")
    if cfg["device"] != "cpu" and not (
        cfg["device"].startswith("cuda:") and cfg["device"][5:].isdigit()
    ):
        raise ValueError("device must be cpu or cuda:N")
    for key in ("quick_trials", "full_trials", "warmup", "iterations", "threads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if cfg["full_trials"] < 3:
        raise ValueError("full_trials >= 3 required for noise estimation")
    for key in (
        "rtol",
        "atol",
        "min_gain",
        "max_regression",
        "max_baseline_regression",
        "noise_multiplier",
    ):
        if type(cfg[key]) not in (float, int) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("min_gain", "max_regression", "max_baseline_regression"):
        if cfg[key] >= 1:
            raise ValueError(f"Invalid fraction {key}")
    for key in ("deterministic", "equal_nan", "bitwise", "check_output_layout"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"Invalid boolean {key}")
    ids = [c["id"] for c in cfg["cases"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty cases required")
    for case in cfg["cases"]:
        if set(case) - {"id", "args", "weight", "quick", "measure", "max_us", "max_peak_bytes"}:
            raise ValueError("Unknown case field")
        for flag in ("quick", "measure"):
            if flag in case and type(case[flag]) is not bool:
                raise ValueError(f"Invalid case flag: {flag}")
        if "max_peak_bytes" in case and cfg["device"] == "cpu":
            raise ValueError("CUDA peak-memory limits require CUDA")
        if case.get("measure", True):
            if (
                type(case.get("weight")) not in (float, int)
                or not math.isfinite(case["weight"])
                or case["weight"] <= 0
            ):
                raise ValueError("Explicit positive case weights required")
        elif "weight" in case:
            raise ValueError("Correctness-only cases cannot have weights")
        if "max_peak_bytes" in case and (
            type(case["max_peak_bytes"]) is not int or case["max_peak_bytes"] < 1
        ):
            raise ValueError("max_peak_bytes must be positive")
    for mode in ("quick", "full"):
        selected, _, _, _ = plan(cfg, mode)
        if not any(c.get("measure", True) for c in selected):
            raise ValueError("Each mode needs measured cases")
    if not set(cfg["quick_seeds"]).issubset(cfg["full_seeds"]):
        raise ValueError("quick seeds must be included in full seeds")
    if not cfg["required_scopes"] or set(cfg["required_scopes"]) - {
        "operator",
        "numerical",
        "integration",
    }:
        raise ValueError("Declare required native validation scopes")
    if set(cfg["env"]) & {"PYTHONPATH", "PYTHONOPTIMIZE", "PYTHONHOME"}:
        raise ValueError("Python import/optimization overrides are not allowed")
    check_ids = [c["id"] for c in cfg["checks"]]
    if len(check_ids) != len(set(check_ids)) or any(not s.isidentifier() for s in check_ids):
        raise ValueError("Native checks need unique identifier names")
    for check in cfg["checks"]:
        if check.get("scope") not in {"operator", "numerical", "integration"}:
            raise ValueError("Unknown check scope")
        if type(check.get("min_tests", 1)) is not int or check.get("min_tests", 1) < 1:
            raise ValueError("min_tests must be positive")
        if "devices" in check and (
            not check["devices"]
            or len(set(check["devices"])) != len(check["devices"])
            or any(type(d) is not int or d < 0 for d in check["devices"])
        ):
            raise ValueError("Check devices must be unique nonnegative visible indices")
        if check["root"] not in cfg["sources"]["candidate"]:
            raise ValueError("Unknown check root")
        if check["kind"] == "gate":
            if not check.get("operators") or len(set(check["operators"])) != len(
                check["operators"]
            ):
                raise ValueError("Declare exact unique gate operators")
            if check.get("scope") != "numerical":
                raise ValueError("Gate scope must be numerical")
        elif check["kind"] == "pytest":
            if not check.get("required_tests") or not check.get("reports") or not check.get("argv"):
                raise ValueError("pytest requires argv, exact report paths and required_tests")
            for report in check["reports"]:
                if Path(report).is_absolute() or ".." in Path(report).parts:
                    raise ValueError("JUnit paths must be relative to the fresh reports directory")
        else:
            raise ValueError("Native check kind must be gate or pytest")
    return cfg


def snapshot(root, dest, artifacts=()):
    root = Path(root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    listed = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
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
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            raise ValueError(f"Required compiled artifact missing: {pattern}")
        names.update(str(p.relative_to(root)) for p in matches)
    dest.mkdir(parents=True)
    files, total = {}, 0
    for name in sorted(names):
        src = root / name
        if src.is_dir():  # Gitlinks are supplied as separate named roots when needed.
            continue
        if not src.is_file():
            continue  # A tracked local deletion is part of this source revision.
        if src.is_symlink() or not src.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source symlink/outside-root file: {src}")
        if src.suffix in {".pyc", ".pyo"}:
            continue
        total += src.stat().st_size
        if total > 512 * 1024 * 1024:
            raise ValueError("Source snapshot exceeds 512 MiB")
        dst = dest / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[name] = file_hash(dst)
    if not files:
        raise ValueError("Empty source snapshot")
    return {"git_head": head, "files": files}


def verify_roots(run, identities, role):
    for root, identity in identities.items():
        base = run / "sources" / role / root
        for name, expected in identity["files"].items():
            path = base / name
            if not path.is_file() or path.is_symlink() or file_hash(path) != expected:
                raise ValueError(f"Frozen source changed: {role}/{root}/{name}")
        # Report outputs and Python caches are permitted; newly injected executable sources are not.
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlink introduced in snapshot")
            if path.is_file() and path.suffix in {
                ".py",
                ".so",
                ".cu",
                ".cuh",
                ".cpp",
                ".pyc",
                ".pyo",
            }:
                name = str(path.relative_to(base))
                if name not in identity["files"]:
                    raise ValueError(f"Unrecorded executable source: {name}")


def environment(cfg, roots):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env.update(cfg["env"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), *roots.values()])
    return env


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def junit(directory, reports, required, min_tests=1):
    expected = {str(Path(n)) for n in reports}
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*.xml")}
    if actual != expected or not expected:
        raise ValueError("Missing or unexpected rank JUnit reports")
    summary = {}
    for name in reports:
        root = ET.parse(directory / name).getroot()
        cases = root.findall(".//testcase")
        if len(cases) < min_tests or any(
            root.find(f".//{tag}") is not None for tag in ("skipped", "failure", "error")
        ):
            raise ValueError(f"Empty, skipped or failed JUnit: {name}")
        ids = [c.get("classname", "") + "::" + c.get("name", "") for c in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate JUnit test identities")
        for pattern in required:
            if not any(fnmatch.fnmatchcase(node, pattern) for node in ids):
                raise ValueError(f"Missing required test {pattern} in {name}")
        summary[name] = ids
    return summary


def native_checks(run, cfg, roots, deadline, records=None):
    records = records if records is not None else []
    for check in cfg["checks"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("Native evaluation time budget exhausted")
        folder = run / "native_checks" / check["id"]
        folder.mkdir(parents=True)
        cwd = Path(roots[check["root"]])
        if check["kind"] == "gate":
            history = cwd / "apodex_core/verification/results/gate_history.json"
            history.unlink(missing_ok=True)
            argv = [
                sys.executable,
                "-m",
                "apodex_core.verification.method.gate",
                "--source",
                "operations",
                "--only",
                ",".join(check["operators"]),
            ]
        else:
            replacements = {"python": sys.executable, "reports": str(folder), **roots}
            argv = [part.format_map(replacements) for part in check["argv"]]
        log = folder / "command.log"
        write_json(folder / "command.json", {"argv": argv, "cwd": str(cwd)})
        started = time.perf_counter()
        check_env = environment(cfg, roots)
        if cfg.get("device", "cpu").startswith("cuda"):
            devices = check.get("devices", [int(cfg["device"].split(":")[1])])
            visible = check_env.get("CUDA_VISIBLE_DEVICES")
            names = visible.split(",") if visible else None
            check_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                names[i] if names else str(i) for i in devices
            )
        with log.open("w") as stream:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=check_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            finally:
                stop(process)
        if code != 0:
            raise ValueError(f"Native check failed ({code}): {check['id']}; see {log}")
        if check["kind"] == "gate":
            observed = read_json(history)
            if (
                set(observed) != set(check["operators"])
                or "GATE RESULT: PASS (no blocking verdicts)" not in log.read_text()
            ):
                raise ValueError("Missing native gate operators/final verdict")
            if any(
                r.get("overall") not in {"PASS", "WARN"}
                or type(r.get("R")) is not int
                or r["R"] <= 0
                for r in observed.values()
            ):
                raise ValueError("Invalid native gate record")
            # Native CLI owns WARN policy; no --emit-observed or local waiver substitute.
            write_json(folder / "observed.json", observed)
            detail = observed
        else:
            detail = junit(
                folder, check["reports"], check["required_tests"], check.get("min_tests", 1)
            )
        records.append(
            {
                "id": check["id"],
                "scope": check["scope"],
                "status": "pass",
                "seconds": time.perf_counter() - started,
                "detail": detail,
            }
        )
    return records


class Worker:
    def __init__(self, run, cfg, roots, role, deadline):
        self.folder = run / "workers" / role
        self.folder.mkdir(parents=True)
        self.deadline = deadline
        self.count = 0
        self.log = (self.folder / "worker.log").open("w")
        request = self.folder / "request.json"
        write_json(
            request,
            {"config": cfg, "roots": roots, "role": role, "adapter": str(run / "adapter.py")},
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "kernel_envolve.native_worker", str(request)],
            stdin=subprocess.PIPE,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=run,
            env=environment(cfg, roots),
            start_new_session=True,
        )

    def call(self, kind, case, seed):
        import torch

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
        stop(self.process)
        self.process.stdin.close()
        self.log.close()


def compare(actual, expected, cfg):
    from .arrays import assert_close
    from .training import equal, tensor_tree

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
        if isinstance(b, dict) and "shape" in b and "dtype" in b:
            keys = ("shape", "dtype", "device")
            if cfg["check_output_layout"]:
                keys += ("stride", "offset", "conjugate", "negative")
            if any(a.get(k) != b[k] for k in keys):
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

    check_meta(actual["output_metadata"], expected["output_metadata"])
    for key in ("output", "state"):
        if key == "state" and not expected[key] and not actual[key]:
            continue
        assert_close(actual[key], expected[key], cfg, key)
        if cfg["bitwise"]:
            left, right = [], []
            tensor_tree(actual[key], lambda x: left.append(x))
            tensor_tree(expected[key], lambda x: right.append(x))
            if len(left) != len(right) or any(not equal(a, b) for a, b in zip(left, right)):
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
    return pointer, run, result


def evaluate_native(task, mode="quick", timeout=1800):
    task = Path(task).resolve()
    if (
        mode not in {"quick", "full"}
        or not math.isfinite(timeout)
        or timeout <= 0
        or sys.flags.optimize
    ):
        raise ValueError("Invalid mode/timeout or optimized Python")
    cfg = load_config(task)
    state = task / ".kernel_native"
    (state / "runs").mkdir(parents=True, exist_ok=True)
    with lock(state / "evaluate.lock"):
        return _evaluate(task, state, cfg, mode, timeout)


def _evaluate(task, state, cfg, mode, timeout):
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
        "signal_only": False,
        "checks": [],
        "samples": [],
        "native_checks": [],
        "identities": {},
        "evaluator_hash": evaluator_hash(),
        "adapter_hash": file_hash(run / "adapter.py"),
        "memory": [],
        "environments": {},
    }
    workers = {}
    try:
        for role, roots in cfg["sources"].items():
            result["identities"][role] = {
                name: snapshot(path, run / "sources" / role / name, cfg["artifacts"].get(name, []))
                for name, path in roots.items()
            }
        contract_cfg = {k: v for k, v in cfg.items() if k != "sources"}
        result["contract"] = digest(
            {
                "config": contract_cfg,
                "adapter": result["adapter_hash"],
                "baseline": result["identities"]["baseline"],
            }
        )
        best = _best(state)
        result["initial_best_hash"] = digest(best[0]) if best else None
        if best and best[2]["contract"] != result["contract"]:
            raise ValueError("Native task contract or baseline changed; use a new task")
        roots = _roots(run, "candidate", result["identities"]["candidate"])
        result["comparator"] = {"kind": "best" if best else "baseline"}
        result["same_candidate_as_best"] = bool(
            best and result["identities"]["candidate"] == best[2]["identities"]["candidate"]
        )
        role_roots = {
            "oracle": _roots(run, "baseline", result["identities"]["baseline"]),
            "candidate": roots,
        }
        role_roots["comparator"] = (
            _roots(best[1], "candidate", best[2]["identities"]["candidate"])
            if best
            else role_roots["oracle"]
        )
        if best and mode == "full":
            role_roots["baseline"] = role_roots["oracle"]
        for role, sources in role_roots.items():
            workers[role] = Worker(run, cfg, sources, role, deadline)
        cases, seeds, trials, batches = plan(cfg, mode)
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

        kinds = ["fresh", "reuse", "repeat"] if cfg["deterministic"] else ["fresh", "reuse"]
        for case in cases:
            for seed in seeds:
                check(case, seed, kinds)
        if mode == "full":
            native_checks(run, cfg, roots, deadline, result["native_checks"])
            scopes = {r["scope"] for r in result["native_checks"]}
            if set(cfg["required_scopes"]) - scopes:
                raise ValueError("Missing required native validation scopes")
        timed = [c for c in cases if c.get("measure", True)]
        for batch in range(batches):
            for case in timed if batch == 0 else reversed(timed):
                for trial in range(trials):
                    expected_cache, repeats = {}, dict.fromkeys(roles, 0)
                    for role in order(roles, batch, trial):
                        repeat = repeats[role]
                        seed = timing_seed(case["id"], batch, trial, repeat, seeds)
                        if seed not in expected_cache:
                            _, oracle = call("oracle", "check", case, seed)
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
        for case in timed:
            check(case, post_seed(case["id"], seeds), ["post"])
        for role, identity in result["identities"].items():
            verify_roots(run, identity, role)
        if best:
            verify_roots(best[1], best[2]["identities"]["candidate"], "candidate")
        if (
            evaluator_hash() != result["evaluator_hash"]
            or file_hash(run / "adapter.py") != result["adapter_hash"]
        ):
            raise ValueError("Evaluator/adapter changed during evaluation")
        result.update(execution="completed", correctness="pass")
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, AssertionError):
            result.update(execution="completed", correctness="fail")
    finally:
        for worker in workers.values():
            worker.close()
    result["seconds"] = time.perf_counter() - start
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
        pointer = {
            "run": str(run.relative_to(state)),
            "result_hash": digest(result),
            "previous": best[0]["run"] if best else None,
        }
        write_json(state / "best.json", pointer)
        return pointer
