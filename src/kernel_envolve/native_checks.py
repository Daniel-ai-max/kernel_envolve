"""Run project-owned checks and require complete, fresh native evidence."""

import fnmatch
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .execution import spawn, stop
from .native_config import environment
from .storage import read_json, write_json


def junit(directory, reports, required, min_tests=1, *, failures=None):
    expected = {str(Path(n)) for n in reports}
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*.xml")}
    if actual != expected or not expected:
        raise ValueError("Missing or unexpected rank JUnit reports")
    summary = {}
    rejected = []
    for name in reports:
        root = ET.parse(directory / name).getroot()
        if root.tag not in {"testsuite", "testsuites"}:
            raise ValueError(f"Invalid JUnit root: {name}")
        cases = root.findall(".//testcase")
        failure_count = sum(len(c.findall("failure")) for c in cases)
        if failure_count != len(root.findall(".//failure")) or any(
            len(c.findall("failure")) > 1 for c in cases
        ):
            raise ValueError(f"Invalid JUnit failure placement: {name}")
        if len(cases) < min_tests or any(
            root.find(f".//{tag}") is not None for tag in ("skipped", "error")
        ):
            raise ValueError(f"Empty, skipped or failed JUnit: {name}")
        ids = [c.get("classname", "") + "::" + c.get("name", "") for c in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate JUnit test identities")
        for pattern in required:
            if not any(fnmatch.fnmatchcase(node, pattern) for node in ids):
                raise ValueError(f"Missing required test {pattern} in {name}")
        for suite in root.iter("testsuite"):
            children = suite.findall(".//testcase")
            if failure_count and not {"tests", "failures", "errors", "skipped"} <= set(
                suite.keys()
            ):
                raise ValueError(f"Missing JUnit completion counts: {name}")
            for field, count in {
                "tests": len(children),
                "failures": sum(c.find("failure") is not None for c in children),
                "errors": 0,
                "skipped": 0,
            }.items():
                if field in suite.attrib and int(suite.get(field)) != count:
                    raise ValueError(f"Inconsistent JUnit {field}: {name}")
        for case, node in zip(cases, ids):
            for failure in case.findall("failure"):
                message = failure.get("message", "")
                # Pytest emits untyped failures: the message is the exception-only
                # representation (plain rewritten assertions start with `assert`).
                kind = failure.get("type")
                assertion = kind in {"AssertionError", "builtins.AssertionError"} or (
                    kind is None
                    and (
                        message == "AssertionError"
                        or message.startswith(("AssertionError:", "assert ", "assert\n"))
                    )
                )
                if not assertion:
                    raise ValueError(f"Non-assertion JUnit failure: {name}: {node}")
                rejected.append({"report": name, "test": node, "message": message})
        summary[name] = ids
    if rejected:
        if failures is None:
            raise ValueError("Failed JUnit assertions")
        failures.extend(rejected)
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
            process = spawn(
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
        record = {
            "id": check["id"],
            "scope": check["scope"],
            "status": "error",
            "seconds": time.perf_counter() - started,
            "returncode": code,
        }
        records.append(record)
        try:
            if check["kind"] == "gate":
                observed = read_json(history)
                write_json(folder / "observed.json", observed)
                record["detail"] = observed
                if set(observed) != set(check["operators"]) or any(
                    r.get("overall") not in {"PASS", "WARN", "FAIL"}
                    or type(r.get("R")) is not int
                    or r["R"] <= 0
                    for r in observed.values()
                ):
                    raise ValueError("Missing or invalid native gate operators/records")
                output = log.read_text()
                if "Traceback (most recent call last):" in output:
                    raise ValueError("Native gate terminated with a Python traceback")
                lines = output.strip().splitlines()
                verdict = lines[-1] if lines else ""
                # These are the operations CLI's terminal verdicts, emitted after
                # writing history and applying its own EXPECTED_WARN policy.
                rejected = code == 1 and verdict == "GATE RESULT: BLOCKED"
                passed = code == 0 and verdict == "GATE RESULT: PASS (no blocking verdicts)"
                if rejected and not any(r["overall"] != "PASS" for r in observed.values()):
                    raise ValueError("Blocked native gate has no nonpassing verdict")
                if not (rejected or passed) or (
                    passed and any(r["overall"] == "FAIL" for r in observed.values())
                ):
                    raise ValueError("Missing or inconsistent native gate final verdict/exit code")
            else:
                failures = []
                record["detail"] = junit(
                    folder,
                    check["reports"],
                    check["required_tests"],
                    check.get("min_tests", 1),
                    failures=failures,
                )
                record["failures"] = failures
                rejected = bool(failures) and code == 1
                if not rejected and (code != 0 or failures):
                    raise ValueError(f"Inconsistent native JUnit verdict/exit code ({code})")
            record["status"] = "fail" if rejected else "pass"
            if rejected:
                raise AssertionError(f"Native check rejected candidate: {check['id']}; see {log}")
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
    return records
