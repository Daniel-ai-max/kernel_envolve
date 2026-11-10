"""Run project-owned checks and require complete, fresh native evidence."""

import fnmatch
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .execution import stop
from .native_config import environment
from .storage import read_json, write_json


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
