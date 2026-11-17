"""Only complete project-owned rejection evidence is a correctness verdict."""

import json
import time

import pytest

from kernel_envolve.native_checks import junit, native_checks


def pytest_check(tmp_path, source):
    root = tmp_path / "core"
    root.mkdir()
    (root / "test_contract.py").write_text(source)
    cfg = {
        "env": {},
        "checks": [
            {
                "id": "operator",
                "scope": "operator",
                "kind": "pytest",
                "root": "core",
                "argv": [
                    "{python}",
                    "-m",
                    "pytest",
                    "test_contract.py",
                    "-q",
                    "--junitxml={reports}/junit.xml",
                ],
                "reports": ["junit.xml"],
                "required_tests": ["test_contract::test_contract"],
            }
        ],
    }
    return cfg, root


@pytest.mark.parametrize(
    "body,exception,status",
    [
        ("assert 1 == 2", AssertionError, "fail"),
        ("raise AssertionError('gradient mismatch')", AssertionError, "fail"),
        ("raise RuntimeError('CUDA launch failed')", ValueError, "error"),
        ("raise ImportError('missing dependency')", ValueError, "error"),
        ("import pytest; pytest.skip('no device')", ValueError, "error"),
    ],
)
def test_real_pytest_failure_classification(tmp_path, body, exception, status):
    cfg, root = pytest_check(tmp_path, f"def test_contract():\n {body}\n")
    records = []
    with pytest.raises(exception):
        native_checks(tmp_path / "run", cfg, {"core": str(root)}, time.monotonic() + 20, records)
    assert records[0]["status"] == status
    assert records[0]["seconds"] > 0
    if status == "fail":
        assert records[0]["returncode"] == 1
        assert records[0]["failures"][0]["test"] == "test_contract::test_contract"
    assert (tmp_path / "run/native_checks/operator/junit.xml").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_rank",
        "coverage",
        "counts",
        "missing_counts",
        "error",
        "skip",
        "runtime",
        "outside",
        "duplicate",
    ],
)
def test_assertion_never_masks_incomplete_junit(tmp_path, problem):
    case = '<testcase classname="mod" name="test_x"><failure message="assert 1 == 2"/></testcase>'
    counts = 'tests="1" failures="1" errors="0" skipped="0"'
    if problem == "counts":
        counts = counts.replace('tests="1"', 'tests="2"')
    if problem == "missing_counts":
        counts = ""
    if problem in {"error", "skip"}:
        case = case.replace(
            "</testcase>", f"<{'error' if problem == 'error' else 'skipped'}/></testcase>"
        )
    if problem == "runtime":
        case = case.replace("assert 1 == 2", "RuntimeError: CUDA launch failed")
    if problem == "outside":
        case += '<failure message="assert 1 == 2"/>'
    if problem == "duplicate":
        case += case
        counts = counts.replace('="1"', '="2"')
    (tmp_path / "rank0.xml").write_text(f"<testsuite {counts}>{case}</testsuite>")
    reports = ["rank0.xml", "rank1.xml"] if problem == "missing_rank" else ["rank0.xml"]
    required = ["mod::missing"] if problem == "coverage" else ["mod::test_x"]
    with pytest.raises(ValueError):
        junit(tmp_path, reports, required, failures=[])


@pytest.mark.parametrize("code", [0, 2, 3, 4, 5, -9])
def test_exit_code_must_match_complete_rejection(tmp_path, code):
    cfg, root = pytest_check(tmp_path, "")
    xml = '<testsuite tests="1" failures="1" errors="0" skipped="0"><testcase classname="test_contract" name="test_contract"><failure message="AssertionError: gradient"/></testcase></testsuite>'
    # A launcher preserves the complete report but exits differently from pytest.
    (root / "launcher.py").write_text(
        f"from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text({xml!r})\nsys.exit({code})\n"
    )
    cfg["checks"][0]["argv"] = ["{python}", "launcher.py", "{reports}/junit.xml"]
    records = []
    with pytest.raises(ValueError, match="exit code"):
        native_checks(tmp_path / "run", cfg, {"core": str(root)}, time.monotonic() + 20, records)
    assert records[0]["status"] == "error"


@pytest.mark.parametrize(
    "mode", ["blocked", "warn", "missing", "runtime", "wrong_code", "all_pass"]
)
def test_gate_rejection_requires_native_terminal_protocol(tmp_path, mode):
    root = tmp_path / "core"
    method = root / "apodex_core/verification/method"
    method.mkdir(parents=True)
    for folder in (root / "apodex_core", method.parent, method):
        (folder / "__init__.py").touch()
    observed = {
        "op": {
            "overall": "WARN" if mode == "warn" else "PASS" if mode == "all_pass" else "FAIL",
            "R": 4096,
        }
    }
    code = "from pathlib import Path\np=Path('apodex_core/verification/results/gate_history.json')\np.parent.mkdir(parents=True,exist_ok=True)\n"
    if mode != "missing":
        code += f"p.write_text({json.dumps(observed)!r})\n"
    code += "print('GATE RESULT: BLOCKED')\n"
    if mode == "runtime":
        code += "raise RuntimeError('after verdict')\n"
    else:
        code += f"raise SystemExit({0 if mode == 'wrong_code' else 1})\n"
    (method / "gate.py").write_text(code)
    cfg = {
        "env": {},
        "checks": [
            {
                "id": "numerical",
                "scope": "numerical",
                "kind": "gate",
                "root": "core",
                "operators": ["op"],
            }
        ],
    }
    records = []
    with pytest.raises(
        AssertionError if mode in {"blocked", "warn"} else (ValueError, FileNotFoundError)
    ):
        native_checks(tmp_path / "run", cfg, {"core": str(root)}, time.monotonic() + 20, records)
    assert records[0]["status"] == ("fail" if mode in {"blocked", "warn"} else "error")


@pytest.mark.parametrize("mode", ["launch", "timeout", "expired"])
def test_execution_errors_have_check_receipts(tmp_path, mode):
    import subprocess

    cfg, root = pytest_check(tmp_path, "")
    if mode == "launch":
        cfg["checks"][0]["argv"] = [str(root / "missing-executable")]
        expected = FileNotFoundError
    else:
        cfg["checks"][0]["argv"] = [
            "{python}",
            "-u",
            "-c",
            "import time; print('started'); time.sleep(10)",
        ]
        expected = TimeoutError if mode == "expired" else subprocess.TimeoutExpired
    records = []
    deadline = time.monotonic() + (-1 if mode == "expired" else 0.4 if mode == "timeout" else 10)
    with pytest.raises(expected):
        native_checks(tmp_path / "run", cfg, {"core": str(root)}, deadline, records)
    record = records[0]
    assert record["id"] == "operator" and record["scope"] == "operator"
    assert record["status"] == "error" and record["returncode"] is None
    assert record["error"]["type"] == expected.__name__
    assert record["seconds"] > 0
    if mode == "timeout":
        assert record["seconds"] >= 0.35
        assert "started" in (tmp_path / "run/native_checks/operator/command.log").read_text()


@pytest.mark.parametrize("body,code", [("pass", 0), ("assert False, 'gradient'", 1)])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_receipt_preserves_command_verdict(
    tmp_path, monkeypatch, body, code, cleanup_fails
):
    import kernel_envolve.native_checks as checks

    cfg, root = pytest_check(tmp_path, f"def test_contract():\n {body}\n")
    real_stop = checks.stop
    cleanup_seconds = 0.03

    def cleanup(process):
        real_stop(process)
        time.sleep(cleanup_seconds)
        if cleanup_fails:
            raise AssertionError("cleanup probe failed")

    monkeypatch.setattr(checks, "stop", cleanup)
    records = []
    if cleanup_fails or code:
        with pytest.raises(RuntimeError if cleanup_fails else AssertionError):
            native_checks(
                tmp_path / "run", cfg, {"core": str(root)}, time.monotonic() + 20, records
            )
    else:
        native_checks(tmp_path / "run", cfg, {"core": str(root)}, time.monotonic() + 20, records)
    record = records[0]
    assert record["returncode"] == code
    assert record["seconds"] >= cleanup_seconds
    assert record["status"] == ("error" if cleanup_fails else "fail" if code else "pass")
    assert len(record["failures"]) == code
    if code:
        assert record["error"]["type"] == "AssertionError"
        assert "gradient" in record["failures"][0]["message"]
    if cleanup_fails:
        assert record["cleanup_error"] == {
            "type": "AssertionError",
            "message": "cleanup probe failed",
        }
    assert (tmp_path / "run/native_checks/operator/junit.xml").exists()


def test_timeout_and_cleanup_errors_are_both_preserved(tmp_path, monkeypatch):
    import kernel_envolve.native_checks as checks

    cfg, root = pytest_check(tmp_path, "")
    cfg["checks"][0]["argv"] = ["{python}", "-c", "import time; time.sleep(10)"]
    real_stop = checks.stop

    def cleanup(process):
        real_stop(process)
        raise OSError("cleanup confirmation failed")

    monkeypatch.setattr(checks, "stop", cleanup)
    records = []
    with pytest.raises(RuntimeError, match="cleanup"):
        native_checks(tmp_path / "run", cfg, {"core": str(root)}, time.monotonic() + 0.2, records)
    assert records[0]["error"]["type"] == "TimeoutExpired"
    assert records[0]["cleanup_error"]["type"] == "OSError"
    assert records[0]["returncode"] is None and records[0]["status"] == "error"


@pytest.mark.parametrize("shape", ["no_suite", "mixed", "wrapped"])
def test_junit_cases_outside_suites_are_incomplete(tmp_path, shape):
    case = '<testcase classname="m" name="test_x"><failure message="assert False"/></testcase>'
    suite = '<testsuite tests="0" failures="0" errors="0" skipped="0"/>'
    if shape == "no_suite":
        xml = f"<testsuites>{case}</testsuites>"
    elif shape == "mixed":
        xml = f"<testsuites>{suite}{case}</testsuites>"
    else:
        xml = f"<testsuite><wrapper>{case}</wrapper></testsuite>"
    (tmp_path / "junit.xml").write_text(xml)
    with pytest.raises(ValueError, match="must belong"):
        junit(tmp_path, ["junit.xml"], ["m::test_x"], failures=[])


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_junit_suite_structure_accepts_complete_flat_and_nested(tmp_path, nested, failed):
    failure = '<failure message="AssertionError: mismatch"/>' if failed else ""
    case = f'<testcase classname="m" name="test_x">{failure}</testcase>'
    attrs = f'tests="1" failures="{int(failed)}" errors="0" skipped="0"'
    suite = f"<testsuite {attrs}>{case}</testsuite>"
    if nested:
        suite = f"<testsuite {attrs}>{suite}</testsuite>"
    (tmp_path / "junit.xml").write_text(f"<testsuites>{suite}</testsuites>")
    failures = []
    assert junit(tmp_path, ["junit.xml"], ["m::test_x"], failures=failures) == {
        "junit.xml": ["m::test_x"]
    }
    assert len(failures) == int(failed)
