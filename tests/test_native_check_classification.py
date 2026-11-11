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
