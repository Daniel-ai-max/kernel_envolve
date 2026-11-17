"""Regression coverage for moved modules, packaged resources and evidence identity."""

import ast
import importlib
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest

from kernel_envolve._paths import IMPORT_ROOT, PACKAGE_ROOT
from kernel_envolve.evaluation.native.config import environment
from kernel_envolve.evaluation.shared import storage

# Existing adapter/worker imports remain valid while implementation uses canonical paths.
ALIASES = {
    "kernel_envolve.manager.linked_delivery": "kernel_envolve.manager.results.linked_delivery",
    "kernel_envolve.arrays": "kernel_envolve.evaluation.shared.arrays",
    "kernel_envolve.execution": "kernel_envolve.evaluation.shared.execution",
    "kernel_envolve.protocol": "kernel_envolve.evaluation.shared.protocol",
    "kernel_envolve.runtime": "kernel_envolve.evaluation.shared.runtime",
    "kernel_envolve.scoring": "kernel_envolve.evaluation.shared.scoring",
    "kernel_envolve.storage": "kernel_envolve.evaluation.shared.storage",
    "kernel_envolve.worker": "kernel_envolve.evaluation.generic.worker",
    "kernel_envolve.native": "kernel_envolve.evaluation.native.evaluator",
    "kernel_envolve.native_checks": "kernel_envolve.evaluation.native.checks",
    "kernel_envolve.native_config": "kernel_envolve.evaluation.native.config",
    "kernel_envolve.native_confirmation": "kernel_envolve.evaluation.native.confirmation",
    "kernel_envolve.native_harness": "kernel_envolve.evaluation.native.harness",
    "kernel_envolve.native_worker": "kernel_envolve.evaluation.native.worker",
    "kernel_envolve.training": "kernel_envolve.evaluation.native.workload",
    "kernel_envolve.manager.process": "kernel_envolve.manager.execution.process",
    "kernel_envolve.manager.gpu": "kernel_envolve.manager.execution.gpu",
    "kernel_envolve.manager.worker": "kernel_envolve.manager.execution.worker",
    "kernel_envolve.manager.bridge": "kernel_envolve.manager.execution.bridge",
    "kernel_envolve.manager.recovery": "kernel_envolve.manager.execution.recovery",
    "kernel_envolve.manager.evidence": "kernel_envolve.manager.results.index",
    "kernel_envolve.manager.archive": "kernel_envolve.manager.results.archive",
    "kernel_envolve.manager.assessment": "kernel_envolve.manager.results.assessment",
    "kernel_envolve.manager.validation_import": "kernel_envolve.manager.results.validation_import",
    "kernel_envolve.manager.publication": "kernel_envolve.manager.results.publication",
    "kernel_envolve.manager.publication_sources": "kernel_envolve.manager.results.publication_sources",
    "kernel_envolve.manager.research_source": "kernel_envolve.manager.results.research_source",
    "kernel_envolve.manager.research_result": "kernel_envolve.manager.results.research_result",
    "kernel_envolve.manager.web": "kernel_envolve.manager.dashboard.web",
    "kernel_envolve.manager.views": "kernel_envolve.manager.dashboard.views",
    "kernel_envolve.manager.command_view": "kernel_envolve.manager.dashboard.command_view",
    "kernel_envolve.manager.row_facts": "kernel_envolve.manager.dashboard.row_facts",
    "kernel_envolve.manager.narrator": "kernel_envolve.manager.dashboard.narrator",
    "kernel_envolve.manager.narrator_client": "kernel_envolve.manager.dashboard.narrator_client",
}


@pytest.mark.parametrize("legacy,canonical", ALIASES.items())
def test_legacy_import_is_same_module(legacy, canonical):
    old = importlib.import_module(legacy)
    new = importlib.import_module(canonical)
    assert old is new
    parent, _, name = legacy.rpartition(".")
    assert getattr(importlib.import_module(parent), name) is new


def test_implementation_does_not_import_compatibility_modules():
    for path in PACKAGE_ROOT.rglob("*.py"):
        module = "kernel_envolve." + ".".join(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
        if module in ALIASES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ALIASES, (path, node.lineno, node.module)
                for alias in node.names:
                    assert f"{node.module}.{alias.name}" not in ALIASES, (path, node.lineno)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ALIASES, (path, node.lineno)


def test_fingerprint_covers_nested_sources_and_relative_names(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "PACKAGE_ROOT", tmp_path)
    (tmp_path / "__init__.py").write_text("")
    nested = tmp_path / "evaluation" / "shared"
    nested.mkdir(parents=True)
    code = nested / "scoring.py"
    code.write_text("policy = 1\n")
    first = storage.evaluator_hash()
    code.write_text("policy = 2\n")
    second = storage.evaluator_hash()
    assert first != second
    code.rename(nested / "policy.py")
    renamed = storage.evaluator_hash()
    assert renamed != second
    (nested / "ignored.pyc").write_bytes(b"cache")
    assert storage.evaluator_hash() == renamed


def test_native_worker_environment_finds_installed_package(tmp_path):
    env = environment({"env": {}}, {"fixture": str(tmp_path)})
    assert env["PYTHONPATH"].split(os.pathsep) == [str(IMPORT_ROOT), str(tmp_path)]
    probe = subprocess.run(
        [sys.executable, "-c", "import kernel_envolve; print(kernel_envolve.__file__)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert Path(probe.stdout.strip()).resolve() == PACKAGE_ROOT / "__init__.py"


def test_candidate_validation_does_not_load_manager(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kernel_envolve.validation; "
            "assert not any(m.startswith('kernel_envolve.manager') for m in sys.modules)",
        ],
        cwd=tmp_path,
        check=True,
        timeout=10,
    )


def test_dashboard_assets_are_packaged():
    root = files("kernel_envolve.manager.dashboard").joinpath("static")
    for name in ("index.html", "app.js", "summary.js", "sync.js", "style.css"):
        assert root.joinpath(name).read_bytes(), name


def test_cli_exports_generic_engine():
    from kernel_envolve import cli
    from kernel_envolve.evaluation.generic import evaluator

    for name in ("evaluate", "audit_result", "promote"):
        assert getattr(cli, name) is getattr(evaluator, name)


@pytest.mark.parametrize(
    "module",
    [
        "kernel_envolve.manager.narrator_client",
        "kernel_envolve.manager.dashboard.narrator_client",
    ],
)
def test_narrator_entrypoint_still_executes(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", module],
        input="not-json",
        text=True,
        cwd=tmp_path,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert result.stderr.strip() == "JSONDecodeError"
