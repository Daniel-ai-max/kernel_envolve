"""Design completeness and source auditing, deliberately without GPU execution."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a7b_design", ROOT / "tools/check_a7b_acceptance.py")
design = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(design)


def load():
    folder = ROOT / "acceptance/a7b"
    return (
        json.loads((folder / "catalog.json").read_text()),
        json.loads((folder / "source-inventory.json").read_text()),
    )


def test_both_source_profiles_are_complete_and_documents_current():
    catalog, inventory = load()
    assert design.check_catalog(catalog, inventory) == (9, 16)
    for operator in catalog["operators"]:
        assert (design.DEFAULT / operator["document"]).read_text() == design.render_operator(
            operator, catalog
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_operator",
        "missing_profile",
        "flatten_priorities",
        "duplicate",
        "false_pass",
        "include_investigation",
        "missing_oracle",
        "unsafe_document",
        "missing_gap",
    ],
)
def test_incomplete_or_misleading_design_is_rejected(mutation):
    catalog, inventory = load()
    if mutation == "missing_operator":
        catalog["operators"].pop()
    elif mutation == "missing_profile":
        catalog["operators"][0]["priorities"].pop("e98")
    elif mutation == "flatten_priorities":
        next(op for op in catalog["operators"] if op["id"] == "cp-alltoall")["priorities"] = {
            "e65": "P0",
            "e98": "P0",
        }
    elif mutation == "duplicate":
        catalog["operators"].append(copy.deepcopy(catalog["operators"][0]))
    elif mutation == "false_pass":
        catalog["operators"][0]["status"] = "verified"
    elif mutation == "include_investigation":
        operator = copy.deepcopy(catalog["operators"][0])
        operator.update(
            id="tensor-primitives",
            source_id=49,
            document="operators/tensor-primitives.md",
            title="基础 Tensor 原语",
        )
        catalog["operators"].append(operator)
    elif mutation == "missing_oracle":
        catalog["operators"][0]["oracle"] = ""
    elif mutation == "unsafe_document":
        catalog["operators"][0]["document"] = "../README.md"
    elif mutation == "missing_gap":
        catalog["operators"][0]["gaps"] = []
    with pytest.raises(ValueError):
        design.check_catalog(catalog, inventory)


def test_new_source_priority_cannot_be_silently_omitted():
    catalog, inventory = load()
    next(row for row in inventory["profiles"]["e98"]["items"] if row["source_id"] == 49)[
        "priority"
    ] = "P2"
    with pytest.raises(ValueError, match="coverage mismatch"):
        design.check_catalog(catalog, inventory)


def test_external_source_check_uses_pinned_git_objects_without_importing(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    file = tmp_path / "checks.py"
    # Importing this would fail. Auditing a candidate's mutable file instead would miss the test.
    file.write_text("raise RuntimeError('must never execute')\ndef test_expected(): pass\n")
    git("add", "checks.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
    revision = git("rev-parse", "HEAD")
    file.write_text("def wrong_current_checkout(): pass\n")
    reference = {
        "repository": "core",
        "path": "checks.py",
        "kind": "test",
        "symbols": ["test_expected"],
    }
    catalog = {
        "audited_sources": {"core": {"revision": revision}},
        "operators": [{"references": [reference]}],
    }
    assert design.check_sources(catalog, {"core": tmp_path}) == 1
    reference["symbols"] = ["test_invented"]
    with pytest.raises(ValueError, match="missing symbols"):
        design.check_sources(catalog, {"core": tmp_path})
