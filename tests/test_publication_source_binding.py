"""Publication must bind a real evaluated snapshot to the implementation in a PR."""

import json
import subprocess

import pytest

from kernel_envolve.manager.cli import task_contract
from kernel_envolve.manager.config import validate
from kernel_envolve.manager.evidence import refresh
from kernel_envolve.manager.publication import sync_pr
from kernel_envolve.manager.store import Store
from kernel_envolve.native import evaluate_native, promote_native

pytest_plugins = ["test_native"]


def test_pr_with_different_implementation_cannot_bind_full_best(native_task, tmp_path, monkeypatch):
    task, candidate = native_task
    cfg = json.loads((task / "native.json").read_text())
    baseline = __import__("pathlib").Path(cfg["sources"]["baseline"]["core"])
    for root in (baseline, candidate):
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/example/kernels"],
            cwd=root,
            check=True,
        )
    run, result = evaluate_native(task, "full", timeout=90)
    assert result["decision"]["eligible_for_best"]
    promote_native(run)
    tested_head = result["identities"]["candidate"]["core"]["git_head"]
    (candidate / "toyop.py").write_text("def run(x): return x*3\n")
    subprocess.run(["git", "add", "toyop.py"], cwd=candidate, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "untested wrong implementation",
        ],
        cwd=candidate,
        check=True,
    )
    wrong_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=candidate, text=True
    ).strip()
    assert wrong_head != tested_head
    spec = {
        "id": "native",
        "name": "Native publication",
        "workspace": str(task),
        "task_dir": str(task),
        "kind": "native",
        "resource": "cpu",
        "command": ["true"],
        "baseline": {
            "repository": "https://github.com/example/kernels",
            "revision": "baseline",
            "entrypoint": "toyop:run",
            "introduced_on": None,
            "introduction_url": None,
            "frozen_on": "2026-11-10",
            "hardware": "CPU",
            "contract": task_contract(task, "native"),
        },
    }
    url = "https://github.com/example/kernels/pull/7"
    store = Store(tmp_path / "manager")
    try:
        store.register(
            validate(
                {"version": 1, "resources": [{"id": "cpu", "device": "cpu"}], "tasks": [spec]},
                tmp_path,
            )
        )
        refresh(store)

        reported_head = tested_head

        def github(argv):
            if argv[0] == "pr":
                return {
                    "url": url,
                    "updatedAt": "2026-11-11T12:00:00Z",
                    "headRefOid": reported_head,
                    "baseRefName": "integration",
                    "baseRefOid": result["identities"]["baseline"]["core"]["git_head"],
                    "headRepository": {"nameWithOwner": "example/kernels"},
                    "state": "OPEN",
                }
            revision = argv[1].split("/git/trees/")[1].split("?")[0]
            root = (
                baseline
                if revision == result["identities"]["baseline"]["core"]["git_head"]
                else candidate
            )
            entries = (
                subprocess.check_output(["git", "ls-tree", "-rz", revision], cwd=root)
                .decode()
                .split("\0")
            )
            tree = []
            for entry in filter(None, entries):
                info, name = entry.split("\t")
                mode, kind, oid = info.split()
                tree.append({"path": name, "mode": mode, "type": kind, "sha": oid})
            return {"truncated": False, "tree": tree}

        monkeypatch.setattr("kernel_envolve.manager.publication.github", github)
        sync_pr(store, "native", str(run), url, source_root="core", base_branch="integration")
        reported_head = wrong_head
        with pytest.raises(ValueError, match="source|implementation|head"):
            sync_pr(store, "native", str(run), url, source_root="core", base_branch="integration")
    finally:
        store.close()
