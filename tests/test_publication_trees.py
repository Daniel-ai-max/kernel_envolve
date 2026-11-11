"""Real Git trees exercise publication scope, helpers and local mirror provenance."""

import hashlib
import json
import shutil
import subprocess

import pytest

from kernel_envolve.manager.publication_sources import bind_source, checkout
from kernel_envolve.native import snapshot
from kernel_envolve.storage import digest

REPO = "example/kernels"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root):
    git(root, "add", "-A")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-qm",
        "fixture",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def source_case(tmp_path):
    baseline, candidate, run = (tmp_path / name for name in ("baseline", "candidate", "run"))
    baseline.mkdir()
    git(baseline, "init", "-q")
    git(baseline, "remote", "add", "origin", f"https://github.com/{REPO}")
    for name, contents in {
        "pkg/kernel.py": "slow implementation\n",
        "pkg/helper.py": "helper\n",
        "foundation.py": "fixed foundation\n",
        "kernel.config.json": "{}\n",
        "test_contract.py": "original harness\n",
    }.items():
        path = baseline / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(contents)
    base = commit(baseline)
    git(tmp_path, "clone", "-q", "--shared", str(baseline), str(candidate))
    (candidate / "pkg/kernel.py").write_text("optimized implementation\n")
    head = commit(candidate)
    cfg = {
        "sources": {"baseline": {"core": str(baseline)}, "candidate": {"core": str(candidate)}},
        "implementation_paths": {"core": ["pkg"]},
        "checks": [],
    }
    spec = {"kind": "native", "baseline": {"repository": f"https://github.com/{REPO}"}}
    pr = {
        "url": f"https://github.com/{REPO}/pull/7",
        "headRepository": {"nameWithOwner": REPO},
        "headRefOid": head,
    }

    def freeze(artifacts=()):
        if run.exists():
            shutil.rmtree(run)
        run.mkdir()
        identities = {
            role: {
                name: snapshot(path, run / "sources" / role / name, artifacts)
                for name, path in roots.items()
            }
            for role, roots in cfg["sources"].items()
        }
        result = {"identities": identities, "adapter_hash": "adapter"}
        verified = {
            "result_hash": digest(result),
            "contract": digest(
                {
                    "config": {k: v for k, v in cfg.items() if k != "sources"},
                    "adapter": "adapter",
                    "baseline": identities["baseline"],
                }
            ),
        }
        (run / "config.json").write_text(json.dumps(cfg))
        (run / "result.json").write_text(json.dumps(result))
        return verified

    def fetch(args):
        revision = args[1].split("/git/trees/")[1].split("?")[0]
        tree = []
        raw = subprocess.check_output(["git", "ls-tree", "-rz", revision], cwd=candidate).decode()
        for entry in filter(None, raw.split("\0")):
            info, name = entry.split("\t")
            mode, kind, oid = info.split()
            tree.append({"path": name, "sha": oid, "mode": mode, "type": kind})
        return {"tree": tree, "truncated": False}

    return baseline, candidate, run, spec, pr, freeze, fetch, cfg, base


def bind(case, verified):
    _, _, run, spec, pr, _, fetch, _, _ = case
    return bind_source(spec, run, "core", pr, fetch, verified)


def test_shared_local_clone_fixture_and_git_metadata_are_not_shipped(source_case):
    baseline, candidate, _, _, _, freeze, _, _, _ = source_case
    for root in (baseline, candidate):
        (root / "test_contract.py").write_text("trusted test fixture overlay\n")
    assert checkout(candidate)[0] == REPO
    proof = bind(source_case, freeze([".git/**/*"]))
    assert proof["source_root"] == "core"
    assert proof["implementation_paths"] == ["pkg"]


def test_snapshot_uncommitted_contents_bind_after_commit(source_case):
    _, candidate, _, _, pr, freeze, _, _, _ = source_case
    (candidate / "pkg/helper.py").write_text("new tested helper\n")
    verified = freeze()
    pr["headRefOid"] = commit(candidate)
    assert bind(source_case, verified)["implementation_digest"]


@pytest.mark.parametrize(
    "path",
    [
        "pkg/helper.py",
        "pkg/new_helper.py",
        "foundation.py",
        "kernel.config.json",
        "test_contract.py",
        "README.md",
    ],
)
def test_untested_head_changes_and_extra_files_are_rejected(source_case, path):
    _, candidate, _, _, pr, freeze, _, _, _ = source_case
    verified = freeze()
    (candidate / path).write_text("untested change\n")
    pr["headRefOid"] = commit(candidate)
    with pytest.raises(ValueError, match="unverified|implementation/helper"):
        bind(source_case, verified)


def test_tested_deletion_binds_but_unrecorded_deletion_does_not(source_case):
    _, candidate, _, _, pr, freeze, _, _, _ = source_case
    (candidate / "pkg/helper.py").unlink()
    verified = freeze()
    pr["headRefOid"] = commit(candidate)
    assert bind(source_case, verified)
    (candidate / "pkg/kernel.py").unlink()
    pr["headRefOid"] = commit(candidate)
    with pytest.raises(ValueError, match="implementation/helper"):
        bind(source_case, verified)


def test_patched_production_foundation_cannot_be_hidden_as_fixture(source_case):
    baseline, candidate, _, _, _, freeze, _, _, _ = source_case
    for root in (baseline, candidate):
        (root / "foundation.py").write_text("different evaluated production dependency\n")
    with pytest.raises(ValueError, match="fixed baseline source"):
        bind(source_case, freeze())


def test_changed_frozen_source_or_scope_cannot_race_verification(source_case):
    _, _, run, _, _, freeze, _, _, _ = source_case
    verified = freeze()
    (run / "sources/candidate/core/pkg/kernel.py").write_text("changed frozen source\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        bind(source_case, verified)
    verified = freeze()
    cfg = json.loads((run / "config.json").read_text())
    cfg["implementation_paths"] = {"core": ["pkg", "foundation.py"]}
    (run / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="configuration changed"):
        bind(source_case, verified)


def test_wrong_repository_and_truncated_tree_are_rejected(source_case):
    _, _, run, spec, pr, freeze, fetch, _, _ = source_case
    verified = freeze()
    with pytest.raises(ValueError, match="repository differs"):
        bind_source(
            spec, run, "core", pr | {"url": "https://github.com/other/repo/pull/7"}, fetch, verified
        )
    with pytest.raises(ValueError, match="incomplete source tree"):
        bind_source(spec, run, "core", pr, lambda _: {"tree": [], "truncated": True}, verified)


def test_mirror_cycles_fail_closed(source_case):
    baseline, candidate, _, _, _, _, _, _, _ = source_case
    git(baseline, "remote", "set-url", "origin", str(candidate))
    with pytest.raises(ValueError, match="Cycle"):
        checkout(candidate)


def test_multiple_roots_bind_one_delivery_and_preserve_task_repository(source_case, tmp_path):
    _, _, _, spec, _, freeze, _, cfg, _ = source_case
    for role in ("baseline", "candidate"):
        sibling = tmp_path / (role + "-megatron")
        sibling.mkdir()
        git(sibling, "init", "-q")
        git(sibling, "remote", "add", "origin", "https://github.com/example/megatron")
        (sibling / "model.py").write_text("fixed model fixture\n")
        commit(sibling)
        cfg["sources"][role]["megatron"] = str(sibling)
    # The task's primary repository may be Megatron while this PR delivers Core.
    spec["baseline"]["repository"] = "https://github.com/example/megatron"
    assert bind(source_case, freeze())["repository"] == REPO
    spec["baseline"]["repository"] = "https://github.com/example/unrelated"
    with pytest.raises(ValueError, match="does not match its source roots"):
        bind(source_case, freeze())


def test_advanced_base_production_changes_are_not_silently_inherited(source_case):
    _, candidate, _, _, pr, freeze, _, _, _ = source_case
    verified = freeze()
    (candidate / "foundation.py").write_text("new upstream dependency\n")
    pr["headRefOid"] = commit(candidate)
    with pytest.raises(ValueError, match="outside implementation_paths"):
        bind(source_case, verified)


def test_fork_head_is_verified_against_its_own_tree(source_case):
    _, candidate, _, _, pr, freeze, fetch, _, _ = source_case
    git(candidate, "remote", "set-url", "origin", "git@github.com:contributor/kernels.git")
    pr["headRepository"] = {"nameWithOwner": "contributor/kernels"}
    requests = []

    def record(args):
        requests.append(args[1])
        return fetch(args)

    case = list(source_case)
    case[6] = record
    proof = bind(case, freeze())
    assert proof["head_repository"] == "contributor/kernels"
    assert requests[-1].startswith("repos/contributor/kernels/git/trees/")


def test_selected_native_root_without_implementation_change_is_rejected(source_case):
    baseline, candidate, _, _, _, freeze, _, _, _ = source_case
    (candidate / "pkg/kernel.py").write_bytes((baseline / "pkg/kernel.py").read_bytes())
    with pytest.raises(ValueError, match="no evaluated implementation change"):
        bind(source_case, freeze())


def test_generic_frozen_candidate_and_fixed_dependencies_bind(source_case):
    baseline, candidate, run, spec, pr, freeze, fetch, _, _ = source_case
    config = {"candidate_files": ["pkg/kernel.py", "pkg/helper.py"]}
    for root in (baseline, candidate):
        (root / "task.json").write_text(json.dumps(config))
    spec["kind"] = "generic"
    spec["task_dir"] = str(baseline)
    spec["baseline"]["revision"] = commit(baseline)
    pr["headRefOid"] = commit(candidate)
    # Share the newly created baseline object with the mocked GitHub tree reader.
    git(candidate, "fetch", "-q", str(baseline), "HEAD")
    freeze()
    shutil.copytree(candidate, run / "source", ignore=shutil.ignore_patterns(".git"))
    files = {
        path.relative_to(run / "source").as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (run / "source").rglob("*")
        if path.is_file()
    }
    result = {"identity": {"files": files}}
    (run / "result.json").write_text(json.dumps(result))
    proof = bind_source(spec, run, "task", pr, fetch, {"result_hash": digest(result)})
    assert proof["implementation_paths"] == config["candidate_files"]
    (run / "source/task.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        bind_source(spec, run, "task", pr, fetch, {"result_hash": digest(result)})


def test_relative_mirror_origin_resolves_from_repository_root(source_case):
    _, candidate, _, _, _, _, _, _, _ = source_case
    git(candidate, "remote", "set-url", "origin", "../baseline")
    assert checkout(candidate / "pkg") == (REPO, "pkg/")
