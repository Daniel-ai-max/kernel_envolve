"""Compare a PR's immutable Git trees with one frozen evaluator source root."""

import hashlib
import re
import stat
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..native_harness import check_paths, covered, protected
from ..storage import digest, read_json


def repository(value):
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?", value
    )
    if not match:
        raise ValueError("Publication requires a GitHub source repository")
    return match[1].lower()


def checkout(path):
    path = Path(path).resolve()
    prefix = subprocess.check_output(
        ["git", "rev-parse", "--show-prefix"], cwd=path, text=True, timeout=10
    ).strip()
    seen = set()
    for _ in range(8):
        identity = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=path, text=True, timeout=10
        ).strip()
        identity = str(Path(identity).resolve())
        if identity in seen:
            raise ValueError("Cycle in local source repository provenance")
        seen.add(identity)
        origin = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=path, text=True, timeout=10
        ).strip()
        try:
            return repository(origin), prefix
        except ValueError:
            parsed = urlparse(origin)
            if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
                source = Path(unquote(parsed.path))
            elif not parsed.scheme and ":" not in origin:
                source = Path(origin)
            else:
                raise ValueError("Local mirrors must lead to a GitHub source repository") from None
            path = (path / source).resolve()
            if not path.is_dir():
                raise ValueError("Missing local source repository provenance")
    raise ValueError("Local repository provenance exceeds 8 hops")


def tree(fetch, repo, revision):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Frozen baseline and PR head must be immutable Git revisions")
    reply = fetch(["api", f"repos/{repo}/git/trees/{revision}?recursive=1"])
    if reply.get("truncated") or not isinstance(reply.get("tree"), list):
        raise ValueError("GitHub returned an incomplete source tree")
    entries = {}
    for item in reply["tree"]:
        name = item["path"]
        if item["type"] == "tree":
            continue
        if name in entries or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid GitHub source tree")
        entries[name] = {k: item[k] for k in ("sha", "mode", "type")}
    return entries


def blob(path, expected_hash):
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("Publication source must contain regular files")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_hash:
        raise ValueError("Frozen source changed during publication verification")
    return {
        "sha": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest(),
        "mode": "100755" if mode & 0o111 else "100644",
        "type": "blob",
    }


def bind_source(spec, run, source_root, pr, fetch, verified):
    """Only the declared root is delivered; sibling fixture repositories stay separate."""
    run = Path(run)
    result = read_json(run / "result.json")
    if digest(result) != verified["result_hash"]:
        raise ValueError("Full result changed during publication verification")
    target = repository(pr["url"].split("/pull/")[0])
    head_repo = pr["headRepository"]["nameWithOwner"].lower()
    if spec["kind"] == "native":
        cfg = read_json(run / "config.json")
        identities = result["identities"]
        contract = digest(
            {
                "config": {k: v for k, v in cfg.items() if k != "sources"},
                "adapter": result["adapter_hash"],
                "baseline": identities["baseline"],
            }
        )
        if contract != verified["contract"]:
            raise ValueError("Frozen configuration changed during publication verification")
        if source_root not in identities["candidate"]:
            raise ValueError("Declare a frozen native source_root")
        selected = cfg["implementation_paths"].get(source_root, [])
        if not selected:
            raise ValueError("Source root has no frozen implementation_paths")
        old, new = (identities[role][source_root] for role in ("baseline", "candidate"))
        base_repo, prefix = checkout(cfg["sources"]["baseline"][source_root])
        candidate_repo, candidate_prefix = checkout(cfg["sources"]["candidate"][source_root])
        if (
            base_repo != target
            or candidate_repo not in {target, head_repo}
            or prefix != candidate_prefix
        ):
            raise ValueError("PR repository differs from the selected source root")
        declared_repo = repository(spec["baseline"]["repository"])
        if declared_repo != base_repo:
            # A Megatron root must not be mistaken for the task's primary Core repository.
            other_repos = set()
            for name, path in cfg["sources"]["baseline"].items():
                if name == source_root:
                    continue
                try:
                    other_repos.add(checkout(path)[0])
                except (OSError, ValueError, subprocess.SubprocessError):
                    continue
            if declared_repo not in other_repos:
                raise ValueError("Task baseline repository does not match its source roots")
        baseline_revision = old["git_head"]
        candidate_files, baseline_files = new["files"], old["files"]
        if not any(
            candidate_files.get(name) != baseline_files.get(name)
            for name in candidate_files.keys() | baseline_files.keys()
            if covered(name, selected)
        ):
            raise ValueError("Selected root has no evaluated implementation change")
        candidate_dir = run / "sources/candidate" / source_root
        baseline_dir = run / "sources/baseline" / source_root
        harness = check_paths(cfg, identities)[source_root]

    else:
        if source_root != "task":
            raise ValueError("Generic publication uses source_root=task")
        config_file = run / "source/task.json"
        if (
            hashlib.sha256(config_file.read_bytes()).hexdigest()
            != result["identity"]["files"]["task.json"]
        ):
            raise ValueError("Frozen configuration changed during publication verification")
        cfg = read_json(config_file)
        selected = cfg.get("candidate_files", ["candidate.py"])
        base_repo, prefix = checkout(spec["task_dir"])
        if base_repo != target or repository(spec["baseline"]["repository"]) != target:
            raise ValueError("PR repository differs from the frozen task repository")
        baseline_revision = spec["baseline"]["revision"]
        candidate_files = baseline_files = result["identity"]["files"]
        candidate_dir = baseline_dir = run / "source"
        harness = set()
    base_tree = tree(fetch, target, baseline_revision)
    head_tree = tree(fetch, head_repo, pr["headRefOid"])
    paths = [prefix + name for name in selected]
    # Checking the whole tree also catches extra PR edits outside the bound source root.
    for name in base_tree.keys() | head_tree.keys():
        if not covered(name, paths) and base_tree.get(name) != head_tree.get(name):
            raise ValueError(f"PR changes an unverified file outside implementation_paths: {name}")
    expected = {
        prefix + name: blob(candidate_dir / name, candidate_files[name])
        for name in candidate_files
        if covered(name, selected)
    }
    actual = {name: value for name, value in head_tree.items() if covered(name, paths)}
    if not expected or actual != expected:
        raise ValueError("PR head implementation/helper contents differ from the evaluated source")
    # The published production foundation must match what was evaluated. Trusted test
    # fixture overlays are intentionally not shipped; outside-scope PR edits still fail.
    for name in baseline_files:
        if (
            ".git" in Path(name).parts
            or covered(name, selected)
            or (spec["kind"] == "native" and (protected(name) or covered(name, harness)))
            or Path(name).suffix in {".md", ".rst"}
        ):
            continue
        if head_tree.get(prefix + name) != blob(baseline_dir / name, baseline_files[name]):
            raise ValueError(f"PR fixed baseline source differs from the evaluated source: {name}")
    return {
        "source_root": source_root,
        "repository": target,
        "head_repository": head_repo,
        "baseline_revision": baseline_revision,
        "implementation_paths": paths,
        "implementation_digest": digest(expected),
        "source_digest": digest({"baseline": baseline_files, "candidate": candidate_files}),
    }
