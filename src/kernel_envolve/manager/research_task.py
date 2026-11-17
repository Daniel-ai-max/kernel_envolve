"""Source provenance for exploration, without an evaluator or source-tree copy."""

import subprocess
from pathlib import Path

from kernel_envolve.evaluation.shared.storage import digest


def baseline_revision(workspace, revision="HEAD"):
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise ValueError("Research workspace must be an existing Git checkout")
    try:
        root = subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        ).strip()
        if Path(root).resolve() != workspace:
            raise ValueError("Research workspace must be its own Git root")
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(workspace),
                "rev-parse",
                "--verify",
                "--end-of-options",
                revision + "^{commit}",
            ],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        ).strip()
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError("Research baseline must resolve to an existing local Git commit") from exc


def source_identity(workspace, revision):
    """This hash is a scheduling identity, never a correctness contract."""
    return digest(
        {
            "kind": "research",
            "version": 1,
            "workspace": str(Path(workspace).resolve()),
            "baseline_revision": revision,
        }
    )


def validate_task(task):
    if task["task_dir"] != task["workspace"]:
        raise ValueError("Research task directory must be its source workspace")
    baseline = task["baseline"]
    revision = baseline_revision(task["workspace"], baseline["revision"])
    if revision != baseline["revision"]:
        raise ValueError("Research baseline must be pinned to a full commit ID")
    if baseline["contract"] != source_identity(task["workspace"], revision):
        raise ValueError("Research source identity does not match its frozen baseline")
