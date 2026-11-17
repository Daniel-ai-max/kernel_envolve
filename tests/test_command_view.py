"""Command labels preserve argv semantics without interpreting executable text."""

import json
import shlex

import pytest

from kernel_envolve.manager.command_view import command_summary
from kernel_envolve.manager.views import task_view

pytest_plugins = ["test_manager"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["python", "diagnose.py", "--mode", "quick"],
            ["python", "diagnose.py", "--mode", "quick"],
        ),
        (
            ["/opt/venv/bin/python3.12", "-u", "-m", "example.cli", "evaluate", "--mode=quick"],
            ["python3.12", "-u", "-m", "example.cli", "evaluate", "--mode=quick"],
        ),
        (
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=3s",
                "400.0",
                "env",
                "--chdir=/work/example",
                "/opt/venv/bin/python",
                "/work/example/probe.py",
                "--output",
                "/work/results/probe.json",
                "--mode",
                "full",
            ],
            ["python", "…/probe.py", "--output", "…/probe.json", "--mode", "full"],
        ),
        (
            [
                "/usr/bin/env",
                "-i",
                "-u",
                "OLD_SETTING",
                "-C",
                "/work",
                "COUNT=2",
                "timeout",
                "-s",
                "TERM",
                "-k",
                "3s",
                "--",
                "2m",
                "python",
                "probe.py",
            ],
            ["python", "probe.py"],
        ),
        (
            ["env", "--", "COUNT=2", "python", "probe.py", "--root=/work/example"],
            ["python", "probe.py", "--root=…/example"],
        ),
        (["/usr/bin/git", "diff", "--stat"], ["git", "diff", "--stat"]),
        (
            ["python", "/work/a b/probe.py", "--name", "two words", "--root", "/"],
            ["python", "…/probe.py", "--name", "two words", "--root", "/"],
        ),
        (
            ["python", "-c", "/literal/path; print('not a script')", "--mode", "quick"],
            ["python", "-c", "/literal/path; print('not a script')", "--mode", "quick"],
        ),
    ],
)
def test_recognizable_commands(argv, expected):
    before = argv.copy()
    assert shlex.split(command_summary(argv)) == expected
    assert argv == before


@pytest.mark.parametrize(
    "argv",
    [
        ["timeout", "--unknown", "3", "python", "probe.py"],
        ["timeout", "--signal"],
        ["timeout", "--signal=", "3", "python", "probe.py"],
        ["timeout", "not-a-duration", "python", "probe.py"],
        ["env", "-S", "python probe.py"],
        ["env", "--chdir"],
        ["env", "COUNT=2"],
        ["custom-wrapper", "--", "python", "probe.py"],
        ["/bin/sh", "-c", "python /work/probe.py; echo $(date)"],
    ],
)
def test_unknown_or_incomplete_wrappers_stay_literal(argv):
    assert command_summary(argv) == shlex.join(argv)


def test_long_paths_do_not_hide_the_script_or_subcommand():
    root = "/work/" + "long-directory/" * 60
    argv = [
        "env",
        "--chdir=" + root,
        "/opt/venv/bin/python",
        root + "probe.py",
        "evaluate",
        "--output=" + root + "result.json",
        "--mode",
        "quick",
    ]
    assert shlex.split(command_summary(argv)) == [
        "python",
        "…/probe.py",
        "evaluate",
        "--output=…/result.json",
        "--mode",
        "quick",
    ]


def test_opaque_shell_and_python_source_are_never_executed(tmp_path):
    marker = tmp_path / "unexpected"
    argv = ["python", "-c", f"__import__('pathlib').Path({str(marker)!r}).touch()"]
    assert command_summary(argv) == shlex.join(argv)
    assert not marker.exists()
    assert len(command_summary(["sh", "-c", "x" * 1000])) == 500
    assert command_summary(["sh", "-c", "x" * 1000]).endswith("…")


def test_gpu_view_keeps_full_argv_and_job_record(managed):
    store, _, _ = managed
    argv = [
        "timeout",
        "--kill-after=3s",
        "30",
        "env",
        "--chdir=/work/example",
        "/opt/venv/bin/python",
        "/work/example/probe.py",
        "--mode",
        "quick",
    ]
    job = store.enqueue(store.task("toy"), "gpu", argv)
    view = task_view(store, "toy")
    assert shlex.split(view["current_work"]) == ["python", "…/probe.py", "--mode", "quick"]
    assert view["current_work_argv"] == argv
    assert json.loads(store.job(job)["command"]) == argv
