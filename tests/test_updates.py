"""Real local remotes exercise publication, offline fallback and pinned subprocesses."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_manager import launch, replace_command
from test_manager import managed as managed

from kernel_envolve import updates


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root, text="next"):
    (root / "revision.txt").write_text(text)
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", text)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / "upstream"
    repo.mkdir()
    for name in ("src", "skills"):
        shutil.copytree(source / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(source / "pyproject.toml", repo / "pyproject.toml")
    git(repo, "init", "--quiet", "-b", "main")
    git(repo, "config", "user.name", "Updater test")
    git(repo, "config", "user.email", "updater@example.test")
    commit(repo, "initial")
    monkeypatch.setenv("KERNEL_ENVOLVE_AUTO_UPDATE", "1")
    monkeypatch.setenv("KERNEL_ENVOLVE_UPDATE_REMOTE", str(repo))
    monkeypatch.setenv("KERNEL_ENVOLVE_UPDATE_CACHE", str(tmp_path / "cache with spaces"))
    return repo


def cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "kernel_envolve.cli", *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=40,
    )


def test_publish_latest_without_touching_downstream_checkout(upstream):
    first = updates.refresh()
    assert first["status"] == "current", first
    old = Path(first["release"])
    revision = commit(upstream)
    # A downstream checkout may have both a feature branch and local edits.
    downstream = upstream.parent / "downstream"
    subprocess.run(["git", "clone", "--quiet", str(upstream), str(downstream)], check=True)
    git(downstream, "checkout", "-b", "local-work")
    (downstream / "revision.txt").write_text("uncommitted")
    second = updates.refresh()
    assert second["revision"] == revision
    assert (old / "revision.txt").read_text() == "initial"
    assert (downstream / "revision.txt").read_text() == "uncommitted"
    assert git(downstream, "branch", "--show-current") == "local-work"
    assert updates.refresh()["revision"] == revision
    assert len(list(old.parent.iterdir())) == 2


def test_offline_retains_last_good_and_first_use_can_fall_back(upstream):
    good = updates.refresh()
    upstream.rename(upstream.with_name("offline"))
    fallback = updates.refresh()
    assert fallback["status"] == "cached"
    assert fallback["revision"] == good["revision"]
    assert "error" in fallback
    assert json.loads(cli("update", "--status").stdout)["status"] == "cached"
    root, _ = updates.cache_root()
    (root / "current.json").unlink()
    assert updates.refresh()["status"] == "installed"


@pytest.mark.parametrize("broken", ["dependency", "python", "code", "skill"])
def test_bad_main_is_not_activated(upstream, broken):
    good = updates.refresh()
    project = upstream / "pyproject.toml"
    if broken == "dependency":
        project.write_text(project.read_text().replace("numpy>=1.26", "numpy>=9999"))
    elif broken == "python":
        project.write_text(project.read_text().replace(">=3.11", ">=99.0"))
    elif broken == "code":
        (upstream / "src/kernel_envolve/cli.py").write_text("invalid syntax !")
    else:
        (upstream / "skills/kernel-envolve/SKILL.md").unlink()
    commit(upstream, broken)
    selected = updates.refresh()
    assert selected["status"] == "cached", selected
    assert selected["revision"] == good["revision"]
    root, _ = updates.cache_root()
    assert updates.current(root)["revision"] == good["revision"]


def test_rewritten_main_does_not_roll_back(upstream):
    initial = git(upstream, "rev-parse", "HEAD")
    commit(upstream)
    good = updates.refresh()
    git(upstream, "reset", "--hard", initial)
    fallback = updates.refresh()
    assert fallback["revision"] == good["revision"]
    assert "rewritten" in fallback["error"]


def test_concurrent_agents_publish_one_complete_release(upstream):
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "kernel_envolve.cli", "update"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    results = []
    for process in processes:
        out, err = process.communicate(timeout=40)
        assert process.returncode == 0, err
        result = json.loads(out)
        assert result["status"] == "current", result
        assert Path(result["skill"]).is_file()
        results.append(result)
    assert len({r["revision"] for r in results}) == 1
    assert len(list(Path(results[0]["release"]).parent.iterdir())) == 1


def test_session_pins_running_process_cli_and_skill(upstream):
    script = (
        "import json,sys,subprocess; from kernel_envolve.storage import evaluator_hash; "
        "from kernel_envolve.updates import selected; "
        "print(json.dumps([selected(), evaluator_hash()]), flush=True); "
        "sys.stdin.readline(); "
        "print(json.dumps([selected(), evaluator_hash()]), flush=True); "
        "subprocess.run(['kernel-envolve','update'],check=True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "kernel_envolve.cli", "session", "--", sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        first = json.loads(process.stdout.readline())
        source = upstream / "src/kernel_envolve/scoring.py"
        source.write_text(source.read_text() + "\n# New evaluator revision\n")
        new = commit(upstream)
        assert updates.refresh()["revision"] == new
        out, err = process.communicate("continue\n", timeout=10)
        assert process.returncode == 0, err
        second, child = out.split("\n", 1)
        assert json.loads(second) == first
        assert json.loads(child)["revision"] == first[0]["revision"] != new
        assert Path(first[0]["skill"]).is_file()
        assert json.loads(cli("update").stdout)["revision"] == new
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_cli_automatically_reexecutes_new_main(upstream):
    # Add a diagnostic command only to the remote to prove dispatch uses new code.
    source = upstream / "src/kernel_envolve/cli.py"
    source.write_text(
        source.read_text().replace(
            '    if argv and argv[0] == "manage":',
            "    if argv == ['probe-main']:\n        print('from main')\n        return 0\n"
            '    if argv and argv[0] == "manage":',
        )
    )
    commit(upstream)
    result = cli("probe-main")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from main"


def test_copied_skill_resolves_current_pinned_entry(upstream):
    result = cli("update")
    assert result.returncode == 0, result.stderr
    version = json.loads(result.stdout)
    commit(upstream)
    pinned = subprocess.run(
        [version["executable"], "update"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert pinned.returncode == 0, pinned.stderr
    assert json.loads(pinned.stdout)["revision"] == version["revision"]
    assert json.loads(cli("update").stdout)["revision"] != version["revision"]


def test_disable_and_status_do_not_contact_remote(upstream, monkeypatch):
    monkeypatch.setenv("KERNEL_ENVOLVE_AUTO_UPDATE", "0")
    assert updates.refresh() == {"status": "disabled"}
    result = cli("update", "--status")
    assert json.loads(result.stdout)["status"] == "installed"
    root, _ = updates.cache_root()
    assert not root.exists()


def test_environment_replaces_old_release_without_losing_training_paths():
    env = updates.environment(
        "/new",
        {
            updates.PIN: "/old",
            "PYTHONPATH": "/old/src:/training/core",
            "PATH": "/old/.entry:/bin",
        },
    )
    assert env["PYTHONPATH"] == "/new/src:/training/core"
    assert env["PATH"] == "/new/.entry:/bin"


def test_fetch_timeout_keeps_active_version(upstream, monkeypatch):
    good = updates.refresh()
    original = updates.command

    def timeout(argv, **kwargs):
        if "fetch" in argv:
            raise subprocess.TimeoutExpired(argv, 15)
        return original(argv, **kwargs)

    monkeypatch.setattr(updates, "command", timeout)
    assert updates.refresh()["revision"] == good["revision"]


def test_interrupted_unpublished_preparation_recovers(upstream):
    root, _ = updates.cache_root()
    revision = git(upstream, "rev-parse", "HEAD")
    partial = root / "releases" / revision
    partial.mkdir(parents=True)
    (partial / "incomplete").touch()
    result = updates.refresh()
    assert result["status"] == "current", result
    assert not (partial / "incomplete").exists()


def test_first_offline_session_does_not_update_mid_round(upstream):
    upstream.rename(upstream.with_name("offline"))
    script = (
        "import os,json,sys; from kernel_envolve import updates; "
        "print('ready',flush=True); sys.stdin.readline(); "
        "print(json.dumps(updates.selected()))"
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "kernel_envolve.cli", "session", "--", sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        upstream.with_name("offline").rename(upstream)
        out, err = process.communicate("continue\n", timeout=10)
        assert process.returncode == 0, err
        assert json.loads(out)["status"] == "pinned-installed"
        root, _ = updates.cache_root()
        assert not updates.current(root)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_managed_round_refreshes_and_gpu_child_inherits_version(upstream, managed):
    store, _, _ = managed
    gpu_child = "from kernel_envolve.updates import selected; print(selected()['revision'])"
    script = (
        "import os,json,pathlib,sys,subprocess; from kernel_envolve.updates import selected; "
        "v=selected(); prompt=sys.stdin.read(); assert v['revision'] in prompt; "
        "assert v['skill'] in prompt; "
        f"out=subprocess.check_output(['kernel-envolve','manage','gpu-run','--',sys.executable,'-c',{gpu_child!r}],text=True); "
        "assert out.strip()==v['revision']; "
        "pathlib.Path(os.environ['KERNEL_ENVOLVE_REPORT']).write_text(json.dumps({'next_action':v['revision']}))"
    )
    replace_command(store, [sys.executable, "-c", script])
    for index in range(2):
        revision = commit(upstream, f"round-{index}")
        key = store.schedule()[0]
        process = launch(store, key)
        out, err = process.communicate(timeout=20)
        assert process.returncode == 0, (out, err, store.job(key))
        assert store.task("toy")["report"]["next_action"] == revision
        # Remove this fixture's cooldown, not its previous receipt/history.
        store.db.execute("UPDATE tasks SET next_at=0 WHERE id='toy'")


def test_optimized_python_flag_survives_update_restart(upstream):
    result = subprocess.run(
        [sys.executable, "-O", "-m", "kernel_envolve.cli", "evaluate", "--task", str(upstream)],
        text=True,
        capture_output=True,
        timeout=40,
    )
    assert result.returncode != 0
    assert "without -O" in result.stderr
