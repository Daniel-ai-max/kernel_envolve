"""Use immutable main snapshots without modifying the installed checkout or environment."""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

REMOTE = "https://github.com/Daniel-ai-max/kernel_envolve.git"
PIN = "KERNEL_ENVOLVE_RELEASE"
SESSION = "KERNEL_ENVOLVE_UPDATE_SESSION"


def enabled():
    return os.environ.get("KERNEL_ENVOLVE_AUTO_UPDATE", "1") != "0"


def cache_root():
    remote = os.environ.get("KERNEL_ENVOLVE_UPDATE_REMOTE", REMOTE)
    identity = json.dumps([remote, sys.executable, sys.prefix, sys.version])
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    base = Path(
        os.environ.get("KERNEL_ENVOLVE_UPDATE_CACHE", Path.home() / ".cache/kernel-envolve")
    )
    return base.expanduser().resolve() / key, remote


def read(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def command(argv, *, env=None, timeout=15):
    process = subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = process.communicate(timeout=timeout)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if process.returncode:
        # Git output can contain credentials in a configured mirror URL.
        raise ValueError(f"{Path(argv[0]).name} failed (exit {process.returncode})")
    return out.strip()


@contextlib.contextmanager
def update_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "update.lock").open("a") as handle:
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError("update lock timed out")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment(release, base=None):
    env = dict(os.environ if base is None else base)
    old = env.get(PIN)
    paths = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if old:
        paths = [p for p in paths if p != str(Path(old) / "src")]
    source = str(Path(release) / "src")
    env["PYTHONPATH"] = os.pathsep.join([source, *[p for p in paths if p != source]])
    env[PIN] = str(release)
    bins = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    if old:
        bins = [p for p in bins if p != str(Path(old) / ".entry")]
    env["PATH"] = os.pathsep.join([str(Path(release) / ".entry"), *bins])
    return env


def compatible(release):
    project = tomllib.loads((release / "pyproject.toml").read_text())["project"]
    version = ".".join(map(str, sys.version_info[:3]))
    if version not in SpecifierSet(project["requires-python"]):
        raise ValueError(f"main requires Python {project['requires-python']}")
    for spec in project.get("dependencies", []):
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        if req.url:
            raise ValueError(f"Install the new dependency explicitly: {req.name}")
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or installed not in req.specifier:
            raise ValueError(f"main requires {spec}; update this Python environment first")


def describe(release):
    release = Path(release)
    return {
        "revision": release.name,
        "release": str(release),
        "executable": str(release / ".entry/kernel-envolve"),
        "skill": str(release / "skills/kernel-envolve/SKILL.md"),
    }


def prepare(root, revision):
    release = root / "releases" / revision
    if (release / ".ready.json").exists():
        compatible(release)
        return describe(release)
    if release.exists():
        shutil.rmtree(release)  # Only an unpublished, interrupted preparation.
    release.parent.mkdir(parents=True, exist_ok=True)
    command(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(root / "repo.git"),
            str(release),
        ]
    )
    command(
        [
            "git",
            "-C",
            str(release),
            "-c",
            "core.hooksPath=/dev/null",
            "checkout",
            "--quiet",
            "--detach",
            revision,
        ]
    )
    compatible(release)
    if not (release / "src/kernel_envolve/updates/__init__.py").is_file():
        raise ValueError("main does not yet contain the automatic-update entry point")
    if not (release / "skills/kernel-envolve/SKILL.md").is_file():
        raise ValueError("main has no kernel-envolve Skill")
    entry = release / ".entry"
    entry.mkdir()
    # Shell quoting supports interpreters/cache paths containing spaces. The small
    # launcher uses this caller's Python, retaining its CUDA stack and native extensions.
    runner = entry / "run.py"
    runner.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(release / 'src')!r})\n"
        "from kernel_envolve.updates import environment\n"
        f"os.environ.update(environment({str(release)!r}))\n"
        "from kernel_envolve.cli import main\n"
        "raise SystemExit(main())\n"
    )
    executable = entry / "kernel-envolve"
    executable.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -P {shlex.quote(str(runner))} "$@"\n'
    )
    executable.chmod(0o755)
    command([str(executable), "--help"], env=environment(release), timeout=30)
    result = describe(release)
    write(release / ".ready.json", result)
    return result


def current(root):
    value = read(root / "current.json")
    revision = value.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        return {}
    release = root / "releases" / revision
    return describe(release) if (release / ".ready.json").exists() else {}


def fetch(root, remote):
    previous = current(root)
    repo = root / "repo.git"
    git = ["git", "--git-dir", str(repo)]
    if not repo.exists():
        command(["git", "init", "--bare", "--quiet", str(repo)])
    env = os.environ | {"GIT_TERMINAL_PROMPT": "0"}
    command(
        [*git, "fetch", "--quiet", "--no-tags", remote, "+refs/heads/main:refs/heads/main"],
        env=env,
    )
    revision = command([*git, "rev-parse", "refs/heads/main"])
    if previous and revision != previous["revision"]:
        try:
            command([*git, "merge-base", "--is-ancestor", previous["revision"], revision])
        except ValueError as exc:
            raise ValueError("main history was rewritten; retaining the active revision") from exc
    result = prepare(root, revision) | {"status": "current", "checked_at": time.time()}
    write(root / "current.json", result)
    return result


def fallback(root, exc):
    previous = current(root)
    # A changed Python environment must not reactivate an incompatible cached release.
    if previous:
        try:
            compatible(Path(previous["release"]))
        except (OSError, ValueError, KeyError):
            previous = {}
    error = "update command timed out" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
    result = previous | {
        "status": "cached" if previous else "installed",
        "error": error,
        "checked_at": time.time(),
    }
    print(
        f"kernel-envolve: update unavailable; using {result['status']} version ({error})",
        file=sys.stderr,
    )
    return result


def refresh():
    """Check main at a session boundary; atomically activate only a tested snapshot."""
    if not enabled():
        return {"status": "disabled"}
    root, remote = cache_root()
    errors = (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError)
    try:
        with update_lock(root):
            try:
                result = fetch(root, remote)
            except errors as exc:
                result = fallback(root, exc)
            write(root / "status.json", result)
            return result
    except errors as exc:
        return fallback(root, exc)


def selected():
    if os.environ.get(PIN):
        return describe(os.environ[PIN]) | {"status": "pinned"}
    if os.environ.get(SESSION):
        return {"status": "pinned-installed"}
    return refresh()


def enter(argv):
    """Called before dispatch; workers and library imports never refresh themselves."""
    if argv[:1] == ["update"]:
        parser = argparse.ArgumentParser(prog="kernel-envolve update")
        parser.add_argument("--status", action="store_true", help="Inspect without fetching")
        args = parser.parse_args(argv[1:])
        if args.status and not os.environ.get(PIN) and not os.environ.get(SESSION):
            root, _ = cache_root()
            result = read(root / "status.json") or {"status": "installed"}
        else:
            result = selected()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if argv[:1] == ["session"]:
        parser = argparse.ArgumentParser(prog="kernel-envolve session")
        parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args(argv[1:])
        child = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not child:
            parser.error("provide an agent or shell command after --")
        release = selected().get("release")
        env = environment(release) if release else os.environ.copy()
        env[SESSION] = "1"
        os.execvpe(child[0], child, env)
    if (
        not argv
        or argv[0] in ("--help", "-h")
        or os.environ.get(PIN)
        or os.environ.get(SESSION)
        or not enabled()
    ):
        return None
    release = refresh().get("release")
    if release:
        # Preserve -O, -B and other interpreter flags across the clean restart.
        command = [
            sys.executable,
            *subprocess._args_from_interpreter_flags(),
            "-P",
            str(Path(release) / ".entry/run.py"),
            *argv,
        ]
        os.execvpe(sys.executable, command, environment(release))
    return None
