"""Build native checks from a fixed harness and an explicit implementation overlay."""

import posixpath
import shutil
import time
from pathlib import Path

from .storage import file_hash

# These are validation code/configuration, never operator implementations. Checking
# descendants also prevents a broad package overlay from replacing its references.
PROTECTED_PARTS = {"tests", "test", "testcases", "verification", "reference", "references"}
PROTECTED_FILES = {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
CODE_SUFFIXES = {".py", ".so", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp", ".ptx", ".cubin"}


def covered(name, paths):
    return any(name == path or name.startswith(path + "/") for path in paths)


def protected(name):
    path = Path(name)
    return (
        bool(set(path.parts) & PROTECTED_PARTS)
        or path.name in PROTECTED_FILES
        or (
            path.suffix == ".py"
            and (path.name.startswith("test_") or path.name.endswith("_test.py"))
        )
    )


def validate_paths(cfg):
    paths = cfg["implementation_paths"]
    if not isinstance(paths, dict) or not paths or not any(paths.values()):
        raise ValueError(
            "Declare implementation_paths for native checks; create a new task contract"
        )
    if set(paths) - cfg["sources"]["baseline"].keys():
        raise ValueError("Unknown implementation_paths root")
    for root, selected in paths.items():
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(p, str) for p in selected)
        ):
            raise ValueError("implementation_paths needs nonempty lists of relative paths")
        for name in selected:
            path = Path(name)
            if (
                not name
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or name == "."
                or path.as_posix() != name
            ):
                raise ValueError("implementation_paths must be canonical relative paths without ..")
            if protected(name) or ".git" in path.parts:
                raise ValueError(f"Implementation path overlaps trusted harness: {root}/{name}")
            if sum(covered(name, [other]) or covered(other, [name]) for other in selected) != 1:
                raise ValueError("implementation_paths entries must not overlap")
            for role in ("baseline", "candidate"):
                base = Path(cfg["sources"][role][root])
                if any(
                    (base / Path(*path.parts[:i])).is_symlink()
                    for i in range(1, len(path.parts) + 1)
                ):
                    raise ValueError(f"Implementation path contains a symlink: {root}/{name}")


def check_paths(cfg, identities):
    """Protect explicit scripts, pytest selectors and Python module entry points."""
    protected_paths = {root: set() for root in identities["baseline"]}
    for check in cfg["checks"]:
        argv = check.get("argv", [])
        modules = ["apodex_core.verification.method.gate"] if check["kind"] == "gate" else []
        for index, token in enumerate(argv[:-1]):
            if token in {"-m", "--module"}:
                module = argv[index + 1]
                if all(part.isidentifier() for part in module.split(".")):
                    modules.append(module)
                if module == "pytest":
                    break  # Subsequent -m arguments are pytest marker expressions.
        for module in modules:
            parts = module.split(".")
            entry = "/".join(parts)
            names = {entry + ".py", entry + "/__main__.py"}
            names.update("/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts) + 1))
            # The module can resolve from any declared PYTHONPATH root.
            for root in protected_paths:
                protected_paths[root].update(names & identities["baseline"][root]["files"].keys())
        for token in argv:
            value = token.split("=", 1)[-1].split("::", 1)[0]
            root = check["root"]
            if Path(value).is_absolute() and any(
                Path(value).is_relative_to(Path(source))
                for roots in cfg["sources"].values()
                for source in roots.values()
            ):
                raise ValueError(
                    "Native checks must use validation-relative paths or {root} placeholders"
                )
            for name in protected_paths:
                prefix = "{" + name + "}/"
                if value.startswith(prefix):
                    root, value = name, value[len(prefix) :]
                    break
            value = posixpath.normpath(value)
            if value and any(covered(n, [value]) for n in identities["baseline"][root]["files"]):
                protected_paths[root].add(value)
    return protected_paths


def validate_overlay(cfg, identities):
    """Reject code drift outside the reviewed implementation scope before workers start."""
    test_paths = check_paths(cfg, identities)
    for root, baseline in identities["baseline"].items():
        old = baseline["files"]
        new = identities["candidate"][root]["files"]
        selected = cfg["implementation_paths"].get(root, [])
        for path in selected:
            if any(covered(test, [path]) or covered(path, [test]) for test in test_paths[root]):
                raise ValueError(f"Implementation path overlaps declared check code: {root}/{path}")
            matches = [n for n in old.keys() | new.keys() if covered(n, [path])]
            if not matches:
                raise ValueError(f"Implementation path has no source files: {root}/{path}")
            if any(protected(n) or Path(n).suffix not in CODE_SUFFIXES for n in matches):
                raise ValueError(
                    f"Implementation path overlaps harness/configuration: {root}/{path}"
                )
        for name in old.keys() | new.keys():
            if old.get(name) != new.get(name) and not covered(name, selected):
                # Documentation may accompany a candidate; executable/configuration
                # differences cannot be hidden from the native validation view.
                if Path(name).suffix not in {".md", ".rst"}:
                    raise ValueError(
                        f"Candidate changed a protected harness/source file: {root}/{name}"
                    )


def prepare_harness(run, cfg, identities, checkpoint=None):
    """Copy the frozen baseline/implementation union once, within the run budget."""
    started = time.perf_counter()
    checkpoint = checkpoint or (lambda: None)
    checkpoint()
    result = {}
    for root, baseline in identities["baseline"].items():
        dest = run / "sources" / "validation" / root
        selected = cfg["implementation_paths"].get(root, [])
        files = {}
        dest.mkdir(parents=True)
        directories = {dest}
        for role in ("baseline", "candidate"):
            source_root = run / "sources" / role / root
            for name, expected in identities[role][root]["files"].items():
                checkpoint()
                if covered(name, selected) != (role == "candidate"):
                    continue
                source, target = source_root / name, dest / name
                if source_root.is_symlink() or any(
                    (source_root / parent).is_symlink()
                    for parent in (Path(name), *Path(name).parents)
                ):
                    raise ValueError(f"Symlink introduced in frozen source: {role}/{root}/{name}")
                if target.parent not in directories:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    directories.add(target.parent)
                shutil.copy2(source, target)
                checkpoint()
                actual = file_hash(target)
                if actual != expected:
                    raise ValueError(f"Frozen source changed: {role}/{root}/{name}")
                files[name] = actual
        result[root] = {
            "git_head": baseline["git_head"],
            "candidate_git_head": identities["candidate"][root]["git_head"],
            "implementation_paths": selected,
            "files": files,
        }
    checkpoint()
    return result, time.perf_counter() - started
