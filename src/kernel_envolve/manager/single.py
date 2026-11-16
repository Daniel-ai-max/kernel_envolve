"""Single-task entry point for the existing manager, evidence index and dashboard."""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..storage import config
from .cli import task_contract
from .config import positive, validate
from .process import exclusive
from .store import Store, encode


def parser():
    result = argparse.ArgumentParser(description="Open the existing dashboard for one kernel")
    source = result.add_mutually_exclusive_group()
    source.add_argument("--task", type=Path)
    source.add_argument(
        "--archive", type=Path, help="Open a portable dashboard archive without research or GPU"
    )
    result.add_argument("--state", type=Path, default=os.environ.get("KERNEL_ENVOLVE_STATE"))
    result.add_argument("--workspace", type=Path)
    result.add_argument("--device", help="Physical GPU UUID (auto-select only one visible GPU)")
    result.add_argument("--seconds", type=float, help="New research window; default 3600 seconds")
    result.add_argument("--port", type=int, default=8787)
    result.add_argument("command", nargs=argparse.REMAINDER, help="Optional actual model host argv")
    return result


def device_for(cfg, requested):
    if cfg["device"] == "cpu":
        if requested not in (None, "cpu"):
            raise ValueError("CPU task cannot use a GPU resource")
        return "cpu"
    if cfg["device"] not in ("cuda", "cuda:0"):
        raise ValueError("Managed CUDA tasks must use logical cuda:0")
    if requested:
        return requested
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10
    )
    devices = dict(line.strip().split(", ", 1) for line in inventory.splitlines() if line.strip())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        devices = list(devices.values())
    else:
        devices = [devices.get(value.strip(), value.strip()) for value in visible.split(",")]
        inventory_uuids = {
            line.strip().split(", ", 1)[1] for line in inventory.splitlines() if line.strip()
        }
        if not devices or any(value not in inventory_uuids for value in devices):
            raise ValueError("No unambiguous visible GPU; select an assigned device with --device")
    if len(devices) != 1:
        raise ValueError("Choose one physical GPU with --device GPU-... to start research")
    return devices[0]


def prepare(store, args):
    task_dir = args.task.resolve()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    matches = [
        store.task(row["id"])
        for row in store.rows("SELECT id FROM tasks")
        if Path(store.task(row["id"])["spec"]["task_dir"]) == task_dir
        and not store.task(row["id"])["superseded_by"]
    ]
    if len(matches) > 1:
        raise ValueError("Multiple task identities match; use the existing manager state directly")
    old = matches[0] if matches else None
    if old and (not command or not old["spec"].get("observe_only")):
        if command and old["spec"].get("host_command") != command:
            raise ValueError("Task already has an executor; reuse it without a new command")
        if args.seconds is not None or args.device is not None or args.workspace is not None:
            raise ValueError(
                "Existing task settings and budgets are retained; use manage configure to change them"
            )
        return old["id"]
    if old and old["desired"] != "active":
        raise ValueError("Viewing cannot resume a stopped task; use explicit manager controls")
    if args.seconds is not None:
        positive(args.seconds, "seconds")
    if not command and (args.seconds is not None or args.device is not None):
        raise ValueError("A device or research budget requires an agent command after --")
    native = (task_dir / "native.json").is_file()
    if native and (task_dir / "task.json").exists():
        raise ValueError("Ambiguous task: both task.json and native.json exist")
    if native:
        from ..native_config import load_config

        cfg = load_config(task_dir)
        roots = [task_dir, *(Path(p) for p in cfg["sources"]["candidate"].values())]
    else:
        cfg = config(task_dir)
        roots = [task_dir]
    workspace = (args.workspace or Path(os.path.commonpath(roots))).resolve()
    if workspace == Path(workspace.anchor) or any(
        not p.resolve().is_relative_to(workspace) for p in roots
    ):
        raise ValueError("Choose a dedicated workspace containing the task and candidate roots")
    kind = "native" if native else "generic"
    frozen = task_contract(task_dir, kind)
    if old and (
        old["spec"]["workspace"] != str(workspace) or old["spec"]["baseline"]["contract"] != frozen
    ):
        raise ValueError(
            "Workspace or contract changed; preserve the previous task and establish a new identity"
        )
    device = device_for(cfg, args.device) if command else "unassigned"
    key = old["id"] if old else "kernel-" + hashlib.sha256(str(task_dir).encode()).hexdigest()[:12]
    resource = {
        "id": "device-" + hashlib.sha256(device.encode()).hexdigest()[:12],
        "device": device,
        "sessions": 1,
    }
    for row in store.rows("SELECT id FROM resources"):
        existing = store.resource(row["id"])["spec"]
        if existing.get("device") == device:
            resource = existing
            break
    seconds = args.seconds if args.seconds is not None else 3600
    baseline = (
        old["spec"]["baseline"]
        if old
        else {
            "repository": str(task_dir),
            "revision": frozen,
            "entrypoint": "adapter.py" if native else "task.py:baseline",
            "introduced_on": None,
            "introduction_url": None,
            "frozen_on": dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "hardware": cfg["device"],
            "contract": frozen,
        }
    )
    spec = {
        "id": key,
        "name": cfg["name"],
        "workspace": str(workspace),
        "task_dir": str(task_dir),
        "kind": kind,
        "resource": resource["id"],
        "baseline": baseline,
        "observe_only": not bool(command),
        "command": [],
    }
    if command:
        spec.update(
            host_driven=True,
            host_command=command,
            command=[
                sys.executable,
                "-m",
                "kernel_envolve.cli",
                "manage",
                "bridge",
                "--root",
                str(store.root / "bridge"),
            ],
            wall_budget_seconds=(old["wall_seconds"] if old else 0) + seconds,
            deadline_at=time.time() + seconds,
        )
    deployment = validate({"version": 1, "resources": [resource], "tasks": [spec]}, workspace)
    if old:
        # Only an explicitly requested first executor can upgrade an observation-only task.
        # Keep its identity, history, best, publications, usage and task controls intact.
        store.register({"version": 1, "resources": deployment["resources"], "tasks": []})
        with store.transaction():
            current = store.task(key)
            if (
                not current["spec"].get("observe_only")
                or current["desired"] != "active"
                or store.one(
                    "SELECT id FROM jobs WHERE task=? AND status IN ('queued','running')", (key,)
                )
            ):
                raise ValueError("Task changed during setup; inspect the existing executor")
            store.db.execute(
                "UPDATE tasks SET spec=? WHERE id=?", (encode(deployment["tasks"][0]), key)
            )
            store.event(key, "executor_configured", {"source": "single-task launcher"})
    else:
        store.register(deployment)
    return key


def main(argv=None):
    args = parser().parse_args(argv)
    if args.archive or not args.task:
        if not args.archive and not args.state:
            raise ValueError("Choose --task, --state or --archive")
        if (
            args.command
            or args.seconds is not None
            or args.device
            or args.workspace
            or (args.archive and args.state)
        ):
            raise ValueError("History viewing does not accept execution settings")
        from .web import serve

        serve(args.archive or args.state, args.port, archive=bool(args.archive), readonly=True)
        return 0
    task = args.task.resolve()
    if args.state is None:
        name = hashlib.sha256(str(task).encode()).hexdigest()[:16]
        legacy = Path.home() / ".cache/kernel-envolve/dashboards" / name
        args.state = (
            legacy
            if (legacy / "manager.sqlite").is_file()
            else Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
            / "kernel-envolve/dashboards"
            / name
        )
    store = Store(args.state)
    try:
        with exclusive(store.root / "dashboard.lock") as handle:
            if handle is None:
                raise ValueError("A dashboard launcher already owns this state; reuse its URL")
            # Bind first: an occupied port must not start research or alter task settings.
            from .service import run
            from .web import server

            with server(store.root, args.port) as http:
                with exclusive(store.root / "coordinator.lock") as coordinator:
                    existing = coordinator is None
                if (
                    args.command
                    and not existing
                    and any(
                        Path(store.task(row["id"])["spec"]["task_dir"]) != task
                        and store.task(row["id"])["desired"] == "active"
                        and not store.task(row["id"])["spec"].get("observe_only")
                        for row in store.rows("SELECT id FROM tasks")
                    )
                ):
                    raise ValueError(
                        "Use the existing multi-task coordinator; this launcher will not start unrelated tasks"
                    )
                task_id = prepare(store, args)
                print(
                    json.dumps(
                        {
                            "url": f"http://127.0.0.1:{http.server_port}",
                            "state": str(store.root),
                            "task": task_id,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if existing:
                    try:
                        http.serve_forever()
                    except KeyboardInterrupt:
                        pass
                else:
                    thread = threading.Thread(target=http.serve_forever, daemon=True)
                    thread.start()
                    try:
                        run(store, observe=not bool(args.command))
                    finally:
                        http.shutdown()
                        thread.join()
    finally:
        store.close()
    return 0
