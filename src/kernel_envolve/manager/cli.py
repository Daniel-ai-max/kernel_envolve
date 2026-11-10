"""Operator commands and the deliberately small coding-agent integration surface."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from ..storage import config, contract, digest, file_hash, read_json
from .config import validate
from .store import Store


def task_contract(task, kind):
    task = Path(task).resolve()
    if kind == "generic":
        return contract(task, config(task))
    from ..native import snapshot
    from ..native_config import load_config

    cfg = load_config(task)
    with tempfile.TemporaryDirectory() as folder:
        baseline = {
            name: snapshot(root, Path(folder) / name, cfg["artifacts"].get(name, []))
            for name, root in cfg["sources"]["baseline"].items()
        }
    return digest(
        {
            "config": {k: v for k, v in cfg.items() if k != "sources"},
            "adapter": file_hash(task / "adapter.py"),
            "baseline": baseline,
        }
    )


def parser():
    root = argparse.ArgumentParser(description="Persistent kernel optimization tasks")
    root.add_argument(
        "--state", type=Path, default=os.environ.get("KERNEL_ENVOLVE_STATE", ".kernel_manager")
    )
    commands = root.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init", help="Validate and idempotently register a deployment")
    init.add_argument("config", type=Path)
    identity = commands.add_parser(
        "contract", help="Print the evaluator's current immutable contract"
    )
    identity.add_argument("--task", type=Path, required=True)
    identity.add_argument("--kind", choices=("generic", "native"), default="generic")
    run = commands.add_parser(
        "run", help="Run the single coordinator; round supervisors survive its restart"
    )
    run.add_argument("--once", action="store_true")
    serve = commands.add_parser(
        "serve", help="Loopback dashboard (use SSH forwarding for remote GPUs)"
    )
    serve.add_argument("--port", type=int, default=8787)
    commands.add_parser("status")
    audit = commands.add_parser("refresh", help="Recheck and index evaluator evidence")
    audit.add_argument("--force", action="store_true")
    for action in ("pause", "resume"):
        control = commands.add_parser(action)
        control.add_argument("task")
    control = commands.add_parser("configure")
    control.add_argument("task")
    control.add_argument(
        "--changes", required=True, help="JSON resource/budget changes; task must be paused"
    )
    replacement = commands.add_parser(
        "supersede", help="Retire a task in favor of an existing identity, preserving all usage"
    )
    replacement.add_argument("task")
    replacement.add_argument("--by", required=True)
    resource = commands.add_parser("resource")
    resource.add_argument("id")
    resource.add_argument("--sessions", type=int, choices=(1, 2))
    resource.add_argument(
        "--clear", action="store_true", help="Clear a quarantine after lock/device checks"
    )
    gpu = commands.add_parser("gpu-run", help="Queue a GPU command under the current agent round")
    gpu.add_argument(
        "--expected-seconds",
        type=float,
        help="Declared command duration; reject if round or cumulative budgets cannot fit it",
    )
    gpu.add_argument("command", nargs=argparse.REMAINDER)
    report = commands.add_parser("report", help="Update the current round's research checkpoint")
    report.add_argument("--file", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "contract":
        print(task_contract(args.task, args.kind))
        return 0
    if args.action == "gpu-run":
        from .gpu import DurationAdmissionError, run

        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        try:
            return run(
                args.state,
                os.environ["KERNEL_ENVOLVE_TASK"],
                os.environ["KERNEL_ENVOLVE_ROUND"],
                command,
                expected_seconds=args.expected_seconds,
            )
        except DurationAdmissionError as exc:
            print(json.dumps({"error_type": "gpu_duration_admission", "error": str(exc)}))
            return 2
    store = Store(args.state)
    try:
        if args.action == "init":
            deployment = validate(read_json(args.config), args.config.resolve().parent)
            for task in deployment["tasks"]:
                if task["kind"] == "native":
                    from ..native_config import load_config

                    cfg = load_config(Path(task["task_dir"]))
                    if any(
                        not Path(root).resolve().is_relative_to(task["workspace"])
                        for root in cfg["sources"]["candidate"].values()
                    ):
                        raise ValueError(
                            "All native candidate roots must be inside the task's dedicated workspace"
                        )
                else:
                    cfg = config(Path(task["task_dir"]))
                resource = next(r for r in deployment["resources"] if r["id"] == task["resource"])
                allowed = ("cpu",) if resource["device"] == "cpu" else ("cuda", "cuda:0")
                if cfg["device"] not in allowed:
                    raise ValueError(
                        "Managed tasks use cpu on CPU resources or cuda:0 on the assigned GPU"
                    )
                if task_contract(task["task_dir"], task["kind"]) != task["baseline"]["contract"]:
                    raise ValueError(f"Frozen contract does not match task {task['id']}")
            store.register(deployment)
        elif args.action in ("pause", "resume", "configure"):
            store.control(
                args.task,
                args.action,
                json.loads(args.changes) if args.action == "configure" else None,
            )
        elif args.action == "supersede":
            print(json.dumps(store.supersede(args.task, args.by), ensure_ascii=False, indent=2))
        elif args.action == "resource":
            from .service import clear_resource, resource_sessions

            if args.clear:
                clear_resource(store, args.id)
            if args.sessions is not None:
                resource_sessions(store, args.id, args.sessions)
        elif args.action == "report":
            store.report(
                os.environ["KERNEL_ENVOLVE_TASK"],
                os.environ["KERNEL_ENVOLVE_ROUND"],
                read_json(args.file),
            )
        elif args.action == "run":
            from .service import run

            run(store, args.once)
        elif args.action == "refresh":
            from .evidence import refresh

            if not refresh(store, args.force):
                raise ValueError("Evidence scan already running; retry after it finishes")
        elif args.action == "serve":
            from .web import serve

            serve(store.root, args.port)
        elif args.action == "status":
            from .views import overview

            print(json.dumps(overview(store), ensure_ascii=False, indent=2))
    finally:
        store.close()
    return 0
