"""Operator commands and the deliberately small coding-agent integration surface."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from kernel_envolve.evaluation.shared.storage import config, contract, digest, file_hash, read_json
from kernel_envolve.manager.config import resource_devices, validate
from kernel_envolve.manager.store import Store


def task_contract(task, kind, baseline=None):
    task = Path(task).resolve()
    if kind == "research":
        from kernel_envolve.manager.research_task import baseline_revision, source_identity

        return source_identity(task, baseline_revision(task, baseline or "HEAD"))
    if baseline is not None:
        raise ValueError("--baseline requires --kind research")
    if kind == "generic":
        return contract(task, config(task))
    from kernel_envolve.evaluation.native.config import load_config
    from kernel_envolve.evaluation.native.evaluator import snapshot

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
        "contract", help="Print an evaluator contract or research source identity"
    )
    identity.add_argument("--task", type=Path, required=True)
    identity.add_argument("--kind", choices=("generic", "native", "research"), default="generic")
    identity.add_argument("--baseline", help="Research baseline Git ref; defaults to HEAD")
    run = commands.add_parser(
        "run", help="Run the single coordinator; round supervisors survive its restart"
    )
    run.add_argument("--once", action="store_true")
    serve = commands.add_parser(
        "serve", help="Loopback dashboard (use SSH forwarding for remote GPUs)"
    )
    serve.add_argument("--port", type=int, default=8787)
    commands.add_parser("status")
    policy = commands.add_parser(
        "policy-install", help="Opt in to fixed 4h + conditional 4h research"
    )
    policy.add_argument("task")
    policy.add_argument("--started-at", type=float, required=True)
    policy.add_argument(
        "--min-gain",
        type=float,
        required=True,
        help="Minimum median latency reduction in full and confirmation",
    )
    policy.add_argument(
        "--deadline-command", required=True, help="Trusted host deadline adapter JSON argv"
    )
    binding = commands.add_parser(
        "policy-bind-delivery",
        help="Link independent native delivery without transferring qualification",
    )
    binding.add_argument("task")
    binding.add_argument("--delivery-task", required=True)
    binding.add_argument("--source-root", required=True)
    for name in ("policy-tick", "policy-show"):
        command = commands.add_parser(name)
        command.add_argument("task")
    narrator = commands.add_parser(
        "narrate", help="Shared asynchronous progress narrator; no research execution"
    )
    narrator.add_argument("--once", action="store_true")
    narrator.add_argument(
        "--exclusive-service",
        action="store_true",
        help="Own narration until exit; existing background narrators skip without stopping research",
    )
    narrator.add_argument("command", nargs=argparse.REMAINDER, help="Tool-free model adapter argv")
    archive = commands.add_parser("archive", help="Save a portable read-only dashboard archive")
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument("--attach", type=Path, action="append", default=[])
    archive.add_argument("--include-logs", action="store_true")
    watch = commands.add_parser(
        "watch-assessment", help="Install a trusted host verifier for independent assessments"
    )
    watch.add_argument(
        "--research", action="store_true", help="Index research claims, never verified correctness"
    )
    watch.add_argument(
        "--validation", action="store_true", help="Index external validation outcomes"
    )
    watch.add_argument("task")
    watch.add_argument("--receipt", type=Path, required=True)
    watch.add_argument("--verifier", type=Path, required=True)
    external = commands.add_parser(
        "watch-validation", help="Follow a source-bound external manager once"
    )
    external.add_argument("task")
    external.add_argument("--source-state", type=Path, required=True)
    external.add_argument("--source-task", required=True)
    publication = commands.add_parser(
        "sync-pr", help="Bind a verified best to its implementation PR"
    )
    publication.add_argument("task")
    publication.add_argument("--run", required=True)
    publication.add_argument("--url", required=True)
    publication.add_argument("--source-root", required=True)
    publication.add_argument("--base-branch", required=True)
    usage = commands.add_parser("import-usage", help="Import scheduler GPU allocation receipts")
    usage.add_argument("--file", type=Path, required=True)
    audit = commands.add_parser("refresh", help="Recheck and index evaluator evidence")
    audit.add_argument("--force", action="store_true")
    audit.add_argument("--task", help="Index only this task, for example after promotion")
    for action in ("pause", "resume"):
        control = commands.add_parser(action)
        control.add_argument("task")
    control = commands.add_parser("configure")
    control.add_argument("task")
    control.add_argument(
        "--changes",
        required=True,
        help="JSON resource/budget/deadline/host_command changes; task must be paused",
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
    bridge = commands.add_parser(
        "bridge", help="One-shot foreground host bridge (not a model runner)"
    )
    bridge.add_argument("--root", type=Path, required=True)
    bridge.add_argument("--idle-seconds", type=float, default=120)
    opening = commands.add_parser("bridge-open", help="Authorize one foreground-host round")
    opening.add_argument("task")
    opening.add_argument("--owner", required=True)
    opening.add_argument("--claim-id", required=True)
    for name in ("bridge-request", "bridge-consume"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--task", required=True)
        command.add_argument("--round", required=True)
        command.add_argument("--owner", required=True)
        command.add_argument("--id", required=True)
        if name == "bridge-request":
            command.add_argument("--file", type=Path)
            command.add_argument(
                "--log-bytes",
                type=int,
                default=4096,
                help="Tail bytes per stream (0–65536); full logs remain at log_files paths",
            )
            command.add_argument(
                "--consume-file",
                type=Path,
                help="Legacy analysis-file format; optional and never required for the next request",
            )
            command.add_argument(
                "--wait-seconds",
                type=float,
                default=30,
                help="Return on completion, waiting at most this many seconds (0–60; default 30)",
            )
        else:
            command.add_argument("--sha256", required=True)
            command.add_argument("--outcome", required=True)
            command.add_argument("--next-action", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == "narrate":
        import signal
        import threading

        from kernel_envolve.manager.dashboard.narrator import command_from_env, serve

        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        command = command or command_from_env()
        if not command:
            raise ValueError("Configure KERNEL_ENVOLVE_NARRATOR_COMMAND or pass an adapter command")
        stopped = threading.Event()

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            serve(
                args.state,
                command,
                stopped,
                once=args.once,
                exclusive_service=args.exclusive_service,
            )
        except KeyboardInterrupt:
            stopped.set()
        finally:
            signal.signal(signal.SIGTERM, previous)
        return 0
    if args.action == "contract":
        print(task_contract(args.task, args.kind, args.baseline))
        return 0
    if args.action == "gpu-run":
        from kernel_envolve.manager.execution.gpu import DurationAdmissionError, run

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
    query = args.action == "bridge-request" and not args.file and not args.consume_file
    store = Store(
        args.state, readonly=query or args.action in ("status", "archive", "serve", "policy-show")
    )
    try:
        if args.action.startswith("policy-"):
            from kernel_envolve.manager import research_policy

            if args.action == "policy-install":
                value = research_policy.install(
                    store,
                    args.task,
                    started_at=args.started_at,
                    min_gain=args.min_gain,
                    deadline_command=json.loads(args.deadline_command),
                )
            elif args.action == "policy-bind-delivery":
                value = research_policy.bind_delivery(
                    store, args.task, args.delivery_task, args.source_root
                )
            elif args.action == "policy-tick":
                value = research_policy.tick(store, args.task)
            else:
                value = research_policy.get(store, args.task)
            print(json.dumps(value, ensure_ascii=False, indent=2))
        elif args.action.startswith("bridge"):
            from kernel_envolve.manager.execution import bridge

            if args.action == "bridge":
                return bridge.run_bridge(store, args.root, args.idle_seconds)
            if args.action == "bridge-open":
                value = bridge.open_host(store, args.task, args.owner, args.claim_id)
            elif args.action == "bridge-request":
                payload = read_json(args.file) if args.file else None
                if args.consume_file:
                    previous = read_json(args.consume_file)
                    fields = {"request", "receipt_sha256", "outcome", "next_action"}
                    if (
                        payload is None
                        or not isinstance(previous, dict)
                        or set(previous) != fields
                        or previous["request"] == args.id
                    ):
                        raise ValueError(
                            "--consume-file requires --file and a distinct prior request, "
                            "receipt_sha256, outcome and next_action"
                        )
                    # Both operations retain their normal idempotency. If submission
                    # fails, the real analysis stays saved and the same call is retryable.
                    bridge.consume(
                        store,
                        args.root,
                        args.task,
                        args.round,
                        args.owner,
                        previous["request"],
                        previous["receipt_sha256"],
                        previous["outcome"],
                        previous["next_action"],
                    )
                value = bridge.request(
                    store,
                    args.root,
                    args.task,
                    args.round,
                    args.owner,
                    args.id,
                    payload,
                    args.wait_seconds,
                    args.log_bytes,
                )
            else:
                value = bridge.consume(
                    store,
                    args.root,
                    args.task,
                    args.round,
                    args.owner,
                    args.id,
                    args.sha256,
                    args.outcome,
                    args.next_action,
                )
            print(json.dumps(value, ensure_ascii=False, indent=2))
        elif args.action == "archive":
            from kernel_envolve.manager.results.archive import export_archive

            print(
                json.dumps(
                    export_archive(
                        store, args.output, attachments=args.attach, include_logs=args.include_logs
                    ),
                    ensure_ascii=False,
                )
            )
        elif args.action == "watch-assessment":
            from kernel_envolve.manager.results.assessment import refresh_sources, watch

            watch(
                store,
                args.task,
                args.receipt,
                args.verifier,
                research=args.research,
                validation=args.validation,
            )
            refresh_sources(store)
        elif args.action == "watch-validation":
            from kernel_envolve.manager.results.assessment import refresh_sources
            from kernel_envolve.manager.results.validation_import import watch_external

            watch_external(store, args.task, args.source_state, args.source_task)
            refresh_sources(store)
        elif args.action == "init":
            deployment = validate(read_json(args.config), args.config.resolve().parent)
            for task in deployment["tasks"]:
                if task["kind"] == "research":
                    # config.validate already checked the immutable source identity.
                    # No evaluator files, imports or source copies are needed.
                    continue
                if task["kind"] == "native":
                    from kernel_envolve.evaluation.native.config import load_config

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
                allowed = ("cpu",) if resource_devices(resource) == ["cpu"] else ("cuda", "cuda:0")
                if not task.get("observe_only") and cfg["device"] not in allowed:
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
            from kernel_envolve.manager.service import clear_resource, resource_sessions

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
            from kernel_envolve.manager.service import run

            run(store, args.once)
        elif args.action == "refresh":
            from kernel_envolve.manager.results.index import refresh

            if not refresh(store, args.force, task_id=args.task):
                raise ValueError("Evidence scan already running; retry after it finishes")
        elif args.action == "serve":
            from kernel_envolve.manager.dashboard.web import serve

            serve(store.root, args.port, readonly=True)
        elif args.action == "sync-pr":
            from kernel_envolve.manager.results.index import refresh
            from kernel_envolve.manager.results.publication import sync_pr

            if not refresh(store, task_id=args.task):
                raise ValueError(
                    "Evidence scan already running; retry publication after it finishes"
                )
            sync_pr(
                store,
                args.task,
                args.run,
                args.url,
                source_root=args.source_root,
                base_branch=args.base_branch,
            )
        elif args.action == "import-usage":
            from kernel_envolve.manager.usage import import_usage

            import_usage(store, read_json(args.file))
        elif args.action == "status":
            from kernel_envolve.manager.dashboard.views import overview

            print(json.dumps(overview(store), ensure_ascii=False, indent=2))
    finally:
        store.close()
    return 0
