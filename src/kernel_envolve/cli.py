"""Command-line parsing and routing; evaluation lives in evaluation/."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from kernel_envolve.evaluation.generic.evaluator import audit_result as audit_result
from kernel_envolve.evaluation.generic.evaluator import evaluate as evaluate
from kernel_envolve.evaluation.generic.evaluator import promote as promote
from kernel_envolve.evaluation.shared.execution import lock as lock
from kernel_envolve.evaluation.shared.execution import managed_lock_fd


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # RSI state binds its validator version; use the caller's pinned installation.
    if argv and argv[0] == "rsi":
        from kernel_envolve.rsi.cli import main as rsi

        return rsi(argv[1:])
    from kernel_envolve.updates import enter

    result = enter(argv)
    if result is not None:
        return result
    if argv and argv[0] == "dashboard":
        from kernel_envolve.manager.single import main as dashboard

        try:
            return dashboard(argv[1:])
        except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            print(f"kernel-envolve dashboard: {exc}", file=sys.stderr)
            return 2
    if argv and argv[0] == "manage":
        from kernel_envolve.manager.cli import main as manage

        try:
            return manage(argv[1:])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(f"kernel-envolve manage: {exc}", file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description="Reproducible kernel iteration evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update", help="Check main and print the current revision and Skill path")
    sub.add_parser("session", help="Run an agent or shell with one fixed tool revision")
    sub.add_parser(
        "manage", help="Supervise persistent tasks, GPU queues and the progress dashboard"
    )
    sub.add_parser("rsi", help="Learn and validate scoped research strategies in a sidecar")
    sub.add_parser("dashboard", help="Open the existing dashboard for a single kernel")
    ev = sub.add_parser("evaluate", help="Check and measure a frozen task snapshot")
    ev.add_argument("--task", type=Path, required=True)
    ev.add_argument("--mode", choices=("quick", "full"), default="quick")
    ev.add_argument("--signal-only", action="store_true")
    ev.add_argument(
        "--case", action="append", dest="cases", help="Diagnostic case id; repeatable, quick only"
    )
    ev.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Diagnostic seed; repeatable, quick only",
    )
    ev.add_argument(
        "--check-only",
        action="store_true",
        help="Skip timing for correctness debugging; quick only",
    )
    ev.add_argument(
        "--baseline", action="store_true", help="Compare fixed baseline, even if best exists"
    )
    ev.add_argument("--timeout", type=float, default=300)
    pr = sub.add_parser(
        "promote", help="Recheck evidence and atomically point best at its snapshot"
    )
    pr.add_argument("run", type=Path)
    native = sub.add_parser("native-evaluate", help="Evaluate a frozen internal training task")
    native.add_argument("--task", type=Path, required=True)
    native.add_argument("--mode", choices=("quick", "full"), default="quick")
    native.add_argument("--timeout", type=float, help="Seconds; defaults to 120 quick / 1800 full")
    native.add_argument("--case", action="append", dest="cases", help="Diagnostic case; quick only")
    native.add_argument(
        "--seed", action="append", dest="seeds", type=int, help="Diagnostic seed; quick only"
    )
    native.add_argument(
        "--check-only",
        action="store_true",
        help="Skip timing; quick checks selected inputs, full checks all inputs and native gates",
    )
    confirmation = sub.add_parser(
        "native-confirm", help="Consume the frozen independent confirmation set once"
    )
    confirmation.add_argument("run", type=Path)
    confirmation.add_argument("--timeout", type=float, default=1800)
    native_pr = sub.add_parser("native-promote", help="Verify native evidence and preserve best")
    native_pr.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("evaluate", "native-evaluate", "native-confirm"):
            # Detect broken adapters before copying sources or starting workers.
            managed_lock_fd()
        if args.command == "native-confirm":
            from kernel_envolve.evaluation.native.confirmation import confirm_native

            receipt = confirm_native(args.run, args.timeout)
            print(json.dumps(receipt, indent=2))
            return (
                2
                if receipt.get("execution") == "error"
                else 0
                if receipt["status"] == "passed"
                else 1
            )
        if args.command == "native-promote":
            from kernel_envolve.evaluation.native.evaluator import promote_native

            print(json.dumps(promote_native(args.run), indent=2))
            return 0
        if args.command == "native-evaluate":
            from kernel_envolve.evaluation.native.evaluator import evaluate_native

            run, result = evaluate_native(
                args.task,
                args.mode,
                args.timeout,
                cases=args.cases,
                seeds=args.seeds,
                check_only=args.check_only,
            )
            print(
                json.dumps(
                    {
                        "run": str(run),
                        "execution": result["execution"],
                        "correctness": result["correctness"],
                        "error": result.get("error"),
                        "seconds": result["seconds"],
                        "phase_wall_seconds": result.get("phase_wall_seconds", {}),
                        "failed_phase": result.get("failed_phase"),
                        **result["decision"],
                    },
                    indent=2,
                )
            )
            return (
                2 if result["execution"] == "error" else 1 if result["correctness"] != "pass" else 0
            )
        if args.command == "promote":
            print(json.dumps(promote(args.run), indent=2))
            return 0
        run, result = evaluate(
            args.task,
            args.mode,
            args.signal_only,
            args.timeout,
            args.baseline,
            args.cases,
            args.seeds,
            args.check_only,
        )
        print(
            json.dumps(
                {
                    "run": str(run),
                    "execution": result["execution"],
                    "correctness": result["correctness"],
                    "worker_process_wall_seconds": result["worker_process_wall_seconds"],
                    "phase_wall_seconds": result.get("phase_wall_seconds", {}),
                    **({"error": result["error"]} if result.get("error") else {}),
                    **result["decision"],
                },
                indent=2,
            )
        )
        return 2 if result["execution"] == "error" else 1 if result["correctness"] != "pass" else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"kernel-envolve: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
