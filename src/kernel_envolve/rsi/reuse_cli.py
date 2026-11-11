"""Operator commands for bounded artifact reuse; never run a model or evaluator.

Invoke with python -m kernel_envolve.rsi.reuse_cli. Status is read-only, including
on stale implementations. Recovery commands only disable or close incomplete
uses; they cannot re-enable stale state or adopt a candidate.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .model import read
from .reuse_memory import ReuseMemory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--archive", type=Path, required=True)
    initialize.add_argument("--max-entries", type=int, default=64)
    initialize.add_argument("--max-trials", type=int, default=4096)
    sub.add_parser("register").add_argument("record")
    status = sub.add_parser("status")
    status.add_argument("--pending-limit", type=int, default=20)
    for name in ("enable", "disable"):
        sub.add_parser(name)
    for name in ("entry-enable", "entry-disable"):
        sub.add_parser(name).add_argument("entry")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--scope", required=True)
    prepare.add_argument("--task", type=Path, required=True)
    prepare.add_argument("--artifacts", type=Path, required=True)
    prepare.add_argument("--profile", type=Path)
    prepare.add_argument("--actions", type=int, default=4)
    observe = sub.add_parser("observe")
    observe.add_argument("ticket")
    observe.add_argument("--final", type=Path, required=True)
    abandon = sub.add_parser("abandon")
    abandon.add_argument("ticket")
    abandon.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    memory = None
    try:
        if args.command == "init":
            memory = ReuseMemory.create(
                args.state, args.archive, max_entries=args.max_entries, max_trials=args.max_trials
            )
            result = {"status": "initialized", "state": str(memory.root)}
        else:
            memory = ReuseMemory(
                args.state,
                recovery=args.command in {"status", "disable", "entry-disable", "abandon"},
                readonly=args.command == "status",
            )
            if args.command == "status":
                result = memory.status(pending_limit=args.pending_limit)
            elif args.command == "register":
                result = {"entry": memory.register(args.record)}
            elif args.command in {"enable", "disable"}:
                enabled = args.command == "enable"
                memory.set_enabled(enabled)
                result = {"enabled": enabled}
            elif args.command in {"entry-enable", "entry-disable"}:
                enabled = args.command == "entry-enable"
                memory.set_entry_enabled(args.entry, enabled)
                result = {"entry": args.entry, "enabled": enabled}
            elif args.command == "prepare":
                result = memory.prepare(
                    args.scope,
                    args.task,
                    read(args.profile, 8192) if args.profile else {},
                    args.artifacts,
                    max_actions=args.actions,
                )
            elif args.command == "observe":
                result = memory.observe(args.ticket, confirmation_run=args.final)
            else:
                result = memory.abandon(args.ticket, args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        print(
            json.dumps({"error": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    finally:
        if memory is not None:
            memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
