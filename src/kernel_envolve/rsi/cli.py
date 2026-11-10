"""Explicit RSI operator commands and an optional low-frequency sidecar."""

import argparse
import contextlib
import json
import signal
import sqlite3
import sys
import time
from pathlib import Path

from ..manager.process import exclusive
from .config import initialize
from .learning import collect, learn, stage, verify_evidence
from .model import read
from .store import Store
from .trials import adopt, checked_receipt, rollback, validate


def audit(store):
    revoked = []
    for scope in store.config["scopes"]:
        while store.active(scope) != "base":
            candidate = store.candidate(store.active(scope))
            try:
                verify_evidence(
                    store, [e for r in candidate["rules"] for e in r["evidence"]], scope
                )
                receipt = store.db.execute(
                    "SELECT id FROM validations WHERE candidate=? ORDER BY rowid DESC LIMIT 1",
                    (candidate["id"],),
                ).fetchone()
                if not receipt:
                    raise ValueError("Missing strategy validation")
                checked_receipt(store, receipt[0])
                break
            except (OSError, ValueError, KeyError, TypeError):
                revoked.append(candidate["id"])
                rollback(store, scope, "Supporting evidence became invalid")
    store.publish()
    return revoked


def cycle(store):
    if not store.get("enabled", True):
        return {"status": "disabled"}
    # Caller owns operation.lock. No previous validator can still be active.
    store.db.execute("UPDATE candidates SET status='rejected' WHERE status='validating'")
    store.publish()  # Recover an interrupted publication before new learning.
    result = {"collected": collect(store)}
    if time.time() - store.get("last_audit", 0) > 300:
        result["revoked"] = audit(store)
        store.set("last_audit", time.time())
    try:
        result["learning"] = learn(store)
    except InterruptedError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["learning"] = {"status": "failed", "error": str(exc)}
    if store.config["auto_validate"]:
        for row in store.db.execute(
            "SELECT c.id FROM candidates c WHERE c.status IN ('candidate','validated') "
            "AND NOT EXISTS (SELECT 1 FROM history h WHERE h.current=c.id) ORDER BY c.rowid LIMIT 128"
        ).fetchall():
            candidate = store.candidate(row[0])
            if candidate["id"] == store.active(candidate["scope"]):
                continue
            if candidate["parent"] != store.active(candidate["scope"]):
                store.db.execute("UPDATE candidates SET status='stale' WHERE id=?", (row[0],))
                continue
            if candidate["status"] == "validated":
                receipt = store.db.execute(
                    "SELECT id FROM validations WHERE candidate=? ORDER BY rowid DESC LIMIT 1",
                    (row[0],),
                ).fetchone()
                result["adopted"] = adopt(store, receipt[0])
                break
            result["validation"] = validate(store, row[0])
            if result["validation"].get("accepted"):
                result["adopted"] = adopt(store, result["validation"]["receipt"])
            break  # At most one expensive validation per cycle.
    return result


def serve(root, interval):
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopped:
            with exclusive(root / "operation.lock") as handle:
                if handle is not None:
                    store = Store(root)
                    try:
                        try:
                            result = cycle(store)
                            store.set("last_cycle", {"at": time.time(), **result})
                            store.set("last_error", None)
                        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                            store.set("last_error", {"at": time.time(), "error": str(exc)})
                    finally:
                        store.close()
            until = time.monotonic() + interval
            while not stopped and time.monotonic() < until:
                time.sleep(min(0.25, max(0, until - time.monotonic())))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bounded cross-task learning; evaluator gates are unchanged"
    )
    parser.add_argument("--state", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init")
    init.add_argument("config", type=Path)
    for action in (
        "collect",
        "learn",
        "cycle",
        "status",
        "audit",
        "publish",
        "enable",
        "disable",
        "detach",
    ):
        commands.add_parser(action)
    proposal = commands.add_parser("propose")
    proposal.add_argument("file", type=Path)
    check = commands.add_parser("validate")
    check.add_argument("candidate")
    activate = commands.add_parser("adopt")
    activate.add_argument("receipt")
    undo = commands.add_parser("rollback")
    undo.add_argument("scope")
    undo.add_argument("--reason", required=True)
    run = commands.add_parser("run")
    run.add_argument("--interval", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            initialize(args.state, args.config)
            with_store = Store(args.state)
            try:
                with_store.enable(True)
            finally:
                with_store.close()
            print(json.dumps({"status": "initialized", "state": str(args.state.resolve())}))
            return 0
        if args.action == "run" and (not 1 <= args.interval <= 86400):
            raise ValueError("Interval must be 1–86400 seconds")
        if args.action == "run":
            return serve(args.state, args.interval)
        # One writer/learner/validator per state, with immediate contention rejection.
        lock = (
            contextlib.nullcontext(True)
            if args.action in ("status", "disable")
            else exclusive(args.state / "operation.lock")
        )
        with lock as handle:
            if handle is None:
                raise ValueError("Another RSI operation is running; no duplicate work started")
            store = Store(args.state)
            try:
                if args.action == "status":
                    result = {
                        "enabled": store.get("enabled", True),
                        "active": {s: store.active(s) for s in store.config["scopes"]},
                        "candidates": [
                            dict(r)
                            for r in store.db.execute(
                                "SELECT id,scope,parent,status FROM candidates"
                            )
                        ],
                        "budgets": [
                            dict(r)
                            for r in store.db.execute(
                                "SELECT kind,SUM(COALESCE(charged,reserved)) AS charged_seconds FROM operations GROUP BY kind"
                            )
                        ],
                        "last_cycle": store.get("last_cycle"),
                        "last_error": store.get("last_error"),
                    }
                elif args.action == "detach":
                    store.detach()
                    result = {"status": "detached"}
                elif args.action in ("enable", "disable"):
                    store.enable(args.action == "enable")
                    result = {"enabled": store.get("enabled")}
                elif args.action == "collect":
                    result = {"collected": collect(store)}
                elif args.action == "learn":
                    result = learn(store)
                elif args.action == "propose":
                    result = {"candidate": stage(store, read(args.file, 65536))}
                elif args.action == "validate":
                    result = validate(store, args.candidate)
                elif args.action == "adopt":
                    result = {"active": adopt(store, args.receipt)}
                elif args.action == "rollback":
                    result = {"active": rollback(store, args.scope, args.reason)}
                elif args.action == "audit":
                    result = {"revoked": audit(store)}
                elif args.action == "publish":
                    store.publish()
                    result = {"status": "published"}
                elif args.action == "cycle":
                    result = cycle(store)
                print(json.dumps(result, ensure_ascii=False, indent=2))
            finally:
                store.close()
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"kernel-envolve rsi: {exc}", file=sys.stderr)
        return 2
