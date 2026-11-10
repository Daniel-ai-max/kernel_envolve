"""Independent RSI ledger. No learning writes to the manager or evaluator database."""

import contextlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

from ..storage import digest, write_json
from .model import context, number, read, select_rules


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = read(self.root / "config.json")
        self.owned()
        self.db = sqlite3.connect(self.root / "rsi.sqlite", timeout=1, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
            self.db.close()
            raise ValueError("Unsupported RSI schema")
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA user_version=1;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence(
                seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, parent TEXT NOT NULL,
                data TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate');
            CREATE TABLE IF NOT EXISTS active(scope TEXT PRIMARY KEY, id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS validations(
                id TEXT PRIMARY KEY, candidate TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operations(
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, reserved REAL NOT NULL,
                charged REAL, started REAL NOT NULL, ended REAL, error TEXT);
            CREATE TABLE IF NOT EXISTS history(
                seq INTEGER PRIMARY KEY, at REAL NOT NULL, scope TEXT NOT NULL,
                previous TEXT NOT NULL, current TEXT NOT NULL, reason TEXT NOT NULL);
        """)
        expected = self.get("config_hash")
        if expected and expected != digest(self.config):
            self.db.close()
            raise ValueError("RSI configuration changed; create a new state for a new contract")
        self.set("config_hash", digest(self.config))

    def owned(self):
        owner = read(Path(self.config["manager"]) / "rsi-owner.json")
        if owner != {"state": str(self.root), "config_hash": digest(self.config)}:
            raise ValueError(
                "RSI configuration changed or this state is not the registered publisher"
            )

    def detach(self):
        self.owned()
        if self.get("enabled", True):
            raise ValueError("Disable RSI before detaching its publisher")
        directory = Path(self.config["manager"]) / "rsi-context"
        for tasks in self.config["scopes"].values():
            for task in tasks:
                (directory / f"{task}.json").unlink(missing_ok=True)
        (Path(self.config["manager"]) / "rsi-owner.json").unlink()

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value)))

    def active(self, scope):
        row = self.db.execute("SELECT id FROM active WHERE scope=?", (scope,)).fetchone()
        return row[0] if row else "base"

    def candidate(self, identity):
        if identity == "base":
            return {"id": "base", "rules": []}
        row = self.db.execute(
            "SELECT data,status FROM candidates WHERE id=?", (identity,)
        ).fetchone()
        if not row:
            raise ValueError("Unknown candidate")
        value = json.loads(row[0])
        if digest({k: v for k, v in value.items() if k != "id"}) != identity:
            raise ValueError("Candidate identity mismatch")
        return value | {"status": row[1]}

    def add_evidence(self, value):
        identity = digest(value)
        self.db.execute(
            "INSERT OR IGNORE INTO evidence(id,data) VALUES(?,?)", (identity, json.dumps(value))
        )
        return identity

    def reserve(self, kind, seconds):
        if kind not in ("learn", "validation"):
            raise ValueError("Unknown budget")
        number(seconds, "reservation", upper=365 * 86400)
        with self.transaction():
            used = self.db.execute(
                "SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM operations WHERE kind=?",
                (kind,),
            ).fetchone()[0]
            if used + seconds > self.config[f"{kind}_budget_seconds"]:
                raise ValueError(f"{kind} budget exhausted")
            key = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO operations(id,kind,reserved,started) VALUES(?,?,?,?)",
                (key, kind, seconds, time.time()),
            )
        return key

    def finish(self, key, elapsed, error=None):
        # A killed process retains its full reservation, including after a restart.
        self.db.execute(
            "UPDATE operations SET charged=?,ended=?,error=? WHERE id=? AND ended IS NULL",
            (elapsed, time.time(), error, key),
        )

    def enable(self, enabled):
        self.owned()
        marker = Path(self.config["manager"]) / "rsi-context" / "DISABLED"
        marker.parent.mkdir(exist_ok=True, mode=0o700)
        if not enabled:
            marker.touch()  # Immediate fail-open switch, even while a validator owns its lock.
        self.set("enabled", enabled)
        self.publish()
        if enabled:
            marker.unlink(missing_ok=True)

    def publish(self):
        """Bounded materialized views; normal agent startup reads only one small file."""
        self.owned()
        from ..manager.store import Store as Manager

        manager = Manager(self.config["manager"], readonly=True)
        try:
            for scope, tasks in self.config["scopes"].items():
                identity = self.active(scope)
                rules = self.candidate(identity)["rules"]
                for task_id in tasks:
                    task = manager.task(task_id)
                    facts = self.config.get("profiles", {}).get(task_id, {})
                    selected = select_rules(rules, facts)
                    rendered = context(selected) if self.get("enabled", True) else ""
                    payload = {
                        "schema": 1,
                        "scope": scope,
                        "version": identity,
                        "task_identity": task["spec"]["identity"],
                        "text": rendered,
                        "selected_rules": [r["id"] for r in selected],
                    }
                    payload["hash"] = digest(payload)
                    directory = Path(self.config["manager"]) / "rsi-context"
                    directory.mkdir(exist_ok=True, mode=0o700)
                    write_json(directory / f"{task_id}.json", payload)
        finally:
            manager.close()
