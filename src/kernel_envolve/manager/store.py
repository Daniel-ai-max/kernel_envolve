"""Small transactional domain store. External work always follows a committed intent."""

import contextlib
import json
import socket
import sqlite3
import time
import uuid
from pathlib import Path

from .config import positive


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, root, *, readonly=False):
        self.root = Path(root).resolve()
        if readonly:
            path = self.root / "manager.sqlite"
            self.db = sqlite3.connect(
                path.as_uri() + "?mode=ro", uri=True, timeout=5, isolation_level=None
            )
            self.db.row_factory = sqlite3.Row
            return
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.root / "manager.sqlite"
        self.db = sqlite3.connect(path, timeout=15, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        path.chmod(0o600)
        self.db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.db.close()
            raise ValueError(f"Unsupported manager schema: {version}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS resources (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, problem TEXT);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS processes (
                job TEXT NOT NULL, pid INTEGER NOT NULL, identity TEXT NOT NULL,
                PRIMARY KEY(job,pid,identity));
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, desired TEXT NOT NULL DEFAULT 'active',
                reason TEXT, failures INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0,
                wall_seconds REAL NOT NULL DEFAULT 0, gpu_seconds REAL NOT NULL DEFAULT 0,
                report TEXT NOT NULL DEFAULT '{}', evidence_error TEXT, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, task TEXT NOT NULL, kind TEXT NOT NULL, resource TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', parent TEXT, command TEXT NOT NULL,
                pid INTEGER, identity TEXT, child_pid INTEGER, child_identity TEXT,
                created REAL NOT NULL, started REAL, heartbeat REAL, ended REAL,
                seconds REAL NOT NULL DEFAULT 0, error TEXT, exit_code INTEGER,
                cancel_requested INTEGER NOT NULL DEFAULT 0);
            CREATE UNIQUE INDEX IF NOT EXISTS one_agent ON jobs(task)
                WHERE kind='agent' AND status IN ('queued', 'running');
            CREATE UNIQUE INDEX IF NOT EXISTS one_gpu ON jobs(resource)
                WHERE kind='gpu' AND status='running';
            CREATE INDEX IF NOT EXISTS queue ON jobs(status, kind, created);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, at REAL NOT NULL, task TEXT, kind TEXT NOT NULL,
                data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                path TEXT PRIMARY KEY, task TEXT NOT NULL, imported REAL NOT NULL,
                data TEXT NOT NULL, error TEXT);
            CREATE TABLE IF NOT EXISTS promotions (
                task TEXT NOT NULL, run TEXT NOT NULL, at REAL NOT NULL, gpu_seconds REAL NOT NULL,
                data TEXT NOT NULL, PRIMARY KEY(task, run));
            PRAGMA user_version=1;
        """)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(jobs)")}
        if "cancel_requested" not in columns:
            with self.transaction():
                columns = {r[1] for r in self.db.execute("PRAGMA table_info(jobs)")}
                if "cancel_requested" not in columns:
                    self.db.execute(
                        "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                    )
        try:
            with self.transaction():
                host = socket.gethostname()
                previous = self.one("SELECT data FROM metadata WHERE key='host'")
                if previous and json.loads(previous["data"]) != host:
                    raise ValueError(
                        "Manager state belongs to another host; reconcile jobs before migration"
                    )
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES('host',?)", (encode(host),))
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def transaction(self, *, readonly=False):
        self.db.execute("BEGIN" if readonly else "BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def rows(self, query, params=()):
        return [dict(r) for r in self.db.execute(query, params)]

    def one(self, query, params=()):
        row = self.db.execute(query, params).fetchone()
        return dict(row) if row else None

    def event(self, task, kind, data):
        self.db.execute(
            "INSERT INTO events(at,task,kind,data) VALUES(?,?,?,?)",
            (time.time(), task, kind, encode(data)),
        )

    def task(self, key):
        row = self.one("SELECT * FROM tasks WHERE id=?", (key,))
        if not row:
            raise ValueError(f"Unknown task: {key}")
        row["spec"], row["report"] = json.loads(row["spec"]), json.loads(row["report"])
        return row

    def resource(self, key):
        row = self.one("SELECT * FROM resources WHERE id=?", (key,))
        if not row:
            raise ValueError(f"Unknown resource: {key}")
        row["spec"] = json.loads(row["spec"])
        return row

    def register(self, config):
        with self.transaction():
            for resource in config["resources"]:
                for existing in self.rows(
                    "SELECT id,spec FROM resources WHERE id<>?", (resource["id"],)
                ):
                    if json.loads(existing["spec"])["device"] == resource["device"]:
                        raise ValueError("Physical device already registered under another ID")
                old = self.one("SELECT spec FROM resources WHERE id=?", (resource["id"],))
                if old and json.loads(old["spec"]) != resource:
                    raise ValueError("Existing resource differs; use controls for session limits")
                self.db.execute(
                    "INSERT OR IGNORE INTO resources(id,spec) VALUES(?,?)",
                    (resource["id"], encode(resource)),
                )
            for task in config["tasks"]:
                if self.root.is_relative_to(task["workspace"]):
                    raise ValueError("Manager state must be outside all task workspaces")
                old = self.one("SELECT spec FROM tasks WHERE id=?", (task["id"],))
                if old and json.loads(old["spec"]) != task:
                    raise ValueError("Existing task differs; create a new identity or use controls")
                for other in self.rows("SELECT id,spec FROM tasks WHERE id<>?", (task["id"],)):
                    a, b = Path(task["workspace"]), Path(json.loads(other["spec"])["workspace"])
                    if a.is_relative_to(b) or b.is_relative_to(a):
                        raise ValueError("Workspace already belongs to another task")
                if not old:
                    self.db.execute(
                        "INSERT INTO tasks(id,spec,created) VALUES(?,?,?)",
                        (task["id"], encode(task), time.time()),
                    )
                    self.event(task["id"], "registered", task["baseline"])

    def control(self, key, action, changes=None):
        with self.transaction():
            task = self.task(key)
            if action in ("pause", "resume"):
                desired = "paused" if action == "pause" else "active"
                self.db.execute(
                    "UPDATE tasks SET desired=?,reason=NULL,failures=0,next_at=0 WHERE id=?",
                    (desired, key),
                )
                if action == "pause":
                    self.db.execute(
                        "UPDATE jobs SET cancel_requested=1 WHERE task=? AND status IN ('queued','running')",
                        (key,),
                    )
                    self.db.execute(
                        "UPDATE jobs SET status='cancelled',ended=?,error='Paused before launch' WHERE task=? AND status='queued'",
                        (time.time(), key),
                    )
            elif action == "configure":
                changes = changes or {}
                if set(changes) - {"resource", "wall_budget_seconds", "gpu_budget_seconds"}:
                    raise ValueError("Only resource assignment and cumulative budgets are mutable")
                if task["desired"] != "paused" or self.one(
                    "SELECT id FROM jobs WHERE task=? AND status IN ('queued','running')", (key,)
                ):
                    raise ValueError("Pause and wait for all jobs to finish before reconfiguring")
                for field, value in changes.items():
                    if field == "resource":
                        self.resource(value)
                    else:
                        positive(value, field, nullable=True)
                self.db.execute(
                    "UPDATE tasks SET spec=? WHERE id=?", (encode(task["spec"] | changes), key)
                )
            else:
                raise ValueError("Unknown action")
            self.event(key, action, changes or {})

    def budget_reason(self, task):
        for kind in ("wall", "gpu"):
            limit = task["spec"][f"{kind}_budget_seconds"]
            if limit is not None and task[f"{kind}_seconds"] >= limit:
                return f"{kind} budget exhausted"
        return None

    def enqueue(self, task, kind, command, parent=None):
        key = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO jobs(id,task,kind,resource,parent,command,created) VALUES(?,?,?,?,?,?,?)",
            (key, task["id"], kind, task["spec"]["resource"], parent, encode(command), time.time()),
        )
        self.event(task["id"], "queued", {"job": key, "kind": kind})
        return key

    def schedule(self):
        """Reserve session capacity and commit launch intents before spawning workers."""
        scheduled = []
        with self.transaction():
            for row in self.rows("SELECT id FROM tasks ORDER BY next_at,created"):
                task = self.task(row["id"])
                if task["desired"] != "active" or task["next_at"] > time.time():
                    continue
                if self.budget_reason(task):
                    self.db.execute(
                        "UPDATE tasks SET desired='budget',reason=? WHERE id=?",
                        (self.budget_reason(task), task["id"]),
                    )
                    continue
                if self.one(
                    "SELECT id FROM jobs WHERE task=? AND status IN ('queued','running')",
                    (task["id"],),
                ):
                    continue
                resource = self.resource(task["spec"]["resource"])
                count = self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE resource=? AND kind='agent' "
                    "AND status IN ('queued','running')",
                    (resource["id"],),
                ).fetchone()[0]
                if resource["problem"] or count >= resource["spec"]["sessions"]:
                    continue
                scheduled.append(self.enqueue(task, "agent", task["spec"]["command"]))
        return scheduled

    def job(self, key):
        row = self.one("SELECT * FROM jobs WHERE id=?", (key,))
        if not row:
            raise ValueError(f"Unknown job: {key}")
        return row

    def claim(self, key, pid, identity):
        with self.transaction():
            job = self.job(key)
            task = self.task(job["task"])
            if job["status"] != "queued" or task["desired"] != "active" or self.budget_reason(task):
                return False
            if job["kind"] == "gpu":
                if self.resource(job["resource"])["problem"]:
                    return False
                first = self.one(
                    "SELECT id FROM jobs WHERE resource=? AND kind='gpu' "
                    "AND status='queued' ORDER BY created,id LIMIT 1",
                    (job["resource"],),
                )
                if first["id"] != key or self.one(
                    "SELECT id FROM jobs WHERE resource=? AND kind='gpu' AND status='running'",
                    (job["resource"],),
                ):
                    return False
                if self.job(job["parent"])["status"] != "running":
                    return False
            self.db.execute(
                "UPDATE jobs SET status='running',pid=?,identity=?,started=?,heartbeat=? "
                "WHERE id=?",
                (pid, identity, time.time(), time.time(), key),
            )
            self.event(job["task"], "started", {"job": key})
            return True

    def heartbeat(self, key, seconds):
        with self.transaction():
            job = self.job(key)
            if job["status"] != "running":
                return
            delta = max(0, seconds - job["seconds"])
            column = "wall_seconds" if job["kind"] == "agent" else "gpu_seconds"
            self.db.execute(
                f"UPDATE tasks SET {column}={column}+? WHERE id=?", (delta, job["task"])
            )
            self.db.execute(
                "UPDATE jobs SET seconds=MAX(seconds,?),heartbeat=? WHERE id=?",
                (seconds, time.time(), key),
            )

    def finish(self, key, status, error=None, exit_code=None):
        if status not in ("succeeded", "failed", "interrupted", "cancelled"):
            raise ValueError("Invalid terminal status")
        with self.transaction():
            job = self.job(key)
            if job["status"] not in ("queued", "running"):
                return
            self.db.execute(
                "UPDATE jobs SET status=?,ended=?,error=?,exit_code=? WHERE id=?",
                (status, time.time(), error, exit_code, key),
            )
            if job["kind"] == "agent":
                task = self.task(job["task"])
                failures = task["failures"] + 1 if status in ("failed", "interrupted") else 0
                delay = (
                    min(300, 5 * 2 ** min(failures, 6))
                    if failures
                    else task["spec"]["cooldown_seconds"]
                )
                desired = task["desired"]
                reason = self.budget_reason(task)
                if desired == "active" and reason:
                    desired = "budget"
                elif desired == "active" and failures >= 3:
                    desired, reason = (
                        "blocked",
                        "Three consecutive failed rounds; inspect and resume",
                    )
                self.db.execute(
                    "UPDATE tasks SET failures=?,next_at=?,desired=?,reason=? WHERE id=?",
                    (failures, time.time() + delay, desired, reason, task["id"]),
                )
            self.event(job["task"], status, {"job": key, "error": error, "exit_code": exit_code})

    def report(self, task_id, job_id, report):
        if not isinstance(report, dict):
            raise ValueError("Report must be a JSON object")
        allowed = {
            "phase",
            "hypothesis",
            "outcome",
            "next_action",
            "session_url",
            "pr_urls",
            "integrated_revision",
        }
        if set(report) - allowed:
            raise ValueError(
                "Report contains unknown fields; performance comes only from evaluator evidence"
            )
        if any(
            not isinstance(v, str) or len(v) > 8000 for k, v in report.items() if k != "pr_urls"
        ):
            raise ValueError("Report fields must be short strings")
        if "pr_urls" in report and (
            not isinstance(report["pr_urls"], list)
            or len(report["pr_urls"]) > 20
            or any(
                not isinstance(x, str) or not x.startswith("https://") for x in report["pr_urls"]
            )
        ):
            raise ValueError("PR links must be HTTPS URLs")
        with self.transaction():
            job = self.job(job_id)
            if job["task"] != task_id or job["kind"] != "agent" or job["status"] != "running":
                raise ValueError("Only the current agent round may report")
            old = self.task(task_id)["report"]
            self.db.execute("UPDATE tasks SET report=? WHERE id=?", (encode(old | report), task_id))
            self.event(task_id, "report", report)
