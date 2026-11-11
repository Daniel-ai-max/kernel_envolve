"""Small transactional domain store. External work always follows a committed intent."""

import contextlib
import json
import socket
import sqlite3
import time
import uuid
from pathlib import Path

from .config import positive, resource_devices


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
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            self.db.close()
            raise ValueError(f"Unsupported manager schema: {version}")
        try:
            if self.one("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"):
                previous = self.one("SELECT data FROM metadata WHERE key='host'")
                if previous and json.loads(previous["data"]) != socket.gethostname():
                    raise ValueError(
                        "Manager state belongs to another host; reconcile jobs before migration"
                    )
            if version == 1:
                self._upgrade_legacy()
        except BaseException:
            self.db.close()
            raise
        self.db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")
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
        """)
        if version == 0:
            with self.transaction():
                self._upgrade_columns()
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

    def _upgrade_legacy(self):
        from .process import alive, exclusive

        message = "Stop the schema 1 coordinator and finish existing jobs before upgrading"
        with exclusive(self.root / "coordinator.lock") as handle:
            if handle is None:
                raise ValueError(message + ": coordinator lock is held")
            with self.transaction():
                # Check and migrate in one write transaction; no old writer can
                # enqueue work between the quiet-state check and the schema change.
                if self.one("SELECT id FROM jobs WHERE status IN ('queued','running') LIMIT 1"):
                    raise ValueError(message + ": queued or running jobs remain")
                for child in self.rows("SELECT pid,identity FROM processes"):
                    if alive(child["pid"], child["identity"]):
                        raise ValueError(message + ": a registered descendant is still alive")
                self._upgrade_columns()

    def _upgrade_columns(self):
        """Caller holds the schema transaction and, for legacy state, coordinator lock."""
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(jobs)")}
        if "cancel_requested" not in columns:
            self.db.execute(
                "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
            )
        if "devices" not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN devices TEXT")
        if "gpu_count" not in columns:
            # Historical single-card charges are never recomputed from new resources.
            self.db.execute("ALTER TABLE jobs ADD COLUMN gpu_count INTEGER NOT NULL DEFAULT 1")
        for row in self.rows(
            "SELECT jobs.id,resources.spec FROM jobs JOIN resources ON jobs.resource=resources.id "
            "WHERE jobs.devices IS NULL"
        ):
            self.db.execute(
                "UPDATE jobs SET devices=? WHERE id=?",
                (encode(resource_devices(json.loads(row["spec"]))), row["id"]),
            )
        self.db.execute("PRAGMA user_version=2")

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
        replacement = self.one("SELECT data FROM metadata WHERE key=?", (f"superseded:{key}",))
        row["superseded_by"] = (
            json.loads(replacement["data"])["superseded_by"] if replacement else None
        )
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
                    if set(resource_devices(json.loads(existing["spec"]))) & set(
                        resource_devices(resource)
                    ):
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
                if old and {"host_driven": False, **json.loads(old["spec"])} != task:
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
            if task["superseded_by"] or task["desired"] == "superseded":
                if action == "pause":
                    return  # Allocation cleanup may pause every task, including history.
                raise ValueError("Superseded tasks cannot be resumed or reconfigured")
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

    def supersede(self, key, replacement):
        """Retire an existing identity without transferring usage or allocating new budget."""
        with self.transaction():
            old, new = self.task(key), self.task(replacement)
            if key == replacement:
                raise ValueError("A task cannot supersede itself")
            previous = self.one("SELECT data FROM metadata WHERE key=?", (f"superseded:{key}",))
            if previous:
                receipt = json.loads(previous["data"])
                if receipt["superseded_by"] == replacement:
                    return receipt
                raise ValueError("Task already has a different replacement")
            if old["desired"] == "superseded":
                raise ValueError("Superseded task is missing its migration receipt")
            if new["superseded_by"] or new["desired"] == "superseded":
                raise ValueError("Replacement must be a current task; cycles are not allowed")
            for row in self.rows("SELECT data FROM metadata WHERE key LIKE 'superseded:%'"):
                if json.loads(row["data"])["superseded_by"] == replacement:
                    raise ValueError("Replacement already has a predecessor")
            if self.one(
                "SELECT id FROM jobs WHERE task=? AND status IN ('queued','running')", (key,)
            ):
                raise ValueError("Wait for all predecessor jobs to finish before superseding")
            a, b = old["spec"], new["spec"]
            if a["resource"] != b["resource"] or a["kind"] != b["kind"]:
                raise ValueError("Replacement must use the same resource and evaluator kind")
            self.resource(a["resource"])
            if any(
                a["baseline"][field] != b["baseline"][field]
                for field in ("repository", "revision", "entrypoint", "hardware")
            ):
                raise ValueError("Replacement must preserve the fixed source baseline and target")
            workspaces = []
            for spec in (a, b):
                workspace, task_dir = (Path(spec[f]).resolve() for f in ("workspace", "task_dir"))
                if (
                    not workspace.is_dir()
                    or not task_dir.is_dir()
                    or not task_dir.is_relative_to(workspace)
                    or self.root.is_relative_to(workspace)
                ):
                    raise ValueError("Replacement requires valid dedicated workspaces")
                workspaces.append(workspace)
            if any(workspaces[i].is_relative_to(workspaces[1 - i]) for i in (0, 1)):
                raise ValueError("Predecessor and replacement workspaces must not overlap")
            for kind in ("wall", "gpu"):
                limit, next_limit = a[f"{kind}_budget_seconds"], b[f"{kind}_budget_seconds"]
                if limit is not None and (
                    next_limit is None or next_limit > limit - old[f"{kind}_seconds"]
                ):
                    raise ValueError(
                        f"Replacement {kind} budget exceeds predecessor's unused budget"
                    )
            receipt = {
                "task": key,
                "superseded_by": replacement,
                "at": time.time(),
                "predecessor": {
                    field: old[field] for field in ("spec", "wall_seconds", "gpu_seconds")
                },
                "replacement": {
                    field: new[field] for field in ("spec", "wall_seconds", "gpu_seconds")
                },
            }
            self.db.execute(
                "INSERT INTO metadata(key,data) VALUES(?,?)", (f"superseded:{key}", encode(receipt))
            )
            self.db.execute(
                "UPDATE tasks SET desired='superseded',reason=? WHERE id=?",
                (f"Replaced by {replacement}", key),
            )
            self.event(key, "superseded", {"superseded_by": replacement})
            self.event(replacement, "supersedes", {"task": key})
            return receipt

    def budget_reason(self, task):
        for kind in ("wall", "gpu"):
            limit = task["spec"][f"{kind}_budget_seconds"]
            if limit is not None and task[f"{kind}_seconds"] >= limit:
                return f"{kind} budget exhausted"
        return None

    def enqueue(self, task, kind, command, parent=None):
        key = uuid.uuid4().hex
        resource = task["spec"]["resource"]
        devices = resource_devices(self.resource(resource)["spec"])
        if parent:
            frozen = self.job(parent)
            resource, devices = frozen["resource"], self.job_devices(frozen)
        self.db.execute(
            "INSERT INTO jobs(id,task,kind,resource,parent,command,created,devices,gpu_count) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                key,
                task["id"],
                kind,
                resource,
                parent,
                encode(command),
                time.time(),
                encode(devices),
                len(devices),
            ),
        )
        self.event(task["id"], "queued", {"job": key, "kind": kind})
        return key

    def schedule(self):
        """Reserve session capacity and commit launch intents before spawning workers."""
        from .recovery import migrate_legacy

        scheduled = []
        with self.transaction():
            for row in self.rows("SELECT id FROM tasks ORDER BY next_at,created"):
                task = migrate_legacy(self, self.task(row["id"]))
                if task["desired"] != "active":
                    continue
                if self.budget_reason(task):
                    self.db.execute(
                        "UPDATE tasks SET desired='budget',reason=? WHERE id=?",
                        (self.budget_reason(task), task["id"]),
                    )
                    continue
                if task["next_at"] > time.time():
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
                permit = None
                if task["spec"].get("host_driven"):
                    ready = self.one(
                        "SELECT data FROM metadata WHERE key=?", ("host-ready:" + task["id"],)
                    )
                    if not ready or json.loads(ready["data"])["expires_at"] <= time.time():
                        continue
                    permit = json.loads(ready["data"])
                job = self.enqueue(task, "agent", task["spec"]["command"])
                if permit:
                    permit["round"] = job
                    self.db.execute(
                        "DELETE FROM metadata WHERE key=?", ("host-ready:" + task["id"],)
                    )
                    self.db.execute(
                        "INSERT INTO metadata VALUES(?,?)", ("host-round:" + job, encode(permit))
                    )
                    self.db.execute(
                        "UPDATE metadata SET data=? WHERE key=?",
                        (encode(permit), f"host-claim:{task['id']}:{permit['claim']}"),
                    )
                scheduled.append(job)
        return scheduled

    def job(self, key):
        row = self.one("SELECT * FROM jobs WHERE id=?", (key,))
        if not row:
            raise ValueError(f"Unknown job: {key}")
        return row

    def job_devices(self, job):
        return json.loads(job["devices"])

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
                f"UPDATE tasks SET {column}={column}+? WHERE id=?",
                (delta * (job["gpu_count"] if job["kind"] == "gpu" else 1), job["task"]),
            )
            self.db.execute(
                "UPDATE jobs SET seconds=MAX(seconds,?),heartbeat=? WHERE id=?",
                (seconds, time.time(), key),
            )

    def finish(self, key, status, error=None, exit_code=None):
        if status not in ("succeeded", "failed", "interrupted", "cancelled", "yielded"):
            raise ValueError("Invalid terminal status")
        from .recovery import assess, implementation

        job = self.job(key)
        if job["status"] not in ("queued", "running"):
            return
        if status == "yielded":
            from .bridge import idle_exit

            if not idle_exit(self, job):
                raise ValueError("Yielded rounds require the bridge's idle receipt")
        # FSx/source reads must never hold the shared manager write transaction.
        current = (
            implementation(self.task(job["task"]))
            if job["kind"] == "agent" and status != "yielded"
            else None
        )
        with self.transaction():
            job = self.job(key)
            if job["status"] not in ("queued", "running"):
                return
            self.db.execute(
                "UPDATE jobs SET status=?,ended=?,error=?,exit_code=? WHERE id=?",
                (status, time.time(), error, exit_code, key),
            )
            if job["kind"] == "agent" and status == "yielded":
                task = self.task(job["task"])
                recovery = assess(self, task, job, status, None)
                recovery.update(
                    state="awaiting_host",
                    round=key,
                    observable_progress=False,
                    reason="Host lease expired; completed results remain consumable",
                    retry_after_seconds=0,
                    attempts=task["failures"],
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    ("recovery:" + task["id"], encode(recovery)),
                )
                if task["desired"] == "active":
                    reason = self.budget_reason(task)
                    self.db.execute(
                        "UPDATE tasks SET desired=?,reason=? WHERE id=?",
                        ("budget" if reason else "active", reason, task["id"]),
                    )
            elif job["kind"] == "agent":
                task = self.task(job["task"])
                recovery = assess(self, task, job, status, current)
                failures = 0 if recovery["observable_progress"] else task["failures"] + 1
                delay = (
                    min(300, 5 * 2 ** min(failures, 6))
                    if failures
                    else task["spec"]["cooldown_seconds"]
                )
                desired = task["desired"]
                reason = self.budget_reason(task)
                if desired == "active" and reason:
                    desired = "budget"
                elif desired == "active" and failures:
                    reason = (
                        error
                        or "Recovery needs new action evidence with a changed outcome checkpoint"
                    )
                recovery.update(reason=reason, retry_after_seconds=delay, attempts=failures)
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    ("recovery:" + task["id"], encode(recovery)),
                )
                self.event(
                    task["id"],
                    "recovery_checkpoint"
                    if recovery["observable_progress"]
                    else "recovery_requested",
                    {k: v for k, v in recovery.items() if k != "execution_signatures"},
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
            progress_key = f"report_progress:{task_id}"
            progress = self.one("SELECT data FROM metadata WHERE key=?", (progress_key,))
            if old | report != old:
                at = time.time()
            elif progress is None:
                # Preserve the last legacy report time on upgrade. Repeating an unchanged
                # checkpoint or finishing/restarting a round is not new research activity.
                previous = self.one(
                    "SELECT MAX(at) AS at FROM events WHERE task=? AND kind='report'", (task_id,)
                )
                at = previous["at"] if previous else None
            else:
                at = json.loads(progress["data"])
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)", (progress_key, encode(at))
            )
            self.db.execute("UPDATE tasks SET report=? WHERE id=?", (encode(old | report), task_id))
            self.event(task_id, "report", report)
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                (
                    "round_report:" + job_id,
                    encode({"round": job_id, "recorded_at": time.time(), "report": old | report}),
                ),
            )

    def consume_result(self, task_id, job_id, record, *, legacy=False):
        """Archive a receipt-bound analysis even after its execution round closes."""
        key = f"consumed_result:{job_id}:{record['request']}"
        with self.transaction():
            job = self.job(job_id)
            if job["task"] != task_id or job["kind"] != "agent":
                raise ValueError("Consumption must belong to its agent round")
            saved = self.one("SELECT data FROM metadata WHERE key=?", (key,))
            if saved:
                if json.loads(saved["data"])["record"] != record:
                    raise ValueError("Result already consumed with a different analysis")
                return
            # Preserve the original observation time when repairing a legacy file/event.
            previous = next(
                (
                    event
                    for event in (
                        self.rows(
                            "SELECT at,data FROM events WHERE task=? AND kind='host_consumed_result' ORDER BY seq DESC",
                            (task_id,),
                        )
                        if legacy
                        else []
                    )
                    if json.loads(event["data"]).get("round") == job_id
                    and json.loads(event["data"]).get("request") == record["request"]
                ),
                None,
            )
            at = previous["at"] if previous else time.time()
            self.db.execute(
                "INSERT INTO metadata VALUES(?,?)",
                (key, encode({"recorded_at": at, "record": record})),
            )
            prior = self.one("SELECT data FROM metadata WHERE key=?", ("round_report:" + job_id,))
            prior = json.loads(prior["data"]) if prior else {}
            report = {
                "outcome": record["outcome"],
                "next_action": record["next_action"],
            }
            if at >= prior.get("recorded_at", 0):
                self.db.execute(
                    "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    (
                        "round_report:" + job_id,
                        encode(
                            {
                                "round": job_id,
                                "recorded_at": at,
                                "report": prior.get("report", {}) | report,
                            }
                        ),
                    ),
                )
                latest = self.one(
                    "SELECT id FROM jobs WHERE task=? AND kind='agent' ORDER BY created DESC,rowid DESC LIMIT 1",
                    (task_id,),
                )
                last_report = self.one(
                    "SELECT MAX(at) AS at FROM events WHERE task=? AND kind='report'", (task_id,)
                )
                if latest["id"] == job_id and at >= (last_report["at"] or 0):
                    task = self.task(task_id)
                    self.db.execute(
                        "UPDATE tasks SET report=? WHERE id=?",
                        (encode(task["report"] | report), task_id),
                    )
                    progress = self.one(
                        "SELECT data FROM metadata WHERE key=?", ("report_progress:" + task_id,)
                    )
                    changed = task["report"] | report != task["report"]
                    if changed or progress is None:
                        self.db.execute(
                            "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                            (
                                "report_progress:" + task_id,
                                encode(at if changed else last_report["at"]),
                            ),
                        )
                    # A projection repair retains the original consumption timestamp.
                    self.db.execute(
                        "INSERT INTO events(at,task,kind,data) VALUES(?,?,?,?)",
                        (at, task_id, "report", encode(report)),
                    )
            if not previous:
                self.event(task_id, "host_consumed_result", record | {"round": job_id})
