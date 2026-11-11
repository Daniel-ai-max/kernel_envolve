"""Optional background feedback for explicitly registered, unadopted artifacts.

The trusted controller calls prepare BETWEEN trials and passes only its bounded
snapshot to research.run. Every offer spends one ordinary full-evaluation action;
feedback comes from existing receipts plus the caller's independent final. This
module executes no candidates, starts no models, and grants no Registry admission.
Keep this database and the raw archive outside the solver's write permissions.
"""

import contextlib
import json
import sqlite3
import time
from pathlib import Path

from ..storage import config, digest, evaluator_hash, file_hash
from . import artifact_lineage as lineage
from .artifact_snapshot import COMPATIBILITY
from .artifact_validation import task_key
from .episodes import MAX_EPISODE_BYTES, _source, load, save
from .model import profile, read, text
from .research import source_id
from .reuse import runtime_key
from .trials import validator_identity


def _implementation():
    # This optional controller must not invalidate unrelated historical lineage.
    return digest({"memory": file_hash(Path(__file__)), "validator": validator_identity()})


def _target(cfg, source, initial, scope):
    # Display names and paths do not earn a retry; different inputs/incumbents do.
    return digest({"task": task_key(cfg, source), "initial": initial, "scope": scope})


class ReuseMemory:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.db = sqlite3.connect(
            f"{(self.root / 'reuse.sqlite').as_uri()}?mode=rw", uri=True, isolation_level=None
        )
        self.db.row_factory = sqlite3.Row
        try:
            self.config = self._get("config")
            if digest(self.config) != self._get("config_hash"):
                raise ValueError("Reuse memory configuration changed")
            self.archive = Path(self.config["archive"])
            self._current()
        except BaseException:
            self.db.close()
            raise

    @classmethod
    def create(cls, root, archive, *, max_entries=64, max_trials=4096):
        """Explicit opt-in, bounded lifetime metadata; not a GPU or API budget."""
        root, archive = Path(root).resolve(), Path(archive).resolve()
        for value, upper in ((max_entries, 64), (max_trials, 4096)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("Invalid reuse memory limit")
        if root.exists() or not archive.is_dir():
            raise ValueError("Use a new memory directory and an existing artifact archive")
        cfg = {
            "schema": 1,
            "archive": str(archive),
            "max_entries": max_entries,
            "max_trials": max_trials,
            "implementation": _implementation(),
            "evaluator": evaluator_hash(),
        }
        root.mkdir(parents=True)
        db = sqlite3.connect(root / "reuse.sqlite")
        try:
            db.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE entries (seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                    data TEXT NOT NULL, enabled INTEGER NOT NULL);
                CREATE TABLE trials (id TEXT PRIMARY KEY, entry TEXT NOT NULL,
                    target TEXT NOT NULL, context TEXT NOT NULL, artifacts TEXT UNIQUE NOT NULL,
                    data TEXT NOT NULL, status TEXT NOT NULL,
                    receipt TEXT, record TEXT UNIQUE);
                CREATE INDEX feedback ON trials(entry,target,status);
                CREATE INDEX scoped_failure ON trials(entry,context,status);
                CREATE UNIQUE INDEX pending_target ON trials(target) WHERE status='pending';
            """)
            for key, value in {"config": cfg, "config_hash": digest(cfg), "enabled": True}.items():
                db.execute("INSERT INTO meta VALUES(?,?)", (key, json.dumps(value)))
            db.commit()
        finally:
            db.close()
        return cls(root)

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def _transaction(self):
        owns = not self.db.in_transaction
        try:
            self.db.execute("BEGIN IMMEDIATE")
            yield
            self.db.commit()
        except BaseException:
            if owns and self.db.in_transaction:
                self.db.rollback()
            raise

    def _get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Uninitialized reuse memory")
        return json.loads(row[0])

    def _current(self):
        if (
            self.config["implementation"] != _implementation()
            or self.config["evaluator"] != evaluator_hash()
        ):
            raise ValueError("Reuse memory code changed; use frozen code or a new memory")

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Enabled must be boolean")
        with self._transaction():
            self.db.execute("UPDATE meta SET value=? WHERE key='enabled'", (json.dumps(enabled),))

    def set_entry_enabled(self, entry, enabled):
        if type(enabled) is not bool:
            raise ValueError("Enabled must be boolean")
        with self._transaction():
            if (
                self.db.execute(
                    "UPDATE entries SET enabled=? WHERE id=?", (int(enabled), entry)
                ).rowcount
                != 1
            ):
                raise ValueError("Unknown reuse entry")

    def _entry(self, row):
        value = json.loads(row["data"])
        if digest({k: value[k] for k in ("scope", "source")}) != row["id"]:
            raise ValueError("Reuse entry identity changed")
        record = lineage.audit(self.archive, value["record"])
        view = load(self.archive / "views", record["snapshot"])
        if (
            record["scope"] != value["scope"]
            or view["proposal"]["source_hash"] != value["source"]
            or view["evaluator_hash"] != self.config["evaluator"]
        ):
            raise ValueError("Reuse entry source, scope or evaluator changed")
        return value, view

    def register(self, record_id):
        """Approve a verified source for ordinary retesting, NOT automatic adoption.

        Registration order is the explicit preference order, never a ranking of
        historical scores. Duplicate source/scope records share one identity and
        cannot erase failures or gain priority. Disable entries to withdraw them.
        """
        self._current()
        record = lineage.audit(self.archive, record_id)
        if record["snapshot"] is None:
            raise ValueError("Only independently qualified artifacts may be registered")
        view = load(self.archive / "views", record["snapshot"])
        if view["evaluator_hash"] != self.config["evaluator"]:
            raise ValueError("Reuse seed evaluator changed; revalidate its fixed source")
        value = {
            "record": record_id,
            "scope": record["scope"],
            "source": view["proposal"]["source_hash"],
        }
        identity = digest({k: value[k] for k in ("scope", "source")})
        with self._transaction():
            if not self.db.execute("SELECT 1 FROM entries WHERE id=?", (identity,)).fetchone():
                if (
                    self.db.execute("SELECT count(*) FROM entries").fetchone()[0]
                    >= self.config["max_entries"]
                ):
                    raise ValueError("Reuse entry limit reached")
                self.db.execute(
                    "INSERT INTO entries(id,data,enabled) VALUES(?,?,1)",
                    (identity, json.dumps(value)),
                )
        return identity

    def _blocked(self, entry, target, context):
        return (
            self.db.execute(
                "SELECT 1 FROM trials WHERE entry=? AND "
                "((target=? AND status!='helpful') OR "
                "(context=? AND status IN ('pending','failed_on_target'))) LIMIT 1",
                (entry, target, context),
            ).fetchone()
            is not None
        )

    def prepare(self, scope, task, task_profile, artifacts, *, max_actions=4):
        """Reserve at most one offer before work, without model/GPU calls.

        Failed correctness/performance blocks the source on this exact task;
        non-improvement and incomplete feedback also distinguish the incumbent.
        Success may be reused. No claims generalize to other shapes or runtimes.
        Only a snapshot crosses into the worker, never this database or archive.
        """
        self._current()
        if type(max_actions) is not int or not 2 <= max_actions <= 8:
            raise ValueError(
                "Reuse requires 2–8 fixed actions, including the ordinary reuse action"
            )
        task, artifacts = Path(task).resolve(), Path(artifacts).resolve()
        if (
            artifacts.is_relative_to(task)
            or self.root.is_relative_to(task)
            or self.archive.is_relative_to(task)
        ):
            raise ValueError("Reuse memory, archive and evidence must be outside the task")
        if (artifacts / "search-trace.json").exists():
            raise ValueError("Prepare requires a future unused research trace")
        if not self._get("enabled"):
            return {"ticket": None, "snapshot": None, "reason": "disabled", "skipped": []}
        if (
            self.db.execute("SELECT count(*) FROM trials").fetchone()[0]
            >= self.config["max_trials"]
        ):
            return {"ticket": None, "snapshot": None, "reason": "trial_limit", "skipped": []}
        cfg, facts = config(task), profile(task_profile)
        reference, initial = _source(task / "task.py"), source_id(_source(task / "candidate.py"))
        target = _target(cfg, reference, initial, scope)
        context = digest({"task": task_key(cfg, reference), "scope": scope})
        skipped = []
        if self.db.execute(
            "SELECT 1 FROM trials WHERE target=? AND status='pending'", (target,)
        ).fetchone():
            return {"ticket": None, "snapshot": None, "reason": "target_pending", "skipped": []}
        # Raw evidence audits run outside the short metadata transaction. All
        # eligibility/pending/disabled checks are repeated while reserving.
        for row in self.db.execute("SELECT * FROM entries WHERE enabled=1 ORDER BY seq").fetchall():
            try:
                metadata = json.loads(row["data"])
                if digest({k: metadata[k] for k in ("scope", "source")}) != row["id"]:
                    raise ValueError("Reuse entry identity changed")
                if metadata["scope"] != scope:
                    continue
                if metadata["source"] == initial:
                    skipped.append({"entry": row["id"], "reason": "already_incumbent"})
                    continue
                if self._blocked(row["id"], target, context):
                    skipped.append({"entry": row["id"], "reason": "prior_unresolved_or_unhelpful"})
                    continue
                entry, view = self._entry(row)
                if (
                    view["profile"] != facts
                    or view["compatibility"] != {k: cfg[k] for k in COMPATIBILITY}
                    or view["proposal"]["task_source_hash"] != source_id(reference)
                ):
                    skipped.append({"entry": row["id"], "reason": "incompatible"})
                    continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                skipped.append(
                    {"entry": row["id"], "reason": "invalid_evidence", "error": str(exc)}
                )
                continue
            plan = {
                "entry": row["id"],
                "record": entry["record"],
                "target": target,
                "context": context,
                "scope": scope,
                "task": str(task),
                "artifacts": str(artifacts),
                "profile": facts,
                "max_actions": max_actions,
                "prepared_at": time.time(),
            }
            identity = digest(plan)
            with self._transaction():
                if not self._get("enabled"):
                    return {
                        "ticket": None,
                        "snapshot": None,
                        "reason": "disabled",
                        "skipped": skipped,
                    }
                if (
                    self.db.execute("SELECT count(*) FROM trials").fetchone()[0]
                    >= self.config["max_trials"]
                ):
                    return {
                        "ticket": None,
                        "snapshot": None,
                        "reason": "trial_limit",
                        "skipped": skipped,
                    }
                if (
                    not self.db.execute(
                        "SELECT enabled FROM entries WHERE id=?", (row["id"],)
                    ).fetchone()[0]
                    or self.db.execute(
                        "SELECT 1 FROM trials WHERE target=? AND status='pending' LIMIT 1",
                        (target,),
                    ).fetchone()
                    or self._blocked(row["id"], target, context)
                ):
                    continue
                if self.db.execute(
                    "SELECT 1 FROM trials WHERE artifacts=?", (str(artifacts),)
                ).fetchone():
                    raise ValueError("Research evidence path already reserved")
                self.db.execute(
                    "INSERT INTO trials VALUES(?,?,?,?,?,?,'pending',NULL,NULL)",
                    (identity, row["id"], target, context, str(artifacts), json.dumps(plan)),
                )
            return {
                "ticket": identity,
                "snapshot": {
                    "path": str(self.archive / "views" / f"{digest(view)}.json"),
                    "identity": digest(view),
                    "profile": facts,
                },
                "reason": "offered_for_retesting",
                "skipped": skipped,
            }
        return {"ticket": None, "snapshot": None, "reason": "no_offer", "skipped": skipped}

    def _ticket(self, identity):
        row = self.db.execute("SELECT * FROM trials WHERE id=?", (identity,)).fetchone()
        if row is None or digest(json.loads(row["data"])) != identity:
            raise ValueError("Unknown or changed reuse ticket")
        return row, json.loads(row["data"])

    def abandon(self, identity, reason):
        """Keep an incomplete use visible and consumed; never infer quality from it."""
        text(reason, "incomplete reuse reason", 1200)
        return self._finish(identity, {"status": "incomplete", "reason": reason, "record": None})

    def _finish(self, identity, report):
        receipt = save(self.root / "feedback", {"ticket": identity, **report})
        with self._transaction():
            row, _ = self._ticket(identity)
            if row["status"] != "pending":
                if row["receipt"] == receipt:
                    return load(self.root / "feedback", receipt)
                raise ValueError("Reuse ticket already consumed")
            self.db.execute(
                "UPDATE trials SET status=?,receipt=?,record=? WHERE id=?",
                (report["status"], receipt, report["record"], identity),
            )
        return {"ticket": identity, **report}

    def observe(self, identity, *, confirmation_run):
        """Rebuild original receipts after the independent final; no new evaluation.

        A changed selected source can form a lineage candidate, but is never
        registered automatically. Helpful here describes this source's measured
        search contribution, not a causal saving, promotion, or campaign ROI.
        """
        self._current()
        row, plan = self._ticket(identity)
        confirmation_run = str(Path(confirmation_run).resolve())
        if row["status"] != "pending":
            prior = load(self.root / "feedback", row["receipt"])
            if prior.get("confirmation_run") != confirmation_run:
                raise ValueError("Reuse ticket already consumed")
            lineage.audit(self.archive, prior["record"])
            if (
                digest(read(Path(plan["artifacts"]) / "search-trace.json", MAX_EPISODE_BYTES))
                != prior["trace_hash"]
            ):
                raise ValueError("Reuse trace changed after feedback")
            return prior
        trace_path = Path(plan["artifacts"]) / "search-trace.json"
        trace = read(trace_path, MAX_EPISODE_BYTES)
        if trace["max_actions"] != plan["max_actions"] or trace.get("adaptive_artifact_budget"):
            raise ValueError("Reuse feedback requires the reserved fixed action budget")
        record_id = lineage.collect(
            self.archive,
            plan["task"],
            plan["artifacts"],
            plan["profile"],
            confirmation_run=confirmation_run,
            parent=plan["record"],
        )
        record = lineage.audit(self.archive, record_id)
        episode = load(self.archive / "episodes", record["episode"])
        if (
            read(trace_path, MAX_EPISODE_BYTES) != trace
            or len(episode["actions"]) > plan["max_actions"]
        ):
            raise ValueError("Reuse trace changed during feedback or exceeded its action budget")
        initial = episode["observations"][0]
        if (
            _target(
                episode["task_config"], episode["task_source"], initial["source"], plan["scope"]
            )
            != plan["target"]
        ):
            raise ValueError("Reuse trial changed the reserved task or initial source")
        final = episode["observations"][episode["final_confirmation"]]
        observed = [
            episode["observations"][v["observation"]] for v in record["parent_observations"]
        ]
        if not record["parent_activated"] or record["scope"] != plan["scope"]:
            status = "not_activated"
        elif any(
            o["evaluator_hash"] != initial["evaluator_hash"]
            or runtime_key(o["environment"]) != runtime_key(initial["environment"])
            for o in [*observed, final]
        ):
            status = "context_changed"
        elif any(
            o["correctness"] != "pass" or not o["decision"]["eligible_for_best"] for o in observed
        ):
            status = "failed_on_target"
        elif not observed or observed[0]["quality"] <= initial["quality"]:
            status = "unhelpful_on_target"
        elif final["correctness"] != "pass" or not final["decision"]["eligible_for_best"]:
            status = "inconclusive_final"
        else:
            status = "helpful"
        return self._finish(
            identity,
            {
                "status": status,
                "confirmation_run": confirmation_run,
                "trace_hash": digest(trace),
                "record": record_id,
                "formed_candidate": record_id if record["kind"] == "candidate" else None,
                "initial_quality": initial["quality"],
                "reuse_observations": record["parent_observations"],
                "final_source": final["source"],
                "final_eligible": final["decision"]["eligible_for_best"],
                "reported_cost": episode["reported_cost"],
                "authority": "Task-local observed reuse outcome; no counterfactual speedup, token saving or adoption claim.",
            },
        )
