"""Explicit background artifact lifecycle; no production Manager hook.

Only the trusted controller may open this database. Solver sandboxes must not
mount it, its archive, validation templates or transport receipts writable.
SQLite publishes a complete catalog atomically; a caller obtains a pinned view
between trials, never by traversing evidence in the optimization hot path.
"""

import contextlib
import json
import math
import sqlite3
import time
from pathlib import Path

from ..storage import digest, evaluator_hash
from . import artifact_lineage as lineage
from . import artifact_validation as validation
from .episodes import load, save
from .model import number, read
from .trials import adapter_identity, validator_identity, verify_template


class Registry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.db = sqlite3.connect(self.root / "registry.sqlite", isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.config = self._get("config")
        if digest(self.config) != self._get("config_hash"):
            raise ValueError("Artifact registry configuration changed")
        self.archive = Path(self.config["archive"])

    @classmethod
    def create(cls, root, archive, seeds, *, max_attempts, max_seconds, max_tokens):
        root, archive = Path(root).resolve(), Path(archive).resolve()
        if root.exists():
            raise ValueError("Use a new artifact registry directory")
        if not seeds or len(seeds) > 64 or len(set(seeds)) != len(seeds):
            raise ValueError("Need distinct seed records")
        slots = {}
        for identity in seeds:
            record = lineage.audit(archive, identity)
            if record["kind"] != "seed" or record["scope"] in slots:
                raise ValueError("Need exactly one verified seed per scope")
            view = load(archive / "views", record["snapshot"])
            if view["evaluator_hash"] != evaluator_hash():
                raise ValueError(
                    "Seed evaluator differs from current evaluator; revalidate the fixed source "
                    "under the current evaluator before creating a new registry"
                )
            slots[record["scope"]] = identity
        if type(max_attempts) is not int or not 1 <= max_attempts <= 32:
            raise ValueError("Invalid admission attempt ceiling")
        number(max_seconds, "registry seconds", 0, 31 * 86400)
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("Invalid registry token budget")
        root.mkdir(parents=True)
        db = sqlite3.connect(root / "registry.sqlite")
        try:
            db.executescript("""
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE catalogs (id TEXT PRIMARY KEY, data TEXT NOT NULL,
                                       status TEXT NOT NULL, receipt TEXT);
                CREATE TABLE learning (id TEXT PRIMARY KEY, data TEXT NOT NULL,
                                       result TEXT);
                CREATE TABLE attempts (id TEXT PRIMARY KEY, data TEXT NOT NULL,
                                       status TEXT NOT NULL, report TEXT,
                                       seconds REAL NOT NULL, tokens INTEGER NOT NULL);
                CREATE TABLE proposals (key TEXT PRIMARY KEY, attempt TEXT NOT NULL);
                CREATE TABLE history (seq INTEGER PRIMARY KEY, at REAL NOT NULL,
                                      previous TEXT, current TEXT, reason TEXT NOT NULL);
            """)
            config = {
                "archive": str(archive),
                "max_attempts": max_attempts,
                "max_seconds": max_seconds,
                "max_tokens": max_tokens,
                "validator": validator_identity(),
                "evaluator": evaluator_hash(),
            }
            catalog = {"parent": None, "slots": slots, "generation": 0}
            identity = digest(catalog)
            db.execute(
                "INSERT INTO catalogs VALUES(?,?,?,NULL)", (identity, json.dumps(catalog), "seed")
            )
            for key, value in {
                "config": config,
                "config_hash": digest(config),
                "seed": identity,
                "active": identity,
                "enabled": True,
            }.items():
                db.execute("INSERT INTO meta VALUES(?,?)", (key, json.dumps(value)))
            db.commit()
        finally:
            db.close()
        return cls(root)

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def _get(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Uninitialized artifact registry")
        return json.loads(row[0])

    def _set(self, key, value):
        self.db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(value), key))

    def _current_code(self):
        if (
            self.config["validator"] != validator_identity()
            or self.config["evaluator"] != evaluator_hash()
        ):
            raise ValueError(
                "Registry validator/evaluator changed; use frozen code and a new registry"
            )

    def _holdouts(self):
        from ..storage import config

        result = set()
        for row in self.db.execute("SELECT data FROM attempts"):
            for task in json.loads(row[0])["suite"]["tasks"]:
                verify_template(task)
                root = Path(task["path"]) / "task"
                result.add(validation.task_key(config(root), (root / "task.py").read_text()))
        return result

    def catalog(self, identity):
        row = self.db.execute(
            "SELECT data,status,receipt FROM catalogs WHERE id=?", (identity,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown catalog")
        value = json.loads(row["data"])
        if digest(value) != identity:
            raise ValueError("Catalog content changed")
        return value, row["status"], row["receipt"]

    def declare_learning(self, entries):
        """Freeze ordered controller receipt locations BEFORE starting learning.

        Each entry names a scope and one future trusted trial receipt JSON path.
        That receipt later supplies task/artifacts/profile/confirmation_run and
        optional origin_task. A failed trial instead records an explicit error.
        """
        self._current_code()
        parent = self._get("active")
        if not self._get("enabled") or parent is None:
            raise ValueError("Artifact evolution is disabled")
        self.audit(parent)
        slots = self.catalog(parent)[0]["slots"]
        if not entries or len(entries) > 128:
            raise ValueError("Invalid learning phase size")
        paths = set()
        for item in entries:
            path = Path(item["receipt"])
            if (
                item["scope"] not in slots
                or not path.is_absolute()
                or path.exists()
                or str(path) in paths
            ):
                raise ValueError("Learning needs unique, unconsumed future receipt paths")
            paths.add(str(path))
        phase = {"parent": parent, "entries": entries, "declared_at": time.time()}
        identity = save(self.root / "phases", phase)
        with self.transaction():
            if self._get("active") != parent or not self._get("enabled"):
                raise ValueError("Learning parent changed")
            self.db.execute("INSERT INTO learning VALUES(?,?,NULL)", (identity, json.dumps(phase)))
        return identity

    def collect_learning(self, identity):
        """Use the first eligible changed descendant per scope in declared order."""
        self._current_code()
        row = self.db.execute("SELECT data,result FROM learning WHERE id=?", (identity,)).fetchone()
        if row is None or row["result"] is not None:
            raise ValueError("Unknown or already consumed learning phase")
        phase = json.loads(row["data"])
        if digest(phase) != identity or load(self.root / "phases", identity) != phase:
            raise ValueError("Learning declaration changed")
        parent, status, _ = self.catalog(phase["parent"])
        if status not in {"seed", "adopted"} or self._get("active") != phase["parent"]:
            raise ValueError("Learning parent is stale")
        slots, changed, observations = dict(parent["slots"]), set(), []
        holdouts = self._holdouts()
        # Do not consume a phase while any declared job has not produced a receipt.
        receipts = [read(e["receipt"]) for e in phase["entries"]]
        for entry, receipt in zip(phase["entries"], receipts, strict=True):
            scope = entry["scope"]
            observation = {
                "receipt": digest(receipt),
                "record": None,
                "error": receipt.get("error"),
            }
            if not receipt.get("error"):
                try:
                    from ..storage import config

                    root = Path(receipt["provenance"]["task"])
                    if (
                        validation.task_key(config(root), (root / "task.py").read_text())
                        in holdouts
                    ):
                        raise ValueError("Validation data cannot form a new generation")
                    record_id = lineage.collect(
                        self.archive, parent=parent["slots"][scope], **receipt["provenance"]
                    )
                    record = lineage.audit(self.archive, record_id)
                    observation["record"] = record_id
                    if (
                        record["kind"] == "candidate"
                        and record["scope"] == scope
                        and scope not in changed
                    ):
                        slots[scope] = record_id
                        changed.add(scope)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    observation["error"] = str(exc)
            observations.append(observation)
        candidate = {
            "parent": phase["parent"],
            "slots": slots,
            "generation": parent["generation"] + 1,
            "learning": identity,
            "observations": observations,
        }
        candidate_id = digest(candidate) if changed else None
        result = {"candidate": candidate_id, "observations": observations}
        with self.transaction():
            if self._get("active") != phase["parent"] or not self._get("enabled"):
                raise ValueError("Learning parent changed or evolution disabled")
            if (
                self.db.execute("SELECT result FROM learning WHERE id=?", (identity,)).fetchone()[0]
                is not None
            ):
                raise ValueError("Learning phase was concurrently consumed")
            self.db.execute(
                "UPDATE learning SET result=? WHERE id=?", (json.dumps(result), identity)
            )
            if candidate_id:
                self.db.execute(
                    "INSERT INTO catalogs VALUES(?,?,?,NULL)",
                    (candidate_id, json.dumps(candidate), "candidate"),
                )
        return result

    def _catalogs(self, identity):
        candidate = self.catalog(identity)[0]
        return {
            "candidate": candidate["slots"],
            "parent": self.catalog(candidate["parent"])[0]["slots"],
            "seed": self.catalog(self._get("seed"))[0]["slots"],
        }

    def reserve(self, candidate_id, suite):
        """Durably charge one attempt per changed parent/scope/source before validation.

        Interruption cannot free the proposal or reservation. Budgets here cover
        admission trials only; learning, GPU leases and provider-wide expenditure
        require the external campaign ledger and hard deadlines.
        """
        self._current_code()
        candidate, status, _ = self.catalog(candidate_id)
        if status != "candidate":
            raise ValueError("Catalog is not an untried candidate")
        self.audit(candidate["parent"])
        catalogs = self._catalogs(candidate_id)
        validation.check_suite(suite, catalogs["candidate"])
        proposal_keys, learning_contracts = [], set()
        for scope, record_id in candidate["slots"].items():
            record = lineage.audit(self.archive, record_id)
            if record_id != catalogs["parent"][scope]:
                if record["kind"] != "candidate" or record["parent"] != catalogs["parent"][scope]:
                    raise ValueError("Candidate does not descend from active parent")
                proposal_keys.append(record["proposal_key"])
            while record is not None:
                episode = load(self.archive / "episodes", record["episode"])
                learning_contracts.add(
                    validation.task_key(episode["task_config"], episode["task_source"])
                )
                record = (
                    load(self.archive / "records", record["parent"]) if record["parent"] else None
                )
        # Also exclude losing/nonselected formation trials, not only the winner.
        for row in self.db.execute("SELECT result FROM learning WHERE result IS NOT NULL"):
            for item in json.loads(row[0])["observations"]:
                if item["record"]:
                    record = lineage.audit(self.archive, item["record"])
                    episode = load(self.archive / "episodes", record["episode"])
                    learning_contracts.add(
                        validation.task_key(episode["task_config"], episode["task_source"])
                    )
        # Retention must cover seed capabilities and every configuration in the
        # parent's admission, not just tasks labelled "retention" by the caller.
        from ..storage import config

        required_retention = set()
        for scope, seed_id in catalogs["seed"].items():
            seed = lineage.audit(self.archive, seed_id)
            episode = load(self.archive / "episodes", seed["episode"])
            required_retention.add(
                (scope, validation.task_key(episode["task_config"], episode["task_source"]))
            )
        parent_attempt = self.catalog(candidate["parent"])[2]
        if parent_attempt:
            _, parent_plan = self._attempt(parent_attempt)
            for task in parent_plan["suite"]["tasks"]:
                verify_template(task)
                root = Path(task["path"]) / "task"
                required_retention.add(
                    (
                        task["scope"],
                        validation.task_key(config(root), (root / "task.py").read_text()),
                    )
                )
        actual_retention = set()
        for task in suite["tasks"]:
            root = Path(task["path"]) / "task"
            key = validation.task_key(config(root), (root / "task.py").read_text())
            if task["partition"] == "fresh" and key in learning_contracts:
                raise ValueError("Fresh validation reuses a formation contract")
            if task["partition"] == "retention":
                actual_retention.add((task["scope"], key))
        if not required_retention <= actual_retention:
            raise ValueError("Retention omits seed or parent-admitted capabilities")
        attempt = {
            "candidate": candidate_id,
            "parent": candidate["parent"],
            "suite": suite,
            "adapter": adapter_identity(suite),
            "validator": validator_identity(),
            "reserved_at": time.time(),
            "proposal_keys": proposal_keys,
        }
        identity = digest(attempt)
        with self.transaction():
            if not self._get("enabled") or self._get("active") != candidate["parent"]:
                raise ValueError("Disabled registry or stale admission parent")
            totals = self.db.execute(
                "SELECT count(*),coalesce(sum(seconds),0),coalesce(sum(tokens),0) FROM attempts"
            ).fetchone()
            if (
                totals[0] >= self.config["max_attempts"]
                or totals[1] + suite["max_seconds"] > self.config["max_seconds"]
                or totals[2] + suite["max_tokens"] > self.config["max_tokens"]
            ):
                raise ValueError("Admission budget exhausted")
            if not proposal_keys or any(
                self.db.execute("SELECT 1 FROM proposals WHERE key=?", (key,)).fetchone()
                for key in proposal_keys
            ):
                raise ValueError("Proposal already attempted; new receipts do not grant a retry")
            self.db.execute(
                "INSERT INTO attempts VALUES(?,?,?,NULL,?,?)",
                (
                    identity,
                    json.dumps(attempt),
                    "reserved",
                    suite["max_seconds"],
                    suite["max_tokens"],
                ),
            )
            self.db.executemany(
                "INSERT INTO proposals VALUES(?,?)", [(key, identity) for key in proposal_keys]
            )
            self.db.execute("UPDATE catalogs SET status='validating' WHERE id=?", (candidate_id,))
        return identity

    def _attempt(self, identity):
        row = self.db.execute("SELECT * FROM attempts WHERE id=?", (identity,)).fetchone()
        if row is None or digest(json.loads(row["data"])) != identity:
            raise ValueError("Unknown or changed admission reservation")
        return dict(row), json.loads(row["data"])

    def _bound_evidence(self, identity, attempt, evidence):
        if evidence["attempt"] != identity:
            raise ValueError("Evidence belongs to a different admission reservation")
        release = read(evidence["release_receipt"])
        if not attempt["reserved_at"] <= release["at"] <= time.time() + 60:
            raise ValueError("Release receipt predates admission or is in the future")
        for path in evidence["calls"]:
            call = read(path, 2 * 1024 * 1024)
            if not attempt["reserved_at"] <= call["started_at"] <= release["at"]:
                raise ValueError("Model call predates admission or follows resource release")

    def _charge_failure(self, row, evidence):
        """Unknown expenditure retains the ceiling; known overruns still count."""
        tokens, seconds, unknown = 0, 0, 0
        for path in set(evidence.get("calls", [])):
            try:
                call = read(path, 2 * 1024 * 1024)
                value = (call.get("response") or {}).get("usage", {}).get("total_tokens")
                if type(value) is not int or value < 0:
                    raise ValueError("unknown usage")
                tokens += value
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                unknown += 1
        for trial in evidence.get("trials", []):
            value = trial.get("seconds")
            if type(value) in (int, float) and math.isfinite(value) and value > 0:
                seconds += value
        return {
            "tokens": max(row["tokens"], tokens),
            "seconds": max(row["seconds"], seconds),
            "known_tokens": tokens,
            "unreadable_or_unknown_calls": unknown,
            "authority": "Conservative admission charge; not complete provider billing",
        }

    def finish(self, identity, evidence):
        """Consume the attempt, including invalid evidence. No automatic adoption."""
        with self.transaction():
            row, attempt = self._attempt(identity)
            if row["status"] != "reserved":
                raise ValueError("Admission attempt already consumed")
            self.db.execute("UPDATE attempts SET status='checking' WHERE id=?", (identity,))
        began = time.monotonic()
        report = {"evidence": evidence, "decision": None, "error": None}
        try:
            self._current_code()
            if attempt["adapter"] != adapter_identity(attempt["suite"]):
                raise ValueError("Adapter changed since reservation")
            self._bound_evidence(identity, attempt, evidence)
            report["decision"] = validation.verify(
                self.archive, self._catalogs(attempt["candidate"]), attempt["suite"], evidence
            )
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            report["error"] = str(exc)
        report["charge"] = (
            {k: report["decision"][k] for k in ("seconds", "tokens")}
            if report["decision"]
            else self._charge_failure(row, evidence)
        )
        report["background_seconds"] = time.monotonic() - began
        receipt = save(self.root / "admissions", report)
        accepted = report["error"] is None and report["decision"]["accepted"]
        with self.transaction():
            self.db.execute(
                "UPDATE attempts SET status=?,report=?,seconds=?,tokens=? WHERE id=?",
                (
                    "validated" if accepted else "rejected",
                    receipt,
                    report["charge"]["seconds"],
                    report["charge"]["tokens"],
                    identity,
                ),
            )
            self.db.execute(
                "UPDATE catalogs SET status=?,receipt=? WHERE id=? AND status='validating'",
                ("validated" if accepted else "rejected", identity, attempt["candidate"]),
            )
        return {"receipt": receipt, **report}

    def _verify_admission(self, identity):
        self._current_code()
        _, status, attempt_id = self.catalog(identity)
        if status not in {"validated", "adopted"}:
            raise ValueError("Catalog is not admitted")
        row, attempt = self._attempt(attempt_id)
        if (
            row["status"] != "validated"
            or attempt["candidate"] != identity
            or attempt["adapter"] != adapter_identity(attempt["suite"])
        ):
            raise ValueError("Admission adapter or receipt changed")
        report = load(self.root / "admissions", row["report"])
        self._bound_evidence(attempt_id, attempt, report["evidence"])
        actual = validation.verify(
            self.archive, self._catalogs(identity), attempt["suite"], report["evidence"]
        )
        if report["error"] is not None or not actual["accepted"] or actual != report["decision"]:
            raise ValueError("Original admission evidence changed or did not qualify")

    def audit(self, identity):
        """Background raw-evidence audit, including every admitted ancestor."""
        self._current_code()
        seen = set()
        while identity is not None:
            if identity in seen or len(seen) >= lineage.MAX_ANCESTORS:
                raise ValueError("Invalid catalog ancestry")
            seen.add(identity)
            catalog, status, _ = self.catalog(identity)
            if status not in {"seed", "adopted"}:
                raise ValueError("Catalog is not active-capable")
            for record_id in catalog["slots"].values():
                lineage.audit(self.archive, record_id)
            if status == "adopted":
                self._verify_admission(identity)
            identity = catalog["parent"]

    def adopt(self, identity):
        self._verify_admission(identity)
        candidate, _, _ = self.catalog(identity)
        self.audit(candidate["parent"])
        with self.transaction():
            if (
                not self._get("enabled")
                or self._get("active") != candidate["parent"]
                or self.catalog(identity)[1] != "validated"
            ):
                raise ValueError("Disabled, stale or consumed adoption")
            self.db.execute("UPDATE catalogs SET status='adopted' WHERE id=?", (identity,))
            self._set("active", identity)
            self.db.execute(
                "INSERT INTO history(at,previous,current,reason) VALUES(?,?,?,?)",
                (time.time(), candidate["parent"], identity, "admitted"),
            )
        return identity

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Enabled must be boolean")
        with self.transaction():
            self._set("enabled", enabled)

    def rollback(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Rollback requires a reason")
        current = self._get("active")
        if current is None:
            return None
        target = self.catalog(current)[0]["parent"]
        invalid = {current}
        while target is not None:
            try:
                self.audit(target)
                break
            except (OSError, ValueError, KeyError, TypeError):
                invalid.add(target)
                target = self.catalog(target)[0]["parent"]
        with self.transaction():
            if self._get("active") != current:
                raise ValueError("Active catalog changed during rollback")
            # Revocation propagates through pending and adopted descendants.
            revoked = invalid
            rows = self.db.execute("SELECT id,data FROM catalogs").fetchall()
            for _ in range(len(rows)):
                before = len(revoked)
                revoked.update(r["id"] for r in rows if json.loads(r["data"])["parent"] in revoked)
                if len(revoked) == before:
                    break
            self.db.executemany(
                "UPDATE catalogs SET status='revoked' WHERE id=?", [(i,) for i in revoked]
            )
            self._set("active", target)
            self.db.execute(
                "INSERT INTO history(at,previous,current,reason) VALUES(?,?,?,?)",
                (time.time(), current, target, reason),
            )
        return target

    def snapshot(self, scope):
        """Resolve BETWEEN trials; stale evidence revokes the active version.

        An already-running trial retains its immutable pinned input. Disabling or
        revoking affects the next resolution, not code already executing remotely.
        """
        if not self._get("enabled"):
            return None
        while (identity := self._get("active")) is not None:
            try:
                self.audit(identity)
                slots = self.catalog(identity)[0]["slots"]
                return (
                    lineage.validation_snapshot(self.archive, slots[scope])
                    if scope in slots
                    else None
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.rollback(f"Background audit failed: {exc}")
        return None
