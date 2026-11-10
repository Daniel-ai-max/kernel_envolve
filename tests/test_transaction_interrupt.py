"""Deterministic deadline delivery around actual SQLite commit boundaries."""

import sqlite3

import pytest

from kernel_envolve.manager.store import Store as Manager
from kernel_envolve.rsi.managed import _DeadlineExpired
from kernel_envolve.rsi.store import Store as RSI


class InterruptedTransaction:
    def __init__(self, connection, error, *, after=True, statement="COMMIT"):
        self.connection, self.error, self.after = connection, error, after
        self.statements = []
        self.statement, self.armed = statement, True

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, statement, *args):
        self.statements.append(statement)
        interrupt = statement == self.statement and self.armed
        if interrupt:
            self.armed = False
        if interrupt and not self.after:
            raise self.error
        value = self.connection.execute(statement, *args)
        if interrupt:
            raise self.error
        return value


@pytest.mark.parametrize("store_type", [Manager, RSI])
@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("statement", ["BEGIN IMMEDIATE", "COMMIT"])
def test_deadline_survives_transaction_boundary_without_masking(store_type, after, statement):
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.execute("CREATE TABLE evidence(value INTEGER)")
    error = _DeadlineExpired("the original deadline")
    store = object.__new__(store_type)
    store.db = InterruptedTransaction(db, error, after=after, statement=statement)
    try:
        with pytest.raises(_DeadlineExpired) as raised, store.transaction():
            store.db.execute("INSERT INTO evidence VALUES(1)")
        assert raised.value is error
        assert not db.in_transaction
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == int(
            after and statement == "COMMIT"
        )
        assert store.db.statements.count("COMMIT") == int(statement == "COMMIT")
        assert store.db.statements.count("ROLLBACK") == int(
            after if statement == "BEGIN IMMEDIATE" else not after
        )
    finally:
        db.close()


@pytest.mark.parametrize("store_type", [Manager, RSI])
def test_exception_in_body_still_rolls_back(store_type):
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.execute("CREATE TABLE evidence(value INTEGER)")
    store = object.__new__(store_type)
    store.db = db
    try:
        with pytest.raises(ValueError, match="body failure"), store.transaction():
            db.execute("INSERT INTO evidence VALUES(1)")
            raise ValueError("body failure")
        assert not db.in_transaction
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.parametrize("store_type", [Manager, RSI])
def test_rejected_nested_transaction_does_not_rollback_the_caller(store_type):
    db = sqlite3.connect(":memory:", isolation_level=None)
    db.execute("CREATE TABLE evidence(value INTEGER)")
    store = object.__new__(store_type)
    store.db = db
    try:
        with store.transaction():
            db.execute("INSERT INTO evidence VALUES(1)")
            with pytest.raises(sqlite3.OperationalError), store.transaction():
                pytest.fail("Nested transaction must not run")
            assert db.in_transaction
        assert db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1
    finally:
        db.close()
