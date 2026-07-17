"""Tests for MemoryEngine.backfill_embeddings (delta + schema-bump, no quarantine, #32).

Design (council R2): schema-bump re-embeds all embeddable nodes once, deletes the
stale vector of any node that fails to encode (so it becomes genuinely missing),
and advances the version unconditionally after the pass. Delta mode embeds only
nodes missing from the vector store. A permanently-failing node stays visible in
``missing`` and is retried every tick -- never dropped, never masked.
"""
from __future__ import annotations

import threading

import numpy as np

from ormah.models.node import CreateNodeRequest
from ormah.engine.memory_engine import _EMBEDDING_SCHEMA_VERSION


def _set_schema_version(engine, version: int) -> None:
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES "
            "('embedding_schema_version', ?)",
            (str(version),),
        )


def _stored_version(engine) -> int:
    return int(engine.db.conn.execute(
        "SELECT value FROM meta WHERE key='embedding_schema_version'"
    ).fetchone()["value"])


def test_backfill_delta_closes_the_gap(engine):
    ids = []
    for i in range(3):
        nid, _ = engine.remember(CreateNodeRequest(title=f"N{i}", content=f"content {i}"))
        ids.append(nid)
    _set_schema_version(engine, _EMBEDDING_SCHEMA_VERSION)  # already current -> delta mode
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (ids[0],))

    result = engine.backfill_embeddings()

    assert result["mode"] == "delta"
    assert result["missing"] == 0
    assert engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids WHERE id = ?", (ids[0],)
    ).fetchone()[0] == 1


def test_backfill_delta_is_noop_when_full(engine):
    engine.remember(CreateNodeRequest(title="Solo", content="content"))
    _set_schema_version(engine, _EMBEDDING_SCHEMA_VERSION)
    engine.backfill_embeddings()  # settle any startup-created node

    result = engine.backfill_embeddings()

    assert result["mode"] == "delta"
    assert result["embedded"] == 0


def test_backfill_schema_bump_reembeds_all_and_bumps_version(engine):
    engine.remember(CreateNodeRequest(title="A", content="alpha"))
    engine.remember(CreateNodeRequest(title="B", content="beta"))
    _set_schema_version(engine, 1)
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    result = engine.backfill_embeddings()

    assert result["mode"] == "schema"
    assert result["missing"] == 0
    assert _stored_version(engine) == _EMBEDDING_SCHEMA_VERSION


def test_schema_bump_poison_node_stays_visible_and_advances_version(engine, monkeypatch):
    """A node that always fails to encode stays genuinely missing (its stale vector
    is deleted) and visible in `missing`; the version still advances after the pass,
    and the next delta run retries it without re-embedding the whole store."""
    engine.remember(CreateNodeRequest(title="poison", content="POISON payload"))
    engine.remember(CreateNodeRequest(title="ok1", content="fine one"))
    engine.remember(CreateNodeRequest(title="ok2", content="fine two"))
    _set_schema_version(engine, 1)

    dim = engine.settings.embedding_dim

    class _SelectiveEncoder:
        def __init__(self):
            self.encode_calls = 0

        def encode(self, text):
            self.encode_calls += 1
            if "POISON" in text:
                raise RuntimeError("poison node")
            return np.ones(dim, dtype="float32")

    enc = _SelectiveEncoder()
    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda settings: enc)

    # Schema bump: poison fails -> its stale vector is deleted -> genuinely missing.
    r1 = engine.backfill_embeddings()
    assert r1["mode"] == "schema"
    assert r1["failed"] == 1
    assert r1["missing"] == 1
    assert _stored_version(engine) == _EMBEDDING_SCHEMA_VERSION  # advances unconditionally
    assert engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors_rowids WHERE id = "
        "(SELECT id FROM nodes WHERE title='poison')"
    ).fetchone()[0] == 0

    # Delta run: retries ONLY the missing poison node (O(gap)), still fails.
    calls_after_schema = enc.encode_calls
    r2 = engine.backfill_embeddings()
    assert r2["mode"] == "delta"
    assert r2["embedded"] == 0
    assert r2["failed"] == 1
    assert r2["missing"] == 1
    assert enc.encode_calls == calls_after_schema + 1  # one retry, not a full re-embed


# ---------------------------------------------------------------------------
# Cooperative cancellation via stop_event (Task A — CRA/IA)
# ---------------------------------------------------------------------------

def test_backfill_stops_before_db_writes_when_event_set(engine):
    """A stop_event that is already set causes backfill to embed nothing."""
    engine.remember(CreateNodeRequest(title="C1", content="cancel me one"))
    engine.remember(CreateNodeRequest(title="C2", content="cancel me two"))
    # Force delta mode (version already current) and create a gap.
    _set_schema_version(engine, _EMBEDDING_SCHEMA_VERSION)
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    stop = threading.Event()
    stop.set()  # already cancelled before the call
    result = engine.backfill_embeddings(stop_event=stop)

    assert result["embedded"] == 0
    assert result["missing"] > 0


def test_backfill_completes_when_event_not_set(engine):
    """A stop_event that is never set does not interfere with normal completion."""
    engine.remember(CreateNodeRequest(title="D1", content="do embed one"))
    engine.remember(CreateNodeRequest(title="D2", content="do embed two"))
    _set_schema_version(engine, _EMBEDDING_SCHEMA_VERSION)
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    result = engine.backfill_embeddings(stop_event=threading.Event())  # never set

    assert result["missing"] == 0


def test_schema_version_not_advanced_when_interrupted(engine):
    """An interrupted schema pass must NOT advance embedding_schema_version."""
    engine.remember(CreateNodeRequest(title="S1", content="schema node one"))
    _set_schema_version(engine, 1)  # trigger schema mode (stored < current)

    before = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key='embedding_schema_version'"
    ).fetchone()

    stop = threading.Event()
    stop.set()
    engine.backfill_embeddings(stop_event=stop)

    after = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key='embedding_schema_version'"
    ).fetchone()

    before_val = before["value"] if before else None
    after_val = after["value"] if after else None
    assert before_val == after_val


def test_schema_mode_does_not_delete_when_interrupted(engine, monkeypatch):
    """Fix B: an interrupted schema pass must NOT delete stale vectors.

    The DELETE is guarded by `not interrupted` — consistent with the version-advance
    guard already in place. A cancel mid-schema stops the encode but must not execute
    the DELETE that follows it.
    """
    # Create a node that will fail encoding (simulates a real encoder error).
    nid_poison, _ = engine.remember(CreateNodeRequest(title="poison", content="POISON"))
    engine.remember(CreateNodeRequest(title="ok", content="fine"))
    # Downgrade schema version so schema mode is triggered.
    _set_schema_version(engine, 1)

    dim = engine.settings.embedding_dim

    class _FailingEncoder:
        def encode(self, text):
            if "POISON" in text:
                raise RuntimeError("encode failure")
            return np.ones(dim, dtype="float32")

    monkeypatch.setattr("ormah.embeddings.encoder.get_encoder", lambda s: _FailingEncoder())

    # Ensure the poison node already has a stale vector in the store.
    # vec0 virtual tables do not support INSERT OR REPLACE — use DELETE + INSERT.
    import struct
    stale_vec = struct.pack(f"{dim}f", *[0.5] * dim)
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid_poison,))
        conn.execute(
            "INSERT INTO node_vectors (id, embedding) VALUES (?, ?)",
            (nid_poison, stale_vec),
        )

    # Confirm the stale vector exists before the interrupted call.
    before_count = engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors WHERE id = ?", (nid_poison,)
    ).fetchone()[0]
    assert before_count == 1, "stale vector should exist before the interrupted call"

    # Run backfill with stop_event already set → interrupted=True.
    stop = threading.Event()
    stop.set()
    engine.backfill_embeddings(stop_event=stop)

    # The stale vector must NOT have been deleted (interrupted guard applied).
    after_count = engine.db.conn.execute(
        "SELECT count(*) FROM node_vectors WHERE id = ?", (nid_poison,)
    ).fetchone()[0]
    assert after_count == 1, (
        "stale vector was deleted despite interrupted=True — "
        "DELETE must be guarded by `not interrupted`"
    )
