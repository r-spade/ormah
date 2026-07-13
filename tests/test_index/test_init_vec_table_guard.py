"""Guard: a dim mismatch must never silently DROP a populated vector store."""
from __future__ import annotations

import numpy as np
import pytest

from ormah.embeddings.vector_store import VectorStore


def _count(engine) -> int:
    return engine.db.conn.execute("SELECT count(*) FROM node_vectors").fetchone()[0]


def test_dim_mismatch_on_populated_store_raises_not_drops(engine):
    dim = engine.settings.embedding_dim
    VectorStore(engine.db).upsert("n1", np.ones(dim, dtype=np.float32))
    before = _count(engine)
    assert before > 0

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        engine.db.init_vec_table(dim + 256)  # e.g. config default 768 vs stored 1024

    assert _count(engine) == before  # nothing dropped


def test_dim_mismatch_on_empty_store_recreates(engine):
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")
    new_dim = engine.settings.embedding_dim + 256

    engine.db.init_vec_table(new_dim)  # empty → safe to recreate, no raise

    row = engine.db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='node_vectors'"
    ).fetchone()
    assert f"FLOAT[{new_dim}]" in row[0]


def test_allow_drop_authorizes_reindex(engine):
    dim = engine.settings.embedding_dim
    VectorStore(engine.db).upsert("n1", np.ones(dim, dtype=np.float32))

    engine.db.init_vec_table(dim + 256, allow_drop=True)  # explicit → drops

    assert _count(engine) == 0
    row = engine.db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='node_vectors'"
    ).fetchone()
    assert f"FLOAT[{dim + 256}]" in row[0]


def test_unparseable_ddl_raises_and_preserves_rows(engine):
    """A node_vectors table whose DDL has no FLOAT[dim] (corrupt/foreign schema)
    must fail closed: never guess-drop, never boot into broken vector search."""
    with engine.db.transaction() as conn:
        conn.execute("DROP TABLE node_vectors")
        conn.execute("CREATE TABLE node_vectors (id TEXT PRIMARY KEY, embedding BLOB)")
        conn.execute("INSERT INTO node_vectors (id, embedding) VALUES ('n1', x'00')")

    with pytest.raises(RuntimeError, match="FLOAT"):
        engine.db.init_vec_table(engine.settings.embedding_dim)

    assert _count(engine) == 1  # rows preserved for inspection/recovery


def test_startup_wiring_respects_reindex_flag(settings):
    """MemoryEngine.__init__ authorizes the drop only when the flag equals the
    configured dim; a stale flag from a previous migration keeps refusing."""
    from ormah.engine.memory_engine import MemoryEngine

    eng = MemoryEngine(settings)
    dim = settings.embedding_dim
    VectorStore(eng.db).upsert("n1", np.ones(dim, dtype=np.float32))
    eng.db.close()

    # Stale flag: authorizes the OLD dim while config asks a new one → refuse.
    settings.embedding_dim = dim + 256
    settings.reindex_on_dim_change = dim  # stale value != new dim
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        MemoryEngine(settings)

    # Correct flag: equals the NEW dim → authorized drop.
    settings.reindex_on_dim_change = dim + 256
    eng2 = MemoryEngine(settings)
    assert eng2.db.conn.execute("SELECT count(*) FROM node_vectors").fetchone()[0] == 0
    eng2.db.close()


def test_authorization_is_one_shot(engine):
    """A consumed reindex authorization must not silently re-authorize a second
    destructive drop — the stale flag left in .env would otherwise reopen the
    data-loss path. (council: codex high)"""
    import numpy as np

    from ormah.embeddings.vector_store import VectorStore

    dim = engine.settings.embedding_dim
    VectorStore(engine.db).upsert("n1", np.ones(dim, dtype=np.float32))

    # First authorized migration: allowed, drops, and consumes the authorization.
    engine.db.init_vec_table(dim + 256, allow_drop=True)
    assert _count(engine) == 0
    marker = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'reindex_consumed_dim'"
    ).fetchone()
    assert marker["value"] == str(dim + 256)

    # Store gets repopulated at the new dim, then a DIFFERENT dim shows up while the
    # stale flag still authorizes dim+256. The spent authorization must NOT allow it.
    VectorStore(engine.db).upsert("n2", np.ones(dim + 256, dtype=np.float32))
    with engine.db.transaction() as conn:
        conn.execute("DROP TABLE node_vectors")
        conn.execute(
            f"CREATE VIRTUAL TABLE node_vectors USING vec0(id TEXT PRIMARY KEY, "
            f"embedding FLOAT[{dim}])"
        )
    VectorStore(engine.db).upsert("n3", np.ones(dim, dtype=np.float32))
    assert _count(engine) == 1

    with pytest.raises(RuntimeError, match="already"):
        engine.db.init_vec_table(dim + 256, allow_drop=True)  # stale flag, spent auth

    assert _count(engine) == 1  # nothing dropped


def test_empty_table_does_not_consume_authorization(engine):
    """An empty table is recreated freely and must not burn the one-shot token."""
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors")

    engine.db.init_vec_table(engine.settings.embedding_dim + 256)  # no allow_drop needed

    marker = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'reindex_consumed_dim'"
    ).fetchone()
    assert marker is None  # authorization untouched


def test_concurrent_init_vec_table_race_consumes_authorization_once(engine):
    """TOCTOU: two concurrent init_vec_table(allow_drop=True) calls against the
    same populated store, both racing the SAME target dim, must spend the
    one-shot authorization's DROP exactly once. (A second call landing on an
    already-migrated table is a legitimate no-op — dims now match, nothing to
    authorize — so success/failure counts don't distinguish fixed from broken;
    the DROP TABLE count does: pre-fix, both threads independently decide
    "not yet consumed" from a stale read and each issues their own DROP,
    corrupting the freshly-recreated table — proven below by a live probe
    against the pre-fix code that gets a bare sqlite3 OperationalError on the
    second DROP.)

    Uses a barrier on the marker SELECT (via sqlite3's per-connection trace
    callback, since sqlite3.Connection is an immutable C type — its methods
    cannot be monkeypatched) to force both threads past the "not yet consumed"
    check before either drops. Fixed code: reads happen inside the same
    BEGIN IMMEDIATE as the drop, so the second thread's read only happens
    after the first thread's transaction commits — the barrier times out and
    is a no-op, and the second thread observes the table already migrated.
    (council: codex high — check-then-drop TOCTOU)"""
    import threading

    dim = engine.settings.embedding_dim
    new_dim = dim + 256
    VectorStore(engine.db).upsert("n1", np.ones(dim, dtype=np.float32))

    barrier = threading.Barrier(2)
    drop_count = 0
    drop_lock = threading.Lock()

    def trace(sql: str) -> None:
        nonlocal drop_count
        upper = sql.strip().upper()
        if "REINDEX_CONSUMED_DIM" in upper and upper.startswith("SELECT"):
            try:
                barrier.wait(timeout=2)
            except threading.BrokenBarrierError:
                pass  # fixed code: second thread arrives only after the first commits
        if upper.startswith("DROP TABLE") and "NODE_VECTORS" in upper:
            with drop_lock:
                drop_count += 1

    # Worker threads get fresh, thread-local connections lazily via db.conn;
    # wrap connection creation (a plain Python method, unlike sqlite3.Connection
    # itself) to attach the trace callback to each new connection.
    orig_new_connection = engine.db._new_connection

    def new_connection_with_trace():
        conn = orig_new_connection()
        conn.set_trace_callback(trace)
        return conn

    results: dict[str, object] = {}

    def worker(name: str):
        try:
            engine.db.init_vec_table(new_dim, allow_drop=True)
            results[name] = "ok"
        except Exception as e:  # noqa: BLE001 — pre-fix races can raise sqlite3 errors too
            results[name] = e

    engine.db._new_connection = new_connection_with_trace
    try:
        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
    finally:
        engine.db._new_connection = orig_new_connection

    assert drop_count == 1, (
        f"expected exactly one DROP TABLE node_vectors, saw {drop_count}: {results}"
    )

    marker = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'reindex_consumed_dim'"
    ).fetchone()
    assert marker["value"] == str(new_dim)  # written exactly once
    assert _count(engine) == 0  # exactly one drop+recreate, not two
