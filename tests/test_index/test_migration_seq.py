"""Tests for nodes.seq column migration and backfill."""

from __future__ import annotations


def test_seq_column_backfilled_by_created(engine):
    """Existing nodes get a monotonic seq ordered by created ASC."""
    # _create_pair inserts two nodes via the builder
    from tests.test_background.test_auto_linker import _create_pair
    id_a, id_b = _create_pair(engine)
    rows = engine.db.conn.execute(
        "SELECT id, seq FROM nodes WHERE id IN (?, ?) ORDER BY seq", (id_a, id_b)
    ).fetchall()
    seqs = [r["seq"] for r in rows]
    assert all(s > 0 for s in seqs)
    assert seqs[0] != seqs[1]  # unique, monotonic


def test_init_schema_migrates_legacy_db_without_seq(tmp_path):
    """Regression: a pre-seq DB must migrate without 'no such column: seq'.

    schema.sql's idx_nodes_seq used to run in executescript BEFORE _migrate added the
    column, so on a real (legacy) store the server failed to start. The index now lives
    only in _migrate, after the column is guaranteed.
    """
    from ormah.index.db import Database

    db = Database(tmp_path / "legacy.db")
    try:
        # Simulate a legacy nodes table (no seq); include the columns schema.sql's other
        # indexes / FTS triggers reference, plus created for the backfill order.
        with db.transaction() as conn:
            conn.execute(
                "CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, tier TEXT, space TEXT, "
                "title TEXT, content TEXT, created TEXT NOT NULL)"
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                "INSERT INTO nodes (id, type, tier, space, title, content, created) "
                "VALUES ('old', 'fact', 'working', NULL, 't', 'c', '2020-01-01')"
            )

        db.init_schema()  # must not raise

        cols = [r[1] for r in db.conn.execute("PRAGMA table_info(nodes)").fetchall()]
        assert "seq" in cols
        idx = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_nodes_seq'"
        ).fetchone()
        assert idx is not None
        assert db.conn.execute("SELECT seq FROM nodes WHERE id='old'").fetchone()[0] == 1
        assert db.conn.execute(
            "SELECT value FROM meta WHERE key='node_seq_next'"
        ).fetchone()[0] == "2"
    finally:
        db.close()
