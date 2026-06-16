from ormah.index.db import Database


def test_migrate_adds_archived_at_and_backfills(tmp_path):
    # Build a legacy-shaped DB: nodes table without archived_at, one archival row.
    db = Database(tmp_path / "index.db")
    db.init_schema()
    db.conn.execute("ALTER TABLE nodes DROP COLUMN archived_at")
    db.conn.execute(
        "INSERT INTO nodes (id, type, tier, source, created, updated, last_accessed, "
        "file_path, file_hash) VALUES "
        "('n1','fact','archival','agent:test','2026-01-01T00:00:00Z','2026-02-01T00:00:00Z',"
        "'2026-02-01T00:00:00Z','/x.md','abc')"
    )
    db.conn.commit()

    db._migrate()

    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(nodes)").fetchall()]
    assert "archived_at" in cols
    row = db.conn.execute("SELECT archived_at FROM nodes WHERE id='n1'").fetchone()
    assert row["archived_at"] == "2026-02-01T00:00:00Z"  # backfilled from updated
    db.close()
