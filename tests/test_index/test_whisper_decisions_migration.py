"""The matched_pattern column must appear on pre-#143 databases too."""

from __future__ import annotations

from ormah.index.db import Database


def test_migration_adds_matched_pattern_to_an_existing_db(tmp_path):
    db_path = tmp_path / "old.db"  # Database.__init__ takes a Path, not a str (db.py:21)

    # A pre-#143 whisper_decisions: same shape, no matched_pattern.
    legacy = Database(db_path)
    with legacy.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS whisper_decisions")
        conn.execute(
            "CREATE TABLE whisper_decisions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, space TEXT,"
            "  prompt_hash TEXT NOT NULL, intent TEXT, outcome TEXT NOT NULL,"
            "  candidate_count INTEGER DEFAULT 0, injected_count INTEGER DEFAULT 0,"
            "  max_gate_score REAL, logged_at TEXT NOT NULL)"
        )
    legacy.close()

    migrated = Database(db_path)  # Path, never str — db.py:23 calls db_path.parent.mkdir()
    migrated.init_schema()
    cols = {
        row[1]
        for row in migrated.conn.execute("PRAGMA table_info(whisper_decisions)").fetchall()
    }
    migrated.close()

    assert "matched_pattern" in cols
