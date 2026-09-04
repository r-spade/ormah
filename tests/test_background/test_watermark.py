"""Tests for the shared seq-watermark helpers (#81)."""

from __future__ import annotations

from ormah.background.watermark import get_watermark, set_watermark


def test_default_is_zero(engine):
    assert get_watermark(engine.db.conn, "duplicate_check_watermark") == 0


def test_roundtrip(engine):
    set_watermark(engine, "duplicate_check_watermark", 42)
    assert get_watermark(engine.db.conn, "duplicate_check_watermark") == 42


def test_overwrite(engine):
    set_watermark(engine, "conflict_check_watermark", 7)
    set_watermark(engine, "conflict_check_watermark", 9)
    assert get_watermark(engine.db.conn, "conflict_check_watermark") == 9


def test_keys_are_independent(engine):
    set_watermark(engine, "duplicate_check_watermark", 5)
    assert get_watermark(engine.db.conn, "conflict_check_watermark") == 0
    # and independent of the auto_linker's key
    assert get_watermark(engine.db.conn, "auto_link_watermark") == 0


def test_malformed_value_reads_as_zero(engine):
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("duplicate_check_watermark", "not-a-number"),
        )
    assert get_watermark(engine.db.conn, "duplicate_check_watermark") == 0


def test_full_rebuild_resets_all_incremental_watermarks(engine):
    """Mass reindex re-allocates seq; every incremental cursor must be cleared
    (upstream already does this for auto_link_watermark, builder.py:36)."""
    set_watermark(engine, "duplicate_check_watermark", 42)
    set_watermark(engine, "conflict_check_watermark", 43)

    engine.builder.full_rebuild()

    assert get_watermark(engine.db.conn, "duplicate_check_watermark") == 0
    assert get_watermark(engine.db.conn, "conflict_check_watermark") == 0
