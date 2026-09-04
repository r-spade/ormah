"""Shared seq-watermark helpers for incremental background jobs (#81).

Generalizes the pattern auto_linker introduced in #26: a job records the seq
of the last fully-processed node under a key in ``meta`` and selects only
``seq > watermark`` on the next run. auto_linker still uses its private copy
(kept untouched to avoid conflicts with queued PRs); migrating it here is a
follow-up.
"""

from __future__ import annotations

DUPLICATE_WATERMARK_KEY = "duplicate_check_watermark"
CONFLICT_WATERMARK_KEY = "conflict_check_watermark"


def get_watermark(conn, key: str) -> int:
    """Return the seq of the last fully-processed node for *key*, or 0."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def set_watermark(engine, key: str, seq: int) -> None:
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, str(seq)),
        )
