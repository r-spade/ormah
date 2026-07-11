"""Whisper effectiveness metrics derived from whisper_log + affinity.

Read-only aggregation for the feedback loop closed in #21 — surfaces coverage
(share of injected memories that drew any feedback) and precision (positive
share of feedback on injected memories) so whisper effectiveness stops being
unmeasurable.

Semantics & known limitations (council-reviewed):
- coverage/precision are LINKED-ONLY: they count affinity rows joined to an
  injected whisper (whisper_log_id NOT NULL AND was_injected = 1). Legacy
  pre-#21 rows (whisper_log_id IS NULL) are excluded but surfaced separately as
  `unlinked_feedback_rows` on `all_time`, so the loss is visible, not silent.
- the window filters `wl.logged_at`, which production writers always emit via
  `.isoformat()` (context_builder.py, memory_engine.py) — so lexicographic
  comparison on that column is safe. `confirmed_at` (written as datetime('now'),
  a different format) is never compared.
- exact feedback attribution uses the surfaced `whisper_log_id`, so feedback for
  an older injected row remains linked to that injected event even if newer
  held-back rows exist for the same node.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta


def _window(conn: sqlite3.Connection, since: str | None) -> dict:
    # Anchor every feedback aggregate in whisper_log via INNER JOIN with
    # was_injected = 1, so numerator and denominator share one universe. The
    # window filters wl.logged_at on BOTH sides (single cohort).
    log_filter = " AND logged_at >= ?" if since else ""
    join_filter = " AND wl.logged_at >= ?" if since else ""
    log_params: tuple = (since,) if since else ()
    join_params: tuple = (since,) if since else ()

    injected = conn.execute(
        "SELECT COUNT(*) FROM whisper_log WHERE was_injected = 1" + log_filter,
        log_params,
    ).fetchone()[0]
    feedback_rows = conn.execute(
        "SELECT COUNT(DISTINCT a.whisper_log_id) FROM affinity a "
        "JOIN whisper_log wl ON wl.id = a.whisper_log_id "
        "WHERE wl.was_injected = 1" + join_filter,
        join_params,
    ).fetchone()[0]
    pos, neg = conn.execute(
        "SELECT "
        "COALESCE(SUM(CASE WHEN a.signal = 1 THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN a.signal = -1 THEN 1 ELSE 0 END), 0) "
        "FROM affinity a "
        "JOIN whisper_log wl ON wl.id = a.whisper_log_id "
        "WHERE wl.was_injected = 1" + join_filter,
        join_params,
    ).fetchone()

    fb_total = pos + neg
    return {
        "injected": injected,
        "feedback_rows": feedback_rows,
        "coverage": feedback_rows / injected if injected else None,
        "positive": pos,
        "negative": neg,
        "precision": pos / fb_total if fb_total else None,
    }


def compute_whisper_health(conn: sqlite3.Connection, now: datetime) -> dict:
    """Return whisper coverage/precision over all_time and last_7d windows.

    ``now`` is injected (never ``datetime.now()`` inside the query) so callers
    and tests are deterministic. See module docstring for the linked-only
    semantics and known undercount.
    """
    since_7d = (now - timedelta(days=7)).isoformat()
    # Read every aggregate under one BEGIN DEFERRED snapshot. Without it, a
    # concurrent writer (FastAPI handler, session_watcher) inserting an injected
    # whisper_log + its affinity between the `injected` count and the
    # `feedback_rows` count could make coverage exceed 100%. The store runs
    # isolation_level=None (autocommit), so the transaction is opened explicitly;
    # the in_transaction guard makes this safe to call inside a caller's tx too.
    own_tx = not conn.in_transaction
    if own_tx:
        conn.execute("BEGIN DEFERRED")
    try:
        all_time = _window(conn, None)
        # Surface legacy/unattributable feedback (whisper_log_id IS NULL) so the
        # linked-only ratios don't silently hide it. all_time only — the 7d
        # cohort is injection-anchored and has no NULL side.
        all_time["unlinked_feedback_rows"] = conn.execute(
            "SELECT COUNT(*) FROM affinity WHERE whisper_log_id IS NULL"
        ).fetchone()[0]
        last_7d = _window(conn, since_7d)
    finally:
        if own_tx:
            conn.execute("COMMIT")
    return {"all_time": all_time, "last_7d": last_7d}
