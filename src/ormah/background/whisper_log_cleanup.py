"""Bounded retention for high-volume whisper candidate diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from ormah.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


def run_whisper_log_cleanup(
    engine: MemoryEngine,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Delete one bounded batch of stale, unreferenced rejected candidates.

    Injected rows remain the durable denominator for whisper health. Rejected
    rows that received affinity or signal feedback also remain because their
    exact whisper_log IDs are part of the learning audit trail.
    """
    settings = engine.settings
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(
        days=settings.whisper_log_rejected_retention_days
    )
    cutoff_iso = cutoff.isoformat()
    batch_size = settings.whisper_log_cleanup_batch_size

    with engine.db.transaction() as conn:
        rows = conn.execute(
            """
            SELECT wl.id, wl.retrieval_event_id
            FROM whisper_log wl
            WHERE wl.was_injected = 0
              AND wl.logged_at < ?
              AND NOT EXISTS (
                  SELECT 1 FROM affinity a WHERE a.whisper_log_id = wl.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM signals s WHERE s.whisper_log_id = wl.id
              )
            ORDER BY wl.logged_at ASC, wl.id ASC
            LIMIT ?
            """,
            (cutoff_iso, batch_size),
        ).fetchall()
        if not rows:
            return {"candidate_rows_deleted": 0, "events_deleted": 0}

        candidate_ids = [int(row["id"]) for row in rows]
        event_ids = {
            int(row["retrieval_event_id"])
            for row in rows
            if row["retrieval_event_id"] is not None
        }
        marks = ",".join("?" for _ in candidate_ids)
        conn.execute(
            f"DELETE FROM whisper_log WHERE id IN ({marks})",
            candidate_ids,
        )

        events_deleted = 0
        if event_ids:
            event_marks = ",".join("?" for _ in event_ids)
            cursor = conn.execute(
                f"""
                DELETE FROM retrieval_events
                WHERE id IN ({event_marks})
                  AND NOT EXISTS (
                      SELECT 1 FROM whisper_log wl
                      WHERE wl.retrieval_event_id = retrieval_events.id
                  )
                """,
                tuple(event_ids),
            )
            events_deleted = cursor.rowcount

    result = {
        "candidate_rows_deleted": len(candidate_ids),
        "events_deleted": events_deleted,
    }
    logger.info(
        "Whisper log cleanup deleted %d rejected candidates and %d orphan events",
        result["candidate_rows_deleted"],
        result["events_deleted"],
    )
    return result
