"""FSRS retrievability-based tier demotion for stale working memories."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from ormah.models.node import Tier, UpdateNodeRequest

logger = logging.getLogger(__name__)


def _compute_retrievability(row, now: datetime) -> float | None:
    """Return FSRS retrievability for a node row, or None if uncomputable."""
    stability = row["stability"] if row["stability"] else 1.0
    anchor_str = row["last_review"] or row["last_accessed"]
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except (ValueError, TypeError):
        return None
    days_since = max((now - anchor).total_seconds() / 86400, 0.001)
    return math.exp(-days_since / stability)


def run_decay(engine) -> None:
    """Auto-demote working nodes whose FSRS retrievability drops below threshold.

    Also writes proactive decay alerts for core and working nodes whose
    retrievability has dropped below ``decay_alert_threshold`` so that
    context_builder can whisper a warning before demotion occurs.
    """
    try:
        settings = engine.settings
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # One-time cleanup: remove legacy pending decay proposals
        with engine.db.transaction() as conn:
            conn.execute(
                "DELETE FROM proposals WHERE type = 'decay' AND status = 'pending'"
            )

        # --- Proactive decay alerts ------------------------------------------
        if getattr(settings, "decay_alert_enabled", True):
            alert_threshold = getattr(settings, "decay_alert_threshold", 0.40)
            refire_days = getattr(settings, "decay_alert_refire_days", 7)
            user_node_id = getattr(engine, "user_node_id", None)

            alert_rows = engine.db.conn.execute(
                "SELECT id, tier, importance, stability, last_review, last_accessed "
                "FROM nodes WHERE tier IN ('core', 'working')"
            ).fetchall()

            alerted = 0
            for row in alert_rows:
                if row["id"] == user_node_id:
                    continue
                r = _compute_retrievability(row, now)
                if r is None or r >= alert_threshold:
                    continue

                # Suppress if already alerted recently
                recent = engine.db.conn.execute(
                    """
                    SELECT id FROM decay_alert_log
                    WHERE node_id = ? AND acknowledged = 0
                      AND alerted_at > datetime(?, '-' || ? || ' days')
                    LIMIT 1
                    """,
                    (row["id"], now_iso, refire_days),
                ).fetchone()
                if recent:
                    continue

                with engine.db.transaction() as conn:
                    conn.execute(
                        "INSERT INTO decay_alert_log "
                        "(node_id, alerted_at, retrievability, tier) "
                        "VALUES (?, ?, ?, ?)",
                        (row["id"], now_iso, r, row["tier"]),
                    )
                alerted += 1

            if alerted:
                logger.info("Decay manager created %d proactive alerts", alerted)

        # --- Demotion --------------------------------------------------------
        rows = engine.db.conn.execute(
            "SELECT id, importance, stability, last_review, last_accessed "
            "FROM nodes WHERE tier = 'working'"
        ).fetchall()

        if not rows:
            return

        user_node_id = getattr(engine, "user_node_id", None)
        importance_threshold = settings.decay_importance_threshold
        r_threshold = settings.fsrs_decay_threshold

        demoted = 0
        for row in rows:
            if row["id"] == user_node_id:
                continue
            # Skip high-importance nodes
            node_importance = row["importance"] if row["importance"] is not None else 0.5
            if node_importance >= importance_threshold:
                continue

            r = _compute_retrievability(row, now)
            if r is None or r >= r_threshold:
                continue

            result = engine.update_node(row["id"], UpdateNodeRequest(tier=Tier.archival))
            if result:
                demoted += 1

        if demoted:
            logger.info("Decay manager demoted %d nodes to archival", demoted)

    except Exception as e:
        logger.warning("Decay manager failed: %s", e)
