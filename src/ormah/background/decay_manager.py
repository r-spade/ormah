"""FSRS retrievability-based tier demotion for stale working memories."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ormah.background.memory_lock import serialized_memory_job
from ormah.engine.lifecycle import retrievability, safe_stability
from ormah.models.node import Tier, UpdateNodeRequest

logger = logging.getLogger(__name__)


@serialized_memory_job
def run_decay(engine) -> None:
    """Auto-demote working nodes whose FSRS retrievability drops below threshold."""
    try:
        settings = engine.settings
        now = datetime.now(timezone.utc)

        # One-time cleanup: remove legacy pending decay proposals
        with engine.db.transaction() as conn:
            conn.execute(
                "DELETE FROM proposals WHERE type = 'decay' AND status = 'pending'"
            )

        rows = engine.db.conn.execute(
            "SELECT id, stability, last_accessed, created "
            "FROM nodes WHERE tier = 'working'"
        ).fetchall()

        if not rows:
            return

        user_node_id = getattr(engine, "user_node_id", None)
        r_threshold = settings.fsrs_decay_threshold

        demoted = 0
        for row in rows:
            if row["id"] == user_node_id:
                continue
            # Decay follows confirmed-use/creation recency. Importance remains
            # a ranking/display/core-cap signal, never a hidden immortality gate.
            stability = safe_stability(
                row["stability"],
                settings.fsrs_initial_stability,
            )
            anchor_str = row["last_accessed"] or row["created"]
            try:
                anchor = datetime.fromisoformat(anchor_str)
            except (ValueError, TypeError):
                continue
            days_since = max((now - anchor).total_seconds() / 86400, 0.001)
            node_retrievability = retrievability(
                days_since,
                stability,
                fallback=settings.fsrs_initial_stability,
            )

            if node_retrievability >= r_threshold:
                continue

            result = engine.update_node(row["id"], UpdateNodeRequest(tier=Tier.archival))
            if result:
                demoted += 1

        if demoted:
            logger.info("Decay manager demoted %d nodes to archival", demoted)

    except Exception as e:
        logger.warning("Decay manager failed: %s", e)
