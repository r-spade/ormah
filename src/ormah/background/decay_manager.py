"""FSRS retrievability-based tier demotion for stale working memories."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ormah import lifecycle
from ormah.background.memory_lock import serialized_memory_job
from ormah.models.node import Tier, UpdateNodeRequest
from ormah.store.file_store import UnresolvedNodeReference

logger = logging.getLogger(__name__)


@serialized_memory_job
def run_decay(engine) -> None:
    """Auto-demote working nodes whose FSRS retrievability drops below threshold.

    Retrievability alone decides (#222/#191). Importance is deliberately not a
    pre-gate: cumulative access and edge counts could push it permanently above
    any threshold, pinning a stale node to working forever. Identity (the self
    node) and core stay protected — core never enters this query.
    """
    try:
        settings = engine.settings
        now = datetime.now(timezone.utc)

        # One-time cleanup: remove legacy pending decay proposals
        with engine.db.transaction() as conn:
            conn.execute(
                "DELETE FROM proposals WHERE type = 'decay' AND status = 'pending'"
            )

        rows = engine.db.conn.execute(
            "SELECT id, stability, last_review, last_accessed "
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

            # Compute FSRS retrievability through the shared implementation (#221).
            # Anchor on use, not on the numeric stability update: the per-day
            # reinforcement cooldown can leave last_review a full window behind
            # the last use, and an actively used node must not read as stale.
            anchor_str = row["last_accessed"] or row["last_review"]
            try:
                anchor = datetime.fromisoformat(anchor_str)
                days_since = max((now - anchor).total_seconds() / 86400, 0.001)
            except (ValueError, TypeError):
                logger.warning(
                    "Decay manager skipped node %s with invalid recency anchor %r",
                    row["id"][:8],
                    anchor_str,
                )
                continue
            # Pass the stored stability raw and let lifecycle own the zero case,
            # with the SAME fallback reinforcement uses. Hardcoding 1.0 here
            # while reinforcement falls back to fsrs_initial_stability is how
            # the two paths silently disagree (council round 3, I3).
            retrievability = lifecycle.retrievability(
                days_since,
                row["stability"],
                fallback_stability=settings.fsrs_initial_stability,
            )

            if retrievability >= r_threshold:
                continue

            try:
                result = engine.update_node(row["id"], UpdateNodeRequest(tier=Tier.archival))
            except UnresolvedNodeReference as exc:
                # Its file will not parse: not demoted, and not counted as demoted.
                # One unreadable file must not end the run for every other node.
                logger.warning("Decay manager could not demote %s: %s", row["id"], exc)
                continue
            if result:
                demoted += 1

        if demoted:
            logger.info("Decay manager demoted %d nodes to archival", demoted)

    except Exception as e:
        logger.warning("Decay manager failed: %s", e)
