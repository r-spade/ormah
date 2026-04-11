"""Background job: auto-promote working memories that receive positive recall hits
across multiple distinct sessions.

A ``working`` node is promoted to ``core`` when it has accumulated positive
affinity signals (signal > 0) from at least ``recall_promotion_sessions``
distinct sessions. This reflects the intuition that a memory repeatedly useful
in different contexts has earned long-term prominence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ormah.models.node import Tier

logger = logging.getLogger(__name__)


def run_recall_promotion(engine) -> None:
    """Promote working-tier nodes that have positive hits across enough sessions."""
    settings = engine.settings
    min_sessions: int = getattr(settings, "recall_promotion_sessions", 3)

    # Find working-tier nodes with sufficient multi-session positive affinity.
    rows = engine.db.conn.execute(
        """
        SELECT a.node_id, COUNT(DISTINCT a.session_id) AS session_count
          FROM affinity a
          JOIN nodes n ON n.id = a.node_id
         WHERE a.signal > 0
           AND n.tier = 'working'
         GROUP BY a.node_id
        HAVING session_count >= ?
        """,
        (min_sessions,),
    ).fetchall()

    if not rows:
        return

    promoted = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for row in rows:
        node_id = row["node_id"]
        session_count = row["session_count"]

        node = engine.file_store.load(node_id)
        if node is None:
            continue

        # Double-check tier — file_store is authoritative.
        if node.tier != Tier.working:
            continue

        node.tier = Tier.core
        node.updated = datetime.now(timezone.utc)
        path = engine.file_store.save(node)
        engine.builder.index_single(path)

        with engine.db.transaction() as conn:
            conn.execute(
                "UPDATE nodes SET tier = 'core', updated = ? WHERE id = ?",
                (now_iso, node_id),
            )

        logger.info(
            "Auto-promoted node %s to core (positive hits in %d sessions)",
            node_id[:8],
            session_count,
        )
        promoted += 1

    if promoted:
        # Enforce core cap after bulk promotion.
        core_rows = engine.db.conn.execute(
            "SELECT id, importance FROM nodes WHERE tier = 'core'"
        ).fetchall()
        core_nodes = [
            engine.file_store.load(r["id"])
            for r in core_rows
            if engine.file_store.load(r["id"]) is not None
        ]
        core_nodes = [n for n in core_nodes if n is not None]
        demoted = engine.tier_manager.enforce_core_cap(core_nodes)
        for d in demoted:
            engine.file_store.save(d)
            with engine.db.transaction() as conn:
                conn.execute(
                    "UPDATE nodes SET tier = 'working', updated = ? WHERE id = ?",
                    (now_iso, d.id),
                )

        logger.info(
            "Recall promoter: promoted %d node(s), demoted %d via cap",
            promoted,
            len(demoted),
        )
