"""Background job: recompute importance scores for all memory nodes."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from ormah.background.memory_lock import serialized_memory_job
from ormah.engine.lifecycle import importance_recency

logger = logging.getLogger(__name__)


def _commit_updates_chunked(db, updates, chunk_size: int = 100) -> None:
    """Apply (importance, node_id) updates in bounded write transactions so a
    full-store batch never holds the write lock long enough to stall foreground writes."""
    for i in range(0, len(updates), chunk_size):
        with db.transaction() as conn:
            for importance_val, nid in updates[i : i + chunk_size]:
                conn.execute(
                    "UPDATE nodes SET importance = ? WHERE id = ?",
                    (importance_val, nid),
                )


@serialized_memory_job
def run_importance_scoring(engine) -> None:
    """Iterate all nodes, compute weighted importance, persist changes."""
    settings = engine.settings
    conn = engine.db.conn

    # Fetch everything upfront — avoids N+1 queries for importance lookups
    rows = conn.execute(
        "SELECT id, access_count, last_accessed, importance, created FROM nodes"
    ).fetchall()
    if not rows:
        return

    # Batch edge counts — one query instead of N per-node queries
    edge_counts: dict[str, int] = {}
    edge_rows = conn.execute(
        "SELECT node_id, SUM(cnt) as edge_count FROM ("
        "  SELECT source_id as node_id, COUNT(*) as cnt FROM edges GROUP BY source_id"
        "  UNION ALL"
        "  SELECT target_id as node_id, COUNT(*) as cnt FROM edges GROUP BY target_id"
        ") GROUP BY node_id"
    ).fetchall()
    for er in edge_rows:
        edge_counts[er["node_id"]] = er["edge_count"]

    now = datetime.now(timezone.utc)

    w_access = settings.importance_access_weight
    w_edge = settings.importance_edge_weight
    w_recency = settings.importance_recency_weight

    # Absolute normalization references
    ref_access = settings.importance_access_reference
    ref_edge = settings.importance_edge_reference

    # Weight normalization — ensures custom configs that don't sum to 1.0 still work
    total_weight = w_access + w_edge + w_recency

    updated = 0
    updates: list[tuple[float, str]] = []
    for r in rows:
        nid = r["id"]
        access_count = r["access_count"] or 0

        # Access signal — absolute normalization
        access_signal = min(1.0, math.log1p(access_count) / math.log1p(ref_access))

        # Centrality signal — absolute normalization
        ec = edge_counts.get(nid, 0)
        edge_signal = min(1.0, math.log1p(ec) / math.log1p(ref_edge))

        # Importance recency is intentionally independent from lifecycle
        # stability and uses confirmed-use/creation recency.
        try:
            anchor_str = r["last_accessed"] or r["created"]
            anchor = datetime.fromisoformat(anchor_str)
            days_ago = max((now - anchor).total_seconds() / 86400, 0)
            recency_signal = importance_recency(
                days_ago,
                settings.importance_recency_half_life_days,
            )
        except (ValueError, TypeError):
            recency_signal = 0.0

        importance = (
            w_access * access_signal
            + w_edge * edge_signal
            + w_recency * recency_signal
        )

        # Normalize by total weight
        if total_weight > 0:
            importance = importance / total_weight

        importance = max(0.0, min(1.0, importance))

        # Only update if delta is meaningful
        old_importance = r["importance"] if r["importance"] is not None else 0.5

        if abs(importance - old_importance) < 0.01:
            continue

        updates.append((round(importance, 4), nid))

        # Update markdown file
        node = engine.file_store.load(nid)
        if node is not None:
            node.importance = round(importance, 4)
            node.touch_updated()
            engine.file_store.save(node)

        updated += 1

    if updates:
        _commit_updates_chunked(engine.db, updates)
    if updated:
        logger.info("Importance scorer: updated %d/%d nodes", updated, len(rows))
