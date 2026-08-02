"""Automatic space/cluster assignment for unassigned nodes."""

from __future__ import annotations

import logging
from collections import Counter

from ormah.models.node import normalize_space

logger = logging.getLogger(__name__)


def run_auto_cluster(engine) -> None:
    """Assign unassigned nodes to spaces based on their connections."""
    try:
        # Skip user-curated globals (space_locked) and the self node — a None space on
        # those means "deliberately global", not "needs a space" (#22 council follow-up).
        unassigned = engine.db.conn.execute(
            "SELECT id FROM nodes "
            "WHERE (space IS NULL OR space = '') AND space_locked = 0 AND id != ?",
            (engine.user_node_id or "",),
        ).fetchall()

        if not unassigned:
            return

        assigned = 0
        updates: list[tuple[str, str]] = []  # (space, node_id)
        for row in unassigned:
            node_id = row["id"]
            # Look at connected nodes' spaces
            neighbors = engine.db.conn.execute(
                """
                SELECT n.space FROM nodes n
                JOIN edges e ON (e.target_id = n.id AND e.source_id = ?)
                            OR (e.source_id = n.id AND e.target_id = ?)
                WHERE n.space IS NOT NULL AND n.space != ''
                """,
                (node_id, node_id),
            ).fetchall()

            if not neighbors:
                continue

            # Majority vote. Normalize at the source so both writes below (the raw
            # index UPDATE and the markdown node.space assignment) stay clean — a stale
            # neighbor with the literal 'null' string must not propagate a phantom space.
            spaces = [n["space"] for n in neighbors]
            most_common = normalize_space(Counter(spaces).most_common(1)[0][0])
            if most_common is None:
                continue

            # Re-check the source of truth (markdown) before writing: the index row may be
            # stale, or the node may have been locked between selection and now. Never
            # reassign a locked node or the self node.
            node = engine.file_store.load(node_id)
            if node is None or node_id == engine.user_node_id:
                continue
            if node.space_locked:
                # The file (source of truth) says locked but the index selected it — heal the
                # stale index row so this node stops resurfacing in the query every run.
                with engine.db.transaction() as conn:
                    conn.execute(
                        "UPDATE nodes SET space = ?, space_locked = 1 WHERE id = ?",
                        (node.space, node_id),
                    )
                continue
            node.space = most_common
            node.touch_updated()   # real edit -> advance `updated` for LWW sync ordering
            # ponytail: a concurrent lock landing between this recheck and save() loses to
            # last-writer here (the index UPDATE below is still guarded). Bounded: hourly job,
            # microsecond window, single-user, self-heals next run. A cross-store file<->SQLite
            # lock would close it but is overkill for this context.
            engine.file_store.save(node)
            updates.append((most_common, node_id))
            assigned += 1

        if updates:
            chunk_size = 100
            for i in range(0, len(updates), chunk_size):
                with engine.db.transaction() as conn:
                    for space_val, node_id in updates[i : i + chunk_size]:
                        # Guard the index write too: a concurrent lock after the recheck
                        # above must not be clobbered.
                        conn.execute(
                            "UPDATE nodes SET space = ? WHERE id = ? AND space_locked = 0",
                            (space_val, node_id),
                        )
        if assigned:
            logger.info("Auto-cluster assigned %d nodes to spaces", assigned)

    except Exception as e:
        logger.warning("Auto-cluster failed: %s", e)
