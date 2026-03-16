"""Automatic space/cluster assignment for unassigned nodes."""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def run_auto_cluster(engine) -> None:
    """Assign unassigned nodes to spaces based on their connections."""
    try:
        unassigned = engine.db.conn.execute(
            "SELECT id FROM nodes WHERE space IS NULL OR space = ''"
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

            # Majority vote
            spaces = [n["space"] for n in neighbors]
            most_common = Counter(spaces).most_common(1)[0][0]

            updates.append((most_common, node_id))

            # Update markdown file
            node = engine.file_store.load(node_id)
            if node:
                node.space = most_common
                engine.file_store.save(node)

            assigned += 1

        if updates:
            with engine.db.transaction() as conn:
                for space_val, node_id in updates:
                    conn.execute(
                        "UPDATE nodes SET space = ? WHERE id = ?", (space_val, node_id)
                    )
        if assigned:
            logger.info("Auto-cluster assigned %d nodes to spaces", assigned)

    except Exception as e:
        logger.warning("Auto-cluster failed: %s", e)
