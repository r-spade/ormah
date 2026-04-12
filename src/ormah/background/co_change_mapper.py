"""Co-change mapping: create related_to edges between memory nodes that co-occur in git commits.

Nodes are associated with files via the ``file_path`` tag (e.g. tag = "file:src/foo.py").
For each pair of files that changed together in >= N commits, the mapper looks up matching
memory nodes and writes a ``related_to`` edge between them.
"""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS co_change_checked (
    file_a      TEXT NOT NULL,
    file_b      TEXT NOT NULL,
    commits     INTEGER NOT NULL DEFAULT 0,
    checked_at  TEXT NOT NULL,
    PRIMARY KEY (file_a, file_b)
);
"""

_TAG_PREFIX = "file:"


def _ensure_table(engine) -> None:
    with engine.db.transaction() as conn:
        conn.execute(_TABLE_DDL)


def _git_root(repo_path: Path) -> Path | None:
    """Return the git root for a given path, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def _get_co_change_counts(repo_root: Path, lookback_commits: int) -> dict[tuple[str, str], int]:
    """Parse git log to count how many commits contain each file pair."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                f"--max-count={lookback_commits}",
                "--name-only",
                "--pretty=format:COMMIT",
                "--diff-filter=AM",  # only Added/Modified
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        logger.warning("git log failed: %s", e)
        return {}

    if result.returncode != 0:
        return {}

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    current_files: list[str] = []

    for line in result.stdout.splitlines():
        line = line.strip()
        if line == "COMMIT":
            # Tally pairs from previous commit
            for i, fa in enumerate(current_files):
                for fb in current_files[i + 1 :]:
                    pair = (min(fa, fb), max(fa, fb))
                    pair_counts[pair] += 1
            current_files = []
        elif line:
            current_files.append(line)

    # Don't forget the last commit
    for i, fa in enumerate(current_files):
        for fb in current_files[i + 1 :]:
            pair = (min(fa, fb), max(fa, fb))
            pair_counts[pair] += 1

    return pair_counts


def _nodes_for_file(engine, file_path: str) -> list[str]:
    """Return node IDs tagged with file:<file_path>."""
    tag = f"{_TAG_PREFIX}{file_path}"
    rows = engine.db.conn.execute(
        """
        SELECT n.id FROM nodes n
        JOIN node_tags t ON t.node_id = n.id
        WHERE t.tag = ?
        """,
        (tag,),
    ).fetchall()
    return [r[0] for r in rows]


def _already_linked(engine, node_a: str, node_b: str) -> bool:
    """Check if an edge already exists between two nodes (either direction)."""
    row = engine.db.conn.execute(
        """
        SELECT 1 FROM edges
        WHERE (source_id = ? AND target_id = ?)
           OR (source_id = ? AND target_id = ?)
        LIMIT 1
        """,
        (node_a, node_b, node_b, node_a),
    ).fetchone()
    return row is not None


def _write_edge(engine, node_a: str, node_b: str, commits: int) -> None:
    """Write a related_to edge and update the source node's markdown."""
    from ormah.models.node import Connection, EdgeType

    now = datetime.now(timezone.utc).isoformat()
    reason = f"co-changed in {commits} git commit(s)"

    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO edges "
            "(source_id, target_id, edge_type, weight, created, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (node_a, node_b, "related_to", round(min(commits / 10, 1.0), 3), now, reason),
        )

    try:
        mem_node = engine.file_store.load(node_a)
        if mem_node is not None:
            mem_node.connections.append(
                Connection(target=node_b, edge=EdgeType.related_to, weight=round(min(commits / 10, 1.0), 2))
            )
            engine.file_store.save(mem_node)
    except Exception as e:
        logger.debug("Failed to persist co-change edge to markdown for %s: %s", node_a[:8], e)


def run_co_change_mapping(engine) -> None:
    """Main entry point for the co-change mapper background job."""
    settings = engine.settings
    if not getattr(settings, "co_change_mapping_enabled", True):
        logger.debug("Co-change mapper skipped: disabled")
        return

    min_commits: int = getattr(settings, "co_change_min_commits", 2)
    lookback: int = getattr(settings, "co_change_lookback_commits", 500)

    _ensure_table(engine)

    # Discover a git repo to scan. Try memory_dir parents, then cwd.
    repo_root: Path | None = None
    for candidate in [settings.memory_dir, Path.cwd()]:
        repo_root = _git_root(candidate)
        if repo_root:
            break

    if repo_root is None:
        logger.debug("Co-change mapper: no git repo found — skipping")
        return

    logger.info("Co-change mapper: scanning git history in %s (lookback=%d)", repo_root, lookback)

    pair_counts = _get_co_change_counts(repo_root, lookback)
    if not pair_counts:
        logger.debug("Co-change mapper: no file pairs found")
        return

    edges_created = 0
    now = datetime.now(timezone.utc).isoformat()

    for (file_a, file_b), count in pair_counts.items():
        if count < min_commits:
            continue

        # Record this pair as seen (upsert)
        with engine.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO co_change_checked (file_a, file_b, commits, checked_at) "
                "VALUES (?, ?, ?, ?)",
                (file_a, file_b, count, now),
            )

        nodes_a = _nodes_for_file(engine, file_a)
        nodes_b = _nodes_for_file(engine, file_b)

        if not nodes_a or not nodes_b:
            continue

        for na in nodes_a:
            for nb in nodes_b:
                if na == nb:
                    continue
                if _already_linked(engine, na, nb):
                    continue
                _write_edge(engine, na, nb, count)
                edges_created += 1

    logger.info("Co-change mapper: created %d new edges", edges_created)
