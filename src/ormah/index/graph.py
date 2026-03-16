"""Graph traversal queries using recursive CTEs."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


_NOT_EXPIRED = "(n.valid_until IS NULL OR n.valid_until > strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'))"


class GraphIndex:
    """Graph queries on the SQLite index."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None and len(node_id) < 36:
            row = self.conn.execute(
                "SELECT * FROM nodes WHERE id LIKE ? LIMIT 1", (node_id + "%",)
            ).fetchone()
        return dict(row) if row else None

    def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch multiple nodes in a single query, keyed by ID."""
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", node_ids
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def get_tags_batch(self, node_ids: list[str]) -> dict[str, set[str]]:
        """Fetch tags for multiple nodes in a single query, keyed by node ID."""
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self.conn.execute(
            f"SELECT node_id, tag FROM node_tags WHERE node_id IN ({placeholders})",
            node_ids,
        ).fetchall()
        result: dict[str, set[str]] = {}
        for row in rows:
            result.setdefault(row["node_id"], set()).add(row["tag"])
        return result

    def get_neighbors(
        self, node_id: str, depth: int = 1, edge_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get neighbors up to `depth` hops using recursive CTE."""
        if depth == 1:
            return self._get_direct_neighbors(node_id, edge_types)

        query = f"""
        WITH RECURSIVE traverse(id, depth) AS (
            VALUES(?, 0)
            UNION
            SELECT
                CASE WHEN e.source_id = traverse.id THEN e.target_id ELSE e.source_id END,
                traverse.depth + 1
            FROM traverse
            JOIN edges e ON e.source_id = traverse.id OR e.target_id = traverse.id
            WHERE traverse.depth < ?
        )
        SELECT DISTINCT n.* FROM nodes n
        JOIN traverse t ON n.id = t.id
        WHERE n.id != ? AND {_NOT_EXPIRED}
        """
        rows = self.conn.execute(query, (node_id, depth, node_id)).fetchall()
        return [dict(r) for r in rows]

    def get_edges_for(self, node_id: str) -> list[dict[str, Any]]:
        """Get all edges connected to a node."""
        query = """
        SELECT * FROM edges
        WHERE source_id = ? OR target_id = ?
        """
        rows = self.conn.execute(query, (node_id, node_id)).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_by_tier(self, tier: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            f"SELECT * FROM nodes n WHERE tier = ? AND {_NOT_EXPIRED}", (tier,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_nodes(
        self,
        limit: int = 10,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query nodes by created timestamp, ordered by created DESC.

        Returns up to *limit* non-expired nodes whose ``created`` column
        falls within the optional *after*/*before* window.
        """
        clauses = [_NOT_EXPIRED]
        params: list[Any] = []
        if after:
            clauses.append("n.created >= ?")
            params.append(after)
        if before:
            clauses.append("n.created <= ?")
            params.append(before)
        where = " AND ".join(clauses)
        query = f"SELECT * FROM nodes n WHERE {where} ORDER BY n.created DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM nodes WHERE type = ?", (node_type,)).fetchall()
        return [dict(r) for r in rows]

    def get_nodes_by_space(self, space: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM nodes WHERE space = ?", (space,)).fetchall()
        return [dict(r) for r in rows]

    def count_by_tier(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT tier, COUNT(*) as cnt FROM nodes GROUP BY tier"
        ).fetchall()
        return {row["tier"]: row["cnt"] for row in rows}

    def fts_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Full-text search using FTS5.

        Uses AND semantics for multi-token queries (all terms must match),
        falling back to OR if AND yields no results.
        """
        fts_queries = _sanitize_fts_query(query)
        if not fts_queries:
            return []
        try:
            # Try AND query first (stricter), fall back to OR (broader)
            for fts_query in fts_queries:
                rows = self.conn.execute(
                    """
                    SELECT f.id, bm25(nodes_fts, 10.0, 1.0, 5.0, 3.0) as rank
                    FROM nodes_fts f
                    WHERE nodes_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
                if rows:
                    return [{"id": row["id"], "score": -row["rank"]} for row in rows]
            return []
        except Exception as e:
            logger.warning("FTS search failed for query %r: %s", query, e)
            return []

    def get_all_edges(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM edges").fetchall()
        return [dict(r) for r in rows]

    def _get_direct_neighbors(
        self, node_id: str, edge_types: list[str] | None
    ) -> list[dict[str, Any]]:
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            query = f"""
            SELECT DISTINCT n.* FROM nodes n
            JOIN edges e ON (e.target_id = n.id AND e.source_id = ?)
                        OR (e.source_id = n.id AND e.target_id = ?)
            WHERE e.edge_type IN ({placeholders})
              AND {_NOT_EXPIRED}
            """
            params = [node_id, node_id] + edge_types
        else:
            query = f"""
            SELECT DISTINCT n.* FROM nodes n
            JOIN edges e ON (e.target_id = n.id AND e.source_id = ?)
                        OR (e.source_id = n.id AND e.target_id = ?)
            WHERE {_NOT_EXPIRED}
            """
            params = [node_id, node_id]
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "user", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those", "am", "not", "no",
    "nor", "so", "if", "or", "and", "but", "for", "of", "to", "in",
    "on", "at", "by", "with", "from", "as", "into", "about", "what",
    "which", "who", "whom", "when", "where", "why", "how", "all", "any",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "than", "too", "very", "just", "because", "also",
})

_IDENTITY_TOKENS = frozenset({"user", "i", "me", "my", "we", "our", "you", "your"})


def _sanitize_fts_query(query: str) -> list[str]:
    """Convert natural language query to FTS5-compatible queries.

    Returns a list of queries to try in order: AND first (stricter),
    then OR (broader fallback). For single-token queries, both are identical.

    When identity-related tokens (user, my, I, etc.) are detected in the
    raw query, ``about_self`` is injected as a search token so that FTS
    naturally prefers identity nodes (which carry ``about_self`` in their
    tags column).
    """
    import re
    # Remove special FTS5 characters
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    raw_tokens = cleaned.lower().split()

    has_identity_token = any(t in _IDENTITY_TOKENS for t in raw_tokens)
    tokens = [t for t in raw_tokens if t not in _STOP_WORDS and len(t) > 1]

    if not tokens:
        # Fall back to all words if everything was filtered
        tokens = [t for t in cleaned.split() if len(t) > 1]

    # Inject AFTER both paths so identity boost applies even when
    # all content tokens were stop words (e.g. "where do I live").
    if has_identity_token and tokens:
        tokens.append("about_self")
    if not tokens:
        return []
    if len(tokens) == 1:
        return [tokens[0]]
    # AND first (all terms must match), OR as fallback (any term matches)
    return [" AND ".join(tokens), " OR ".join(tokens)]
