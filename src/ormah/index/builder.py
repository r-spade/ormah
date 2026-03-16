"""Index builder: full rebuild and incremental updates from markdown files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ormah.index.db import Database
from ormah.store.file_store import FileStore
from ormah.store.markdown import parse_node

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds and updates the SQLite index from markdown source files."""

    def __init__(self, db: Database, file_store: FileStore) -> None:
        self.db = db
        self.file_store = file_store

    def full_rebuild(self) -> int:
        """Drop and rebuild the entire index from markdown files. Returns node count."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM node_tags")
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes_fts")
            conn.execute("DELETE FROM nodes")
            try:
                conn.execute("DELETE FROM node_vectors")
            except Exception:
                pass  # table may not exist

        # Two-pass: nodes first, then edges (to satisfy FK constraints)
        paths = list(self.file_store.list_paths())
        count = 0
        with self.db.transaction():
            for path in paths:
                try:
                    self._index_file_nodes_only(path)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to index %s: %s", path, e)

            for path in paths:
                try:
                    self._index_file_edges(path)
                except Exception as e:
                    logger.warning("Failed to index edges for %s: %s", path, e)

        return count

    def incremental_update(self) -> tuple[int, int]:
        """Update index for changed/new files. Returns (added, updated) counts."""
        conn = self.db.conn
        added = 0
        updated = 0

        indexed: dict[str, str] = {}
        for row in conn.execute("SELECT id, file_hash FROM nodes").fetchall():
            indexed[row["id"]] = row["file_hash"]

        indexed_ids = set(indexed.keys())
        disk_ids: set[str] = set()

        with self.db.transaction():
            for path in self.file_store.list_paths():
                try:
                    file_hash = self.file_store.file_hash(path)
                    node = parse_node(path.read_text(encoding="utf-8"))
                    disk_ids.add(node.id)

                    if node.id not in indexed:
                        self._index_file(path)
                        added += 1
                    elif indexed[node.id] != file_hash:
                        self._remove_node(node.id, keep_vectors=True)
                        self._index_file(path)
                        updated += 1
                except Exception as e:
                    logger.warning("Failed to process %s: %s", path, e)

            # Remove nodes whose files were deleted
            removed_ids = indexed_ids - disk_ids
            for node_id in removed_ids:
                self._remove_node(node_id)

        return added, updated

    def index_single(self, path: Path) -> None:
        """Index or re-index a single file."""
        node = parse_node(path.read_text(encoding="utf-8"))
        with self.db.transaction():
            self._remove_node(node.id)
            self._index_file(path)

    def _index_file(self, path: Path) -> None:
        """Index a single markdown file into the database (nodes + edges)."""
        self._index_file_nodes_only(path)
        self._index_file_edges(path)

    def _index_file_nodes_only(self, path: Path) -> None:
        """Index node, tags, and FTS from a markdown file (no edges)."""
        text = path.read_text(encoding="utf-8")
        node = parse_node(text)
        file_hash = self.file_store.file_hash(path)
        conn = self.db.conn

        conn.execute(
            """
            INSERT OR REPLACE INTO nodes
            (id, type, tier, source, space, title, content, created, updated,
             last_accessed, access_count, confidence, importance,
             valid_until, stability, last_review, file_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.type.value,
                node.tier.value,
                node.source,
                node.space,
                node.title,
                node.content,
                node.created.isoformat(),
                node.updated.isoformat(),
                node.last_accessed.isoformat(),
                node.access_count,
                node.confidence,
                node.importance,
                node.valid_until.isoformat() if node.valid_until else None,
                node.stability,
                node.last_review.isoformat() if node.last_review else None,
                str(path),
                file_hash,
            ),
        )

        # Tags
        for tag in node.tags:
            conn.execute(
                "INSERT OR IGNORE INTO node_tags (node_id, tag) VALUES (?, ?)",
                (node.id, tag),
            )

        # FTS
        tags_str = " ".join(node.tags)
        conn.execute(
            "INSERT INTO nodes_fts (id, title, content, tags) VALUES (?, ?, ?, ?)",
            (node.id, node.title or "", node.content, tags_str),
        )

    def _index_file_edges(self, path: Path) -> None:
        """Index edges from a markdown file's connections."""
        text = path.read_text(encoding="utf-8")
        node = parse_node(text)
        conn = self.db.conn

        for c in node.connections:
            # Only insert if target node exists (avoids FK violation)
            target_exists = conn.execute(
                "SELECT 1 FROM nodes WHERE id = ?", (c.target,)
            ).fetchone()
            if not target_exists:
                continue

            # Skip if reverse edge already exists (avoid bidirectional duplicates)
            reverse_exists = conn.execute(
                "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                (c.target, node.id, c.edge.value),
            ).fetchone()
            if reverse_exists:
                continue

            conn.execute(
                """
                INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight, created)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node.id, c.target, c.edge.value, c.weight, node.created.isoformat()),
            )

    def _remove_node(self, node_id: str, *, keep_vectors: bool = False) -> None:
        """Remove a node and its related data from the index.

        Args:
            keep_vectors: If True, preserve the node_vectors row. Used by
                incremental_update where the markdown file changed but the
                embedding content hasn't — deleting the vector would cause
                permanent embedding loss since the index updater doesn't
                re-embed.
        """
        conn = self.db.conn
        conn.execute("DELETE FROM node_tags WHERE node_id = ?", (node_id,))
        conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id)
        )
        conn.execute("DELETE FROM nodes_fts WHERE id = ?", (node_id,))
        conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        # Vector cleanup if table exists
        if not keep_vectors:
            try:
                conn.execute("DELETE FROM node_vectors WHERE id = ?", (node_id,))
            except Exception:
                pass
