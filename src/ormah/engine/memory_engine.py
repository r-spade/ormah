"""Central facade for all memory operations."""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ormah.config import Settings
from ormah.engine.context_builder import ContextBuilder
from ormah.engine.tier_manager import TierManager
from ormah.engine.traversal import (
    format_node_with_neighbors,
    format_search_results,
)
from ormah.index.builder import IndexBuilder
from ormah.index.db import Database
from ormah.index.graph import GraphIndex
from ormah.models.node import (
    ConnectRequest,
    Connection,
    CreateNodeRequest,
    EdgeType,
    MemoryNode,
    NodeType,
    Tier,
    UpdateNodeRequest,
)
from ormah.store.file_store import FileStore

logger = logging.getLogger(__name__)

# Edge type factors for spreading activation scoring.
# Higher factor = tighter structural link = more activation propagated.
_EMBEDDING_SCHEMA_VERSION = 2


def _generate_title(content: str, max_chars: int = 60) -> str:
    """Generate a short title from the first line/sentence of content."""
    # Take the first line
    first_line = content.strip().split("\n", 1)[0].strip()
    if len(first_line) <= max_chars:
        return first_line
    # Truncate at last word boundary within max_chars
    truncated = first_line[:max_chars].rsplit(" ", 1)[0]
    return truncated + "…" if truncated else first_line[:max_chars]


def _embedding_text(title: str | None, content: str, max_content_chars: int = 512) -> str:
    """Build text for embedding. Truncates content to avoid topic averaging in long docs."""
    prefix = title or ""
    truncated = content[:max_content_chars]
    return f"{prefix} {truncated}".strip()


# Edge type factors for spreading activation scoring.
# Higher factor = tighter structural link = more activation propagated.
_EDGE_TYPE_FACTORS: dict[str, float] = {
    "supports": 1.0,
    "part_of": 1.0,
    "depends_on": 1.0,
    "defines": 1.0,
    "derived_from": 1.0,
    "evolved_from": 0.8,
    "related_to": 0.7,
    "instance_of": 0.7,
    "caused_by": 0.7,
    "preceded_by": 0.7,
    "contradicts": 0.4,
}


class MemoryEngine:
    """Main facade: remember(), recall(), connect(), context()."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.file_store = FileStore(settings.nodes_dir)
        self.db = Database(settings.db_path)
        self.db.init_schema()
        self.db.init_vec_table(settings.embedding_dim)

        self.graph = GraphIndex(self.db.conn)
        self.builder = IndexBuilder(self.db, self.file_store)
        self.tier_manager = TierManager(settings.core_memory_cap)
        self.context_builder = ContextBuilder(self.graph, engine=self)

        self.user_node_id: str | None = None

        # Lazy-loaded components
        self._hybrid_search = None

    def startup(self) -> None:
        """Initialize on server start: rebuild index if empty, ensure self node."""
        count = self.db.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        if count == 0:
            n = self.builder.full_rebuild()
            logger.info("Initial index rebuild: %d nodes", n)

        # Rebuild FTS index if tokenizer was migrated
        fts_rebuild_row = self.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'fts_needs_rebuild'"
        ).fetchone()
        if fts_rebuild_row and fts_rebuild_row["value"] == "1":
            logger.info("Rebuilding FTS index after tokenizer migration")
            n = self.builder.full_rebuild()
            logger.info("FTS rebuild complete: %d nodes", n)
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_needs_rebuild', '0')"
                )

        # Re-embed nodes if the vector store is missing entries or schema version changed
        vec_count = self.db.conn.execute("SELECT count(*) FROM node_vectors").fetchone()[0]
        stored_version_row = self.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'embedding_schema_version'"
        ).fetchone()
        stored_version = int(stored_version_row["value"]) if stored_version_row else 0
        needs_reindex = (count > 0 and vec_count < count) or stored_version < _EMBEDDING_SCHEMA_VERSION

        if needs_reindex:
            reason = "schema version change" if stored_version < _EMBEDDING_SCHEMA_VERSION else "missing entries"
            logger.info("Re-indexing embeddings (%s): vec=%d, nodes=%d, schema v%d→v%d",
                        reason, vec_count, count, stored_version, _EMBEDDING_SCHEMA_VERSION)
            self._reindex_all_embeddings()
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_schema_version', ?)",
                    (str(_EMBEDDING_SCHEMA_VERSION),),
                )

        # One-time FSRS data migration: seed stability from access patterns
        self._migrate_fsrs()

        self._ensure_self_node()
        self._migrate_identity_tiers()
        self._warmup_embedder()

    def _migrate_fsrs(self) -> None:
        """Seed FSRS stability from access_count on first run, updating both DB and markdown."""
        fsrs_migrated = self.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'fsrs_migrated'"
        ).fetchone()
        if fsrs_migrated:
            return

        rows = self.db.conn.execute(
            "SELECT id, access_count, last_accessed FROM nodes"
        ).fetchall()

        with self.db.transaction() as conn:
            for r in rows:
                access_count = r["access_count"] or 0
                stability = min(30.0, access_count * 2.0) if access_count > 0 else 1.0
                last_review = r["last_accessed"]

                # Update DB
                conn.execute(
                    "UPDATE nodes SET stability = ?, last_review = ? WHERE id = ?",
                    (stability, last_review, r["id"]),
                )

                # Update markdown file
                node = self.file_store.load(r["id"])
                if node is not None:
                    node.stability = stability
                    if last_review:
                        try:
                            node.last_review = datetime.fromisoformat(last_review)
                        except (ValueError, TypeError):
                            pass
                    self.file_store.save(node)

            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('fsrs_migrated', '1')"
            )
        logger.info("FSRS data migration complete: seeded %d nodes from access_count", len(rows))

    def _ensure_self_node(self) -> None:
        """Load or create the user self node."""
        row = self.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'user_node_id'"
        ).fetchone()

        if row:
            node_id = row["value"]
            # Verify node still exists on disk
            if self.file_store.load(node_id) is not None:
                self.user_node_id = node_id
                return
            # Node gone from disk — fall through to create a new one

        # Create self node
        node = MemoryNode(
            type=NodeType.person,
            tier=Tier.core,
            source="system:self",
            space=None,
            tags=["self", "identity"],
            title="Self",
            content="The user's identity and personal information.",
        )

        path = self.file_store.save(node)
        self.builder.index_single(path)
        self._index_embedding(node)

        # Persist in meta table
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('user_node_id', ?)",
                (node.id,),
            )
        self.user_node_id = node.id
        logger.info("Created user self node: %s", node.id[:8])

    def _migrate_identity_tiers(self) -> None:
        """One-time migration: fix identity node tiers and edges.

        Phase 1 (identity_tier_migrated): Demote core identity preferences to working.
        Phase 2 (identity_edges_repaired): Re-link orphaned about_self nodes that lost
        their 'defines' edge (e.g. after consolidation merged the linked original).
        """
        conn = self.db.conn
        if not self.user_node_id:
            return

        # Phase 1: demote core identity preferences to working
        if not conn.execute(
            "SELECT value FROM meta WHERE key = 'identity_tier_migrated'"
        ).fetchone():
            with self.db.transaction() as conn:
                identity_ids = [
                    r["target_id"]
                    for r in conn.execute(
                        "SELECT target_id FROM edges WHERE source_id = ? AND edge_type = 'defines'",
                        (self.user_node_id,),
                    ).fetchall()
                ]
                if identity_ids:
                    placeholders = ",".join("?" * len(identity_ids))
                    demoted = conn.execute(
                        f"UPDATE nodes SET tier = 'working' WHERE id IN ({placeholders}) "
                        f"AND tier = 'core' AND type NOT IN ('person')",
                        identity_ids,
                    ).rowcount
                    if demoted:
                        for nid in identity_ids:
                            node = self.file_store.load(nid)
                            if node and node.tier == Tier.core and node.type != NodeType.person:
                                node.tier = Tier.working
                                self.file_store.save(node)
                        logger.info("Migrated %d identity nodes from core to working tier", demoted)

                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('identity_tier_migrated', '1')"
                )

        # Phase 2: re-link orphaned about_self nodes missing their defines edge
        if not self.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'identity_edges_repaired'"
        ).fetchone():
            with self.db.transaction() as conn:
                # Find non-archival nodes with about_self tag but no defines edge
                orphaned = conn.execute(
                    """
                    SELECT n.id FROM nodes n
                    JOIN node_tags t ON n.id = t.node_id AND t.tag = 'about_self'
                    WHERE n.tier != 'archival'
                      AND n.id != ?
                      AND n.id NOT IN (
                          SELECT target_id FROM edges
                          WHERE source_id = ? AND edge_type = 'defines'
                      )
                    """,
                    (self.user_node_id, self.user_node_id),
                ).fetchall()

                now = datetime.now(timezone.utc).isoformat()
                linked = 0
                for row in orphaned:
                    conn.execute(
                        "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, created) "
                        "VALUES (?, ?, 'defines', 1.0, ?)",
                        (self.user_node_id, row["id"], now),
                    )
                    linked += 1

                # Also demote any remaining core preferences (missed by phase 1
                # because they had no defines edge at that time)
                demoted = conn.execute(
                    """
                    UPDATE nodes SET tier = 'working'
                    WHERE id IN (
                        SELECT n.id FROM nodes n
                        JOIN node_tags t ON n.id = t.node_id AND t.tag = 'about_self'
                        WHERE n.tier = 'core' AND n.type NOT IN ('person')
                        AND n.id != ?
                    )
                    """,
                    (self.user_node_id,),
                ).rowcount
                if demoted:
                    # Update file store copies
                    for row in conn.execute(
                        """
                        SELECT n.id FROM nodes n
                        JOIN node_tags t ON n.id = t.node_id AND t.tag = 'about_self'
                        WHERE n.tier = 'working' AND n.type NOT IN ('person')
                        AND n.id != ?
                        """,
                        (self.user_node_id,),
                    ).fetchall():
                        node = self.file_store.load(row["id"])
                        if node and node.tier == Tier.core:
                            node.tier = Tier.working
                            self.file_store.save(node)

                if linked:
                    logger.info("Re-linked %d orphaned identity nodes", linked)
                if demoted:
                    logger.info("Demoted %d remaining core preferences to working", demoted)

                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('identity_edges_repaired', '1')"
                )

    def _warmup_embedder(self) -> None:
        """Load the embedding model now so the first request doesn't stall.

        On a fresh install the local model may need to be downloaded (~420 MB).
        Running this during startup ensures the health endpoint only becomes
        reachable after the model is ready.
        """
        try:
            from ormah.embeddings.encoder import get_encoder

            get_encoder(self.settings).encode("")
        except Exception as e:
            logger.warning("Embedding model warmup failed: %s", e)

    def shutdown(self) -> None:
        self.db.close()

    # --- Core operations ---

    def remember(self, req: CreateNodeRequest, agent_id: str | None = None) -> tuple[str, str]:
        """Store a new memory. Returns (node_id, formatted_text)."""
        title = req.title
        if not title and req.content.strip():
            title = _generate_title(req.content)

        node = MemoryNode(
            type=req.type,
            tier=req.tier,
            source=req.source or f"agent:{agent_id or 'unknown'}",
            space=req.space,
            tags=req.tags,
            connections=req.connections,
            title=title,
            content=req.content,
            confidence=req.confidence,
        )

        # Mark and promote identity nodes
        if req.about_self:
            if "about_self" not in node.tags:
                node.tags.append("about_self")
            if node.type == NodeType.person:
                node.tier = Tier.core

        # Enforce core cap
        if node.tier == Tier.core:
            protected = {self.user_node_id} if self.user_node_id else set()
            core_nodes = [
                self.file_store.load(r["id"])
                for r in self.graph.get_nodes_by_tier("core")
            ]
            core_nodes = [n for n in core_nodes if n is not None]
            core_nodes.append(node)
            demoted = self.tier_manager.enforce_core_cap(core_nodes, protected_ids=protected)
            for d in demoted:
                self.file_store.save(d)

        path = self.file_store.save(node)
        self.builder.index_single(path)

        # Index embedding and auto-link to similar nodes
        self._index_embedding(node)
        auto_links = self._auto_link_node(node)

        # Link to self node if about_self
        if req.about_self and self.user_node_id:
            self._link_to_self(node)

        formatted = f"Remembered [{node.type.value}]: {node.title or node.content[:80]}\nID: {node.id}"
        if auto_links:
            formatted += f"\nAuto-linked to {len(auto_links)} related memories:"
            for link_id, link_title, sim in auto_links:
                formatted += f"\n  → {link_title} ({sim:.0%} similar)"
        return node.id, formatted

    def recall_node(self, node_id: str) -> str | None:
        """Get a specific node with its neighbors, formatted as text."""
        node = self.graph.get_node(node_id)
        if node is None:
            return None

        # Touch access
        self._touch_access(node_id)

        edges = self.graph.get_edges_for(node_id)
        neighbors = self.graph.get_neighbors(node_id, depth=1)
        return format_node_with_neighbors(node, edges, neighbors)

    # Stop words for detecting "pure temporal" queries (no topical signal).
    _STOP_WORDS = frozenset({
        "what", "did", "we", "do", "i", "you", "the", "a", "is", "are",
        "was", "were", "have", "has", "had", "been", "be", "will", "would",
        "could", "should", "can", "may", "might", "shall", "on", "in", "at",
        "to", "for", "of", "with", "by", "from", "up", "about", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such",
        "no", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "as", "until", "while", "and", "but", "or",
        "nor", "if", "that", "which", "who", "whom", "this", "these",
        "those", "am", "an", "any", "work", "worked", "working",
        "me", "my", "our", "us", "show", "tell", "give", "get",
    })

    def recall_search_structured(
        self, query: str, limit: int = 10, default_space: str | None = None,
        touch_access: bool = True, **filters,
    ) -> list[dict]:
        """Search memories and return structured results (list of dicts).

        Same logic as recall_search but returns raw dicts instead of formatted text.
        Used by the UI and any consumer that needs structured data.

        When *touch_access* is False, access_count and last_accessed are not
        updated — useful for context loading that shouldn't inflate access stats.
        """
        # Auto-extract temporal filters from query when none provided
        if not filters.get("created_after") and not filters.get("created_before"):
            from ormah.engine.prompt_classifier import (
                extract_time_params, has_temporal_phrases, strip_temporal_phrases,
            )
            if has_temporal_phrases(query):
                time_params = extract_time_params(query)
                filters.update(time_params)
                query = strip_temporal_phrases(query)

        explicit_spaces = filters.get("spaces")

        search = self._get_hybrid_search()
        if search is not None:
            results = search.search(query, limit=limit, **filters)
            if default_space and not explicit_spaces:
                results = self._apply_space_scores(results, default_space)

            results = results[:limit]

            # Detect pure temporal queries (no topical signal after stripping)
            query_words = {w.lower() for w in re.findall(r'\w+', query)} - self._STOP_WORDS
            is_pure_temporal = len(query_words) == 0

            # Temporal supplement: for pure temporal queries, discard semantic
            # results entirely and use SQL recency.  For topical temporal queries,
            # supplement if under limit (existing behavior).
            created_after = filters.get("created_after")
            created_before = filters.get("created_before")
            if created_after:
                if is_pure_temporal or len(results) < limit:
                    results = self._supplement_temporal(
                        results if not is_pure_temporal else [],
                        limit, created_after, created_before, filters,
                    )

            results = self._spread_activation(results, limit)
            if touch_access:
                for r in results:
                    if r.get("source") not in ("activated", "conflict"):
                        self._touch_access(r["node"]["id"])
            return results

        # Fallback to FTS only
        fts_results = self.graph.fts_search(query, limit=limit)
        created_after = filters.get("created_after")
        created_before = filters.get("created_before")
        enriched = []
        for r in fts_results:
            node = self.graph.get_node(r["id"])
            if node:
                if created_after and (node.get("created") or "") < created_after:
                    continue
                if created_before and (node.get("created") or "") > created_before:
                    continue
                enriched.append({"node": node, "score": r["score"], "source": "fts"})

        if default_space and not explicit_spaces:
            enriched = self._apply_space_scores(enriched, default_space)

        enriched = enriched[:limit]

        # Temporal fallback for FTS-only path
        if created_after and len(enriched) < limit:
            enriched = self._supplement_temporal(
                enriched, limit, created_after, created_before, filters,
            )

        enriched = self._spread_activation(enriched, limit)
        if touch_access:
            for r in enriched:
                if r.get("source") not in ("activated", "conflict"):
                    self._touch_access(r["node"]["id"])

        return enriched

    def recall_search(self, query: str, limit: int = 10, default_space: str | None = None, **filters) -> str:
        """Search memories and return formatted results.

        If default_space is set and no explicit spaces filter is provided,
        results are score-adjusted: current-project results keep their score,
        global results are scaled by space_boost_global, and other-project
        results by space_boost_other.
        """
        # Auto-extract temporal filters from query when none provided
        if not filters.get("created_after") and not filters.get("created_before"):
            from ormah.engine.prompt_classifier import (
                extract_time_params, has_temporal_phrases, strip_temporal_phrases,
            )
            if has_temporal_phrases(query):
                time_params = extract_time_params(query)
                filters.update(time_params)
                query = strip_temporal_phrases(query)

        explicit_spaces = filters.get("spaces")

        search = self._get_hybrid_search()
        if search is not None:
            results = search.search(query, limit=limit, **filters)
            if default_space and not explicit_spaces:
                results = self._apply_space_scores(results, default_space)

            results = results[:limit]

            # Detect pure temporal queries
            query_words = {w.lower() for w in re.findall(r'\w+', query)} - self._STOP_WORDS
            is_pure_temporal = len(query_words) == 0

            created_after = filters.get("created_after")
            created_before = filters.get("created_before")
            if created_after:
                if is_pure_temporal or len(results) < limit:
                    results = self._supplement_temporal(
                        results if not is_pure_temporal else [],
                        limit, created_after, created_before, filters,
                    )

            results = self._spread_activation(results, limit)
            for r in results:
                if r.get("source") not in ("activated", "conflict"):
                    self._touch_access(r["node"]["id"])
            return format_search_results(results)

        # Fallback to FTS only
        fts_results = self.graph.fts_search(query, limit=limit)
        created_after = filters.get("created_after")
        created_before = filters.get("created_before")
        enriched = []
        for r in fts_results:
            node = self.graph.get_node(r["id"])
            if node:
                if created_after and (node.get("created") or "") < created_after:
                    continue
                if created_before and (node.get("created") or "") > created_before:
                    continue
                enriched.append({"node": node, "score": r["score"], "source": "fts"})

        if default_space and not explicit_spaces:
            enriched = self._apply_space_scores(enriched, default_space)

        enriched = enriched[:limit]

        # Temporal fallback for FTS-only path
        if created_after and len(enriched) < limit:
            enriched = self._supplement_temporal(
                enriched, limit, created_after, created_before, filters,
            )

        enriched = self._spread_activation(enriched, limit)
        for r in enriched:
            if r.get("source") not in ("activated", "conflict"):
                self._touch_access(r["node"]["id"])

        return format_search_results(enriched)

    def update_node(self, node_id: str, req: UpdateNodeRequest) -> str | None:
        """Update a memory node. Returns formatted confirmation or None."""
        node = self.file_store.load(node_id)
        if node is None:
            return None

        # Snapshot old state and track changed fields for audit
        old_snapshot = node.model_dump(mode="json")
        changed_fields = []
        for field in ("content", "type", "tier", "space", "tags", "title", "confidence", "valid_until"):
            if getattr(req, field, None) is not None:
                changed_fields.append(field)

        if req.content is not None:
            node.content = req.content
        if req.type is not None:
            node.type = req.type
        if req.tier is not None:
            node.tier = req.tier
        if req.space is not None:
            node.space = req.space
        if req.tags is not None:
            node.tags = req.tags
        if req.title is not None:
            node.title = req.title
        if req.confidence is not None:
            node.confidence = req.confidence
        if req.valid_until is not None:
            node.valid_until = req.valid_until
        if req.add_connections:
            node.connections.extend(req.add_connections)
            changed_fields.append("connections")

        node.updated = datetime.now(timezone.utc)
        path = self.file_store.save(node)
        self.builder.index_single(path)
        self._index_embedding(node)

        # Invalidate auto-linker checked pairs if content/title changed
        if req.content is not None or req.title is not None:
            with self.db.transaction() as conn:
                conn.execute(
                    "DELETE FROM auto_link_checked WHERE node_a = ? OR node_b = ?",
                    (node_id, node_id),
                )

        # Audit log
        self._write_audit_log(
            operation="update",
            node_id=node_id,
            node_snapshot=json.dumps(old_snapshot),
            detail=json.dumps({"changed_fields": changed_fields}),
        )

        return f"Updated [{node.type.value}]: {node.title or node.content[:80]}\nID: {node.id}"

    def delete_node(self, node_id: str) -> str | None:
        """Delete a memory node from disk and index. Returns confirmation or None."""
        if node_id == self.user_node_id:
            return "Cannot delete the user self node."

        # Load full node from disk for audit snapshot
        full_node = self.file_store.load(node_id)
        if full_node is None:
            # Fall back to graph index to check existence
            node = self.graph.get_node(node_id)
            if node is None:
                return None
            title = node.get("title") or node.get("content", "")[:60]
            snapshot = json.dumps(node)
        else:
            title = full_node.title or full_node.content[:60]
            snapshot = json.dumps(full_node.model_dump(mode="json"))
            node = self.graph.get_node(node_id)

        # Audit log before deletion
        self._write_audit_log(
            operation="delete",
            node_id=node_id,
            node_snapshot=snapshot,
        )

        # Remove from index and clean up auto-linker checked pairs
        with self.db.transaction() as conn:
            self.builder._remove_node(node_id)
            conn.execute(
                "DELETE FROM auto_link_checked WHERE node_a = ? OR node_b = ?",
                (node_id, node_id),
            )

        # Soft-delete from disk (move to deleted/ directory)
        self.file_store.soft_delete(node_id)

        node_type = full_node.type.value if full_node else node.get("type", "unknown") if node else "unknown"
        return f"Deleted [{node_type}]: {title}\nID: {node_id}"

    def connect(self, req: ConnectRequest) -> str:
        """Create an edge between two nodes."""
        # Verify both nodes exist
        if self.graph.get_node(req.source_id) is None:
            return f"Source node {req.source_id} not found."
        if self.graph.get_node(req.target_id) is None:
            return f"Target node {req.target_id} not found."

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight, created)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    req.source_id,
                    req.target_id,
                    req.edge.value,
                    req.weight,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        # Also update the source node's markdown file
        source_node = self.file_store.load(req.source_id)
        if source_node:
            from ormah.models.node import Connection

            source_node.connections.append(
                Connection(target=req.target_id, edge=req.edge, weight=req.weight)
            )
            self.file_store.save(source_node)

        return f"Connected {req.source_id[:8]}... →[{req.edge.value}]→ {req.target_id[:8]}..."

    def get_context(self, space: str | None = None, task_hint: str | None = None) -> str:
        """Get core memories formatted for system prompt."""
        return self.context_builder.build_core_context(
            space=space,
            user_node_id=self.user_node_id,
            task_hint=task_hint,
            max_nodes=self.settings.context_max_nodes,
        )

    def get_whisper_context(
        self,
        prompt: str,
        space: str | None = None,
        recent_prompts: list[str] | None = None,
    ) -> str:
        """Get compact whisper context for involuntary recall injection."""
        return self.context_builder.build_whisper_context(
            prompt=prompt,
            space=space,
            user_node_id=self.user_node_id,
            max_nodes=self.settings.whisper_max_nodes,
            min_score=self.settings.whisper_min_relevance_score,
            identity_max=self.settings.whisper_identity_max_nodes,
            max_content_len=self.settings.whisper_content_max_chars,
            reranker_enabled=self.settings.whisper_reranker_enabled,
            reranker_model=self.settings.whisper_reranker_model,
            reranker_min_score=self.settings.whisper_reranker_min_score,
            reranker_blend_alpha=self.settings.whisper_reranker_blend_alpha,
            reranker_max_doc_chars=self.settings.whisper_reranker_max_doc_chars,
            recent_prompts=recent_prompts,
            injection_gate=self.settings.whisper_injection_gate,
            topic_shift_enabled=self.settings.whisper_topic_shift_enabled,
            topic_shift_threshold=self.settings.whisper_topic_shift_threshold,
            content_total_budget=self.settings.whisper_content_total_budget,
            content_min_per_node=self.settings.whisper_content_min_per_node,
            content_max_per_node=self.settings.whisper_content_max_per_node,
        )

    def mark_outdated(self, node_id: str, reason: str | None = None) -> str | None:
        """Mark a memory as outdated: set valid_until to now, optionally append reason."""
        node = self.file_store.load(node_id)
        if node is None:
            return None

        # Snapshot for audit
        old_valid_until = node.valid_until.isoformat() if node.valid_until else None

        node.valid_until = datetime.now(timezone.utc)
        if reason:
            node.content = node.content.rstrip() + f"\n\n[Outdated: {reason}]"
        node.updated = datetime.now(timezone.utc)

        path = self.file_store.save(node)
        self.builder.index_single(path)

        # Audit log
        self._write_audit_log(
            operation="mark_outdated",
            node_id=node_id,
            detail=json.dumps({
                "reason": reason,
                "old_valid_until": old_valid_until,
            }),
        )

        return (
            f"Marked outdated [{node.type.value}]: {node.title or node.content[:80]}\n"
            f"ID: {node.id} | valid_until: {node.valid_until.isoformat()}"
        )

    def rebuild_index(self) -> int:
        """Full rebuild of the index from markdown files, including embeddings."""
        count = self.builder.full_rebuild()
        self._reindex_all_embeddings()
        return count

    def _reindex_all_embeddings(self) -> None:
        """Re-embed all nodes in the index."""
        try:
            from ormah.embeddings.vector_store import VectorStore
            from ormah.embeddings.encoder import get_encoder

            encoder = get_encoder(self.settings)
            vec_store = VectorStore(self.db)

            nodes = self.db.conn.execute("SELECT id, title, content FROM nodes").fetchall()
            max_chars = self.settings.embedding_max_content_chars

            # Build all embeddings first
            all_items: list[tuple[str, Any]] = []
            failed = 0
            for n in nodes:
                text = _embedding_text(n["title"], n["content"], max_chars)
                if text:
                    try:
                        embedding = encoder.encode(text)
                        all_items.append((n["id"], embedding))
                    except Exception as e:
                        logger.warning("Failed to embed node %s: %s", n["id"][:8], e)
                        failed += 1

            # Upsert in small chunks with WAL checkpoint after each.
            # sqlite-vec vec0 virtual tables can silently drop rows in
            # large transactions; chunking prevents this.
            chunk_size = 100
            for i in range(0, len(all_items), chunk_size):
                chunk = all_items[i : i + chunk_size]
                vec_store.upsert_batch(chunk)
                self.db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            # Verify the count actually matches
            vec_count = self.db.conn.execute("SELECT count(*) FROM node_vectors").fetchone()[0]
            logger.info(
                "Re-embedded %d/%d nodes (vec_count=%d, failed=%d)",
                len(all_items), len(nodes), vec_count, failed,
            )
            if vec_count < len(all_items):
                logger.warning(
                    "Vec table has fewer entries (%d) than embedded (%d) — "
                    "possible sqlite-vec persistence issue",
                    vec_count, len(all_items),
                )
        except Exception as e:
            logger.warning("Failed to reindex embeddings: %s", e)

    def stats(self) -> dict:
        """Get memory store statistics."""
        tier_counts = self.graph.count_by_tier()
        total = sum(tier_counts.values())
        edge_count = self.db.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {
            "total_nodes": total,
            "by_tier": tier_counts,
            "total_edges": edge_count,
        }

    # --- Merge operations ---

    def execute_merge(
        self,
        node_id_a: str,
        node_id_b: str,
        proposal_id: str | None = None,
        merged_content: str | None = None,
        merged_title: str | None = None,
    ) -> str:
        """Merge two nodes, keeping the better one. Returns confirmation text.

        When *merged_content* or *merged_title* are provided (typically from an
        LLM-based merge), the kept node's content/title are overwritten so that
        details from both nodes are preserved.
        """
        node_a = self.file_store.load(node_id_a)
        node_b = self.file_store.load(node_id_b)
        if node_a is None:
            return f"Node {node_id_a} not found."
        if node_b is None:
            return f"Node {node_id_b} not found."

        kept, removed = self._pick_keeper(node_a, node_b)

        # Apply LLM-generated merged content if provided
        if merged_content is not None:
            kept.content = merged_content
        if merged_title is not None:
            kept.title = merged_title

        # Snapshot removed node before deletion
        snapshot = removed.model_dump(mode="json")

        # Capture all edges referencing the removed node BEFORE any deletions
        edge_rows = self.db.conn.execute(
            "SELECT source_id, target_id, edge_type, weight, created FROM edges "
            "WHERE source_id = ? OR target_id = ?",
            (removed.id, removed.id),
        ).fetchall()
        original_edges = [dict(r) for r in edge_rows]

        # Capture incoming edges for the kept node that aren't in its markdown.
        # index_single calls _remove_node which wipes ALL edges (including
        # incoming ones like self→kept "defines").  We need to restore these.
        kept_incoming = self.db.conn.execute(
            "SELECT source_id, target_id, edge_type, weight, created FROM edges "
            "WHERE target_id = ? AND source_id != ?",
            (kept.id, removed.id),
        ).fetchall()
        kept_incoming_edges = [dict(r) for r in kept_incoming]

        # Merge tags from removed into kept
        removed_tags = set(removed.tags) - set(kept.tags)
        if removed_tags:
            kept.tags.extend(sorted(removed_tags))

        # Save kept node, re-index, re-embed
        # NOTE: index_single calls _remove_node internally which wipes edges,
        # so we must remap edges and restore incoming edges AFTER this step.
        kept.updated = datetime.now(timezone.utc)
        path = self.file_store.save(kept)
        self.builder.index_single(path)
        self._index_embedding(kept)

        with self.db.transaction() as conn:
            # Delete removed node from index
            self.builder._remove_node(removed.id)

            # Remap edges: point removed→kept (skip self-loops and duplicates)
            # Done AFTER index_single since that wipes and rebuilds edges for kept node.
            from ormah.models.node import Connection, EdgeType

            affected_node_ids: set[str] = set()
            for edge in original_edges:
                new_source = kept.id if edge["source_id"] == removed.id else edge["source_id"]
                new_target = kept.id if edge["target_id"] == removed.id else edge["target_id"]

                # Skip self-loops
                if new_source == new_target:
                    continue

                # Skip if edge already exists in either direction
                existing = conn.execute(
                    "SELECT 1 FROM edges WHERE "
                    "((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)) "
                    "AND edge_type = ?",
                    (new_source, new_target, new_target, new_source, edge["edge_type"]),
                ).fetchone()
                if existing:
                    continue

                conn.execute(
                    "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (new_source, new_target, edge["edge_type"], edge["weight"], edge["created"]),
                )

                # Track which nodes need their markdown files updated
                if edge["source_id"] != removed.id:
                    affected_node_ids.add(edge["source_id"])

            # Restore incoming edges for the kept node that were wiped by index_single
            for edge in kept_incoming_edges:
                existing = conn.execute(
                    "SELECT 1 FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                    (edge["source_id"], edge["target_id"], edge["edge_type"]),
                ).fetchone()
                if not existing:
                    conn.execute(
                        "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (edge["source_id"], edge["target_id"], edge["edge_type"],
                         edge["weight"], edge["created"]),
                    )

            # Clean up auto-linker checked pairs:
            # - removed node: delete all (node is gone)
            # - kept node: invalidate if content changed (merged_content applied)
            conn.execute(
                "DELETE FROM auto_link_checked WHERE node_a = ? OR node_b = ?",
                (removed.id, removed.id),
            )
            if merged_content is not None or merged_title is not None:
                conn.execute(
                    "DELETE FROM auto_link_checked WHERE node_a = ? OR node_b = ?",
                    (kept.id, kept.id),
                )

            # Insert into merge_history
            merge_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO merge_history "
                "(id, proposal_id, kept_node_id, removed_node_id, removed_node_snapshot, "
                "original_edges, merged_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    merge_id,
                    proposal_id,
                    kept.id,
                    removed.id,
                    json.dumps(snapshot),
                    json.dumps(original_edges),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        self.file_store.soft_delete(removed.id)

        # Update markdown files: rewrite connections from removed→kept
        for node_id in affected_node_ids:
            neighbor = self.file_store.load(node_id)
            if neighbor is None:
                continue
            updated = False
            for c in neighbor.connections:
                if c.target == removed.id:
                    c.target = kept.id
                    updated = True
            if updated:
                self.file_store.save(neighbor)

        # Also fix the kept node's own connections that pointed to removed
        reload_kept = self.file_store.load(kept.id)
        if reload_kept:
            reload_kept.connections = [
                c for c in reload_kept.connections if c.target != removed.id
            ]
            self.file_store.save(reload_kept)

        kept_title = kept.title or kept.content[:60]
        removed_title = removed.title or removed.content[:60]
        return (
            f"Merged: kept \"{kept_title}\" ({kept.id[:8]}), "
            f"removed \"{removed_title}\" ({removed.id[:8]}). "
            f"Merge ID: {merge_id[:8]}"
        )

    def undo_merge(self, merge_id: str) -> str:
        """Rollback a merge by ID (supports prefix match). Returns confirmation."""
        # Support prefix match
        row = self.db.conn.execute(
            "SELECT * FROM merge_history WHERE id = ?", (merge_id,)
        ).fetchone()
        if row is None:
            row = self.db.conn.execute(
                "SELECT * FROM merge_history WHERE id LIKE ?", (merge_id + "%",)
            ).fetchone()
        if row is None:
            return f"Merge {merge_id} not found."
        if row["undone_at"] is not None:
            return f"Merge {row['id'][:8]} was already undone at {row['undone_at']}."

        # Reconstruct removed node from snapshot
        snapshot = json.loads(row["removed_node_snapshot"])
        node = MemoryNode.model_validate(snapshot)
        path = self.file_store.save(node)
        self.builder.index_single(path)
        self._index_embedding(node)

        # Delete remapped edges that were created during merge
        # (edges involving kept_id that originated from removed_id)
        original_edges = json.loads(row["original_edges"])
        with self.db.transaction() as conn:
            for edge in original_edges:
                remapped_source = row["kept_node_id"] if edge["source_id"] == row["removed_node_id"] else edge["source_id"]
                remapped_target = row["kept_node_id"] if edge["target_id"] == row["removed_node_id"] else edge["target_id"]
                if remapped_source == remapped_target:
                    continue
                conn.execute(
                    "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                    (remapped_source, remapped_target, edge["edge_type"]),
                )

            # Restore original edges (check both endpoints still exist)
            for edge in original_edges:
                src_exists = conn.execute(
                    "SELECT 1 FROM nodes WHERE id = ?", (edge["source_id"],)
                ).fetchone()
                tgt_exists = conn.execute(
                    "SELECT 1 FROM nodes WHERE id = ?", (edge["target_id"],)
                ).fetchone()
                if src_exists and tgt_exists:
                    conn.execute(
                        "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, created) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (edge["source_id"], edge["target_id"], edge["edge_type"],
                         edge["weight"], edge["created"]),
                    )

            # Mark undone
            conn.execute(
                "UPDATE merge_history SET undone_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), row["id"]),
            )

        title = node.title or node.content[:60]
        return f"Undone merge {row['id'][:8]}: restored \"{title}\" ({node.id[:8]})"

    def list_merges(self, limit: int = 20) -> list[dict]:
        """List recent merge history."""
        rows = self.db.conn.execute(
            "SELECT * FROM merge_history ORDER BY merged_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Audit log ---

    def _write_audit_log(
        self,
        operation: str,
        node_id: str,
        node_snapshot: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Insert an entry into the audit_log table."""
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO audit_log (operation, node_id, node_snapshot, detail, performed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    operation,
                    node_id,
                    node_snapshot,
                    detail,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_audit_log(
        self,
        limit: int = 20,
        node_id: str | None = None,
        operation: str | None = None,
    ) -> list[dict]:
        """List recent audit log entries, optionally filtered by node_id or operation."""
        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list = []
        if node_id is not None:
            query += " AND node_id = ?"
            params.append(node_id)
        if operation is not None:
            query += " AND operation = ?"
            params.append(operation)
        query += " ORDER BY performed_at DESC LIMIT ?"
        params.append(limit)
        rows = self.db.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _pick_keeper(a: MemoryNode, b: MemoryNode) -> tuple[MemoryNode, MemoryNode]:
        """Pick which node to keep: higher tier > longer content > more recent."""
        tier_rank = {"core": 3, "working": 2, "archival": 1}
        rank_a = tier_rank.get(a.tier.value, 0)
        rank_b = tier_rank.get(b.tier.value, 0)
        if rank_a != rank_b:
            return (a, b) if rank_a > rank_b else (b, a)
        if len(a.content) != len(b.content):
            return (a, b) if len(a.content) >= len(b.content) else (b, a)
        return (a, b) if a.updated >= b.updated else (b, a)

    def get_self(self) -> str:
        """Get formatted identity profile for the user."""
        if not self.user_node_id:
            return "No user identity profile exists yet."

        # Touch access on self node
        self._touch_access(self.user_node_id)

        # Get all nodes linked via defines edges from self node
        identity_nodes = self.graph.get_neighbors(
            self.user_node_id, depth=1, edge_types=["defines"]
        )

        # Touch access on each identity node
        for n in identity_nodes:
            self._touch_access(n["id"])

        from ormah.engine.traversal import format_identity_section

        return format_identity_section(identity_nodes)

    def _link_to_self(self, node: MemoryNode) -> None:
        """Create a defines edge from self node to the given node."""
        now = datetime.now(timezone.utc).isoformat()

        # Insert edge into DB
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source_id, target_id, edge_type, weight, created) "
                "VALUES (?, ?, 'defines', 1.0, ?)",
                (self.user_node_id, node.id, now),
            )

        # Update the self node's markdown connections
        self_node = self.file_store.load(self.user_node_id)
        if self_node:
            self_node.connections.append(
                Connection(target=node.id, edge=EdgeType.defines, weight=1.0)
            )
            self.file_store.save(self_node)

    def _touch_access(self, node_id: str) -> None:
        """Update access stats and FSRS stability on both disk and DB."""
        node = self.file_store.load(node_id)
        if node is None:
            return
        now = datetime.now(timezone.utc)

        # FSRS stability update
        review_anchor = node.last_review or node.last_accessed
        days_since = max((now - review_anchor).total_seconds() / 86400, 0.001)
        retrievability = math.exp(-days_since / node.stability)
        new_stability = node.stability * self.settings.fsrs_stability_growth * (retrievability ** -0.2)
        node.stability = round(min(new_stability, self.settings.fsrs_max_stability), 2)
        node.last_review = now

        # Standard access tracking
        node.last_accessed = now
        node.access_count += 1

        self.file_store.save(node)
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE nodes SET access_count = ?, last_accessed = ?, stability = ?, last_review = ? WHERE id = ?",
                (node.access_count, node.last_accessed.isoformat(), node.stability, node.last_review.isoformat(), node_id),
            )

    # --- Private helpers ---

    def _supplement_temporal(
        self,
        results: list[dict],
        limit: int,
        created_after: str,
        created_before: str | None,
        filters: dict,
    ) -> list[dict]:
        """Supplement results with SQL-based recent nodes when temporal filters are active.

        Fetches nodes directly by ``created`` column, deduplicates against
        existing results, applies type/tier/space filters, and appends to
        the result list up to *limit*.
        """
        existing_ids = {r["node"]["id"] for r in results}
        needed = limit - len(results)
        # Fetch more than needed to account for filter losses
        recent = self.graph.get_recent_nodes(
            limit=needed * 3,
            after=created_after,
            before=created_before,
        )

        types_filter = filters.get("types")
        tiers_filter = filters.get("tiers")
        spaces_filter = filters.get("spaces")

        added = 0
        for node in recent:
            if added >= needed:
                break
            if node["id"] in existing_ids:
                continue
            if types_filter and node["type"] not in types_filter:
                continue
            if tiers_filter and node["tier"] not in tiers_filter:
                continue
            if spaces_filter and node.get("space") not in spaces_filter:
                continue
            existing_ids.add(node["id"])
            # Use a low base score so these don't outrank relevance-matched results
            results.append({"node": node, "score": 0.001, "source": "temporal"})
            added += 1

        return results

    def _apply_space_scores(
        self, results: list[dict], default_space: str
    ) -> list[dict]:
        """Apply multiplicative space factors to search scores and re-sort.

        Current project: 1.0x (no penalty)
        Global (space=None): space_boost_global (default 0.85x)
        Other project: space_boost_other (default 0.6x)
        """
        boost_global = self.settings.space_boost_global
        boost_other = self.settings.space_boost_other

        for r in results:
            node = r["node"] if "node" in r else r
            space = node.get("space")
            if space == default_space:
                factor = 1.0
            elif space is None:
                factor = boost_global
            else:
                factor = boost_other
            r["score"] = r.get("score", 0.0) * factor

        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results

    def _spread_activation(self, results: list[dict], limit: int) -> list[dict]:
        """Enrich search results with graph neighbors via spreading activation.

        Takes the top seed_count results as seeds, traverses their depth-1 edges,
        scores neighbors, deduplicates against direct hits, and merges activated
        neighbors into the result list sorted by score — so a highly-connected
        neighbor can outrank a low-relevance direct hit.

        Results activated via ``contradicts`` edges are labelled with
        ``source="conflict"`` so the formatter can present them distinctly.
        """
        if not results:
            return results

        seed_count = self.settings.activation_seed_count
        max_per_seed = self.settings.activation_max_per_seed
        decay = self.settings.activation_decay

        seeds = results[:seed_count]
        direct_ids = {r["node"]["id"] if "node" in r else r["id"] for r in results}

        # neighbor_id -> (activation_score, seed_id, edge_type)
        activated: dict[str, tuple[float, str, str]] = {}

        for seed in seeds:
            seed_node = seed["node"] if "node" in seed else seed
            seed_id = seed_node["id"]
            seed_score = seed.get("score", 1.0)

            edges = self.graph.get_edges_for(seed_id)

            # Score all neighbors for this seed
            candidates: list[tuple[str, float, str]] = []  # (neighbor_id, score, edge_type)
            for edge in edges:
                neighbor_id = (
                    edge["target_id"] if edge["source_id"] == seed_id else edge["source_id"]
                )
                if neighbor_id in direct_ids:
                    continue

                edge_type = edge["edge_type"]
                edge_weight = edge.get("weight", 0.5)
                type_factor = _EDGE_TYPE_FACTORS.get(edge_type, 0.5)
                score = seed_score * edge_weight * type_factor * decay
                candidates.append((neighbor_id, score, edge_type))

            # Sort by score descending, cap per seed
            candidates.sort(key=lambda x: x[1], reverse=True)
            for neighbor_id, score, edge_type in candidates[:max_per_seed]:
                existing = activated.get(neighbor_id)
                if existing is None or score > existing[0]:
                    activated[neighbor_id] = (score, seed_id, edge_type)

        # Build activated result entries (batch-fetch all nodes at once)
        activated_node_map = self.graph.get_nodes_batch(list(activated.keys()))
        activated_results = []
        for neighbor_id, (score, seed_id, edge_type) in activated.items():
            node = activated_node_map.get(neighbor_id)
            if node is None:
                continue
            source = "conflict" if edge_type == "contradicts" else "activated"
            activated_results.append({
                "node": node,
                "score": score,
                "source": source,
                "activated_by": seed_id,
                "activation_edge": edge_type,
            })

        # Merge direct + activated results, sort by score descending
        merged = results + activated_results
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)

        return merged

    def _get_hybrid_search(self):
        if self._hybrid_search is not None:
            return self._hybrid_search
        try:
            from ormah.embeddings.hybrid_search import HybridSearch

            self._hybrid_search = HybridSearch(self.db, self.settings)
            return self._hybrid_search
        except ImportError:
            return None

    def _index_embedding(self, node: MemoryNode) -> None:
        try:
            from ormah.embeddings.vector_store import VectorStore
            from ormah.embeddings.encoder import get_encoder

            encoder = get_encoder(self.settings)
            vec_store = VectorStore(self.db)
            text = _embedding_text(node.title, node.content, self.settings.embedding_max_content_chars)
            embedding = encoder.encode(text)
            vec_store.upsert(node.id, embedding)
        except Exception as e:
            logger.warning("Failed to index embedding for node %s: %s", node.id[:8], e)

    def _auto_link_node(self, node: MemoryNode) -> list[tuple[str, str, float]]:
        """Find similar existing nodes and create edges. Returns [(id, title, similarity)]."""
        try:
            from ormah.embeddings.vector_store import VectorStore
            from ormah.embeddings.encoder import get_encoder

            encoder = get_encoder(self.settings)
            vec_store = VectorStore(self.db)

            text = _embedding_text(node.title, node.content, self.settings.embedding_max_content_chars)
            if not text:
                return []

            query_vec = encoder.encode(text)
            similar = vec_store.search(query_vec, limit=6)

            threshold = self.settings.auto_link_similarity_threshold
            cross_space_penalty = self.settings.auto_link_cross_space_penalty
            links = []

            for match in similar:
                if match["id"] == node.id:
                    continue

                similarity = match["similarity"]

                # Penalize cross-space pairs
                other = self.graph.get_node(match["id"])
                if other is not None:
                    src_space = node.space or ""
                    tgt_space = other.get("space") or ""
                    if src_space != tgt_space:
                        similarity -= cross_space_penalty

                if similarity < threshold:
                    continue

                # Check edge doesn't already exist
                existing = self.db.conn.execute(
                    "SELECT 1 FROM edges WHERE "
                    "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                    (node.id, match["id"], match["id"], node.id),
                ).fetchone()
                if existing:
                    continue

                other = self.graph.get_node(match["id"])
                title = (other.get("title") or other.get("content", "")[:50]) if other else match["id"][:8]
                links.append((match["id"], title, similarity))

            if links:
                with self.db.transaction() as conn:
                    for match_id, title, similarity in links:
                        conn.execute(
                            "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
                            "VALUES (?, ?, 'related_to', ?, ?)",
                            (node.id, match_id, round(similarity, 3),
                             datetime.now(timezone.utc).isoformat()),
                        )

                        # Also update the markdown file
                        from ormah.models.node import Connection, EdgeType
                        node.connections.append(
                            Connection(target=match_id, edge=EdgeType.related_to,
                                       weight=round(similarity, 2))
                        )
                self.file_store.save(node)  # persist auto-linked connections

            return links

        except Exception as e:
            logger.debug("Auto-link on remember failed: %s", e)
            return []

    # --- Provenance ---

    @staticmethod
    def get_file_provenance(watch_dir: str, rel_path: str) -> list[str]:
        """Return node_ids associated with a file via hippocampus state.

        Reads the hippocampus state file for *watch_dir* and returns the
        ``node_ids`` list stored for *rel_path*.  Returns an empty list if
        the state file doesn't exist or the file has no entry.
        """
        from ormah.background.hippocampus import _load_state

        state = _load_state(Path(watch_dir))
        entry = state.get(rel_path)
        if entry is None:
            return []
        return entry.get("node_ids", [])

    # --- Conversation ingestion ---

    def ingest_conversation(
        self,
        content: str,
        space: str | None = None,
        agent_id: str | None = None,
        dry_run: bool = False,
        extra_tags: list[str] | None = None,
    ) -> list[dict] | str:
        """Extract and store memories from raw conversation text.

        Uses the configured LLM to identify memorable information,
        deduplicates against existing memories, and creates nodes.

        When *dry_run* is True, runs extraction and dedup but skips storage.
        Returns list of dicts with extracted/created memory info,
        or an error string if the LLM is unavailable.
        """
        if len(content.strip()) < 50:
            return []

        extracted = self._extract_memories_llm(content)
        if isinstance(extracted, str):
            return extracted  # error message

        if not extracted:
            return []

        created = []
        skipped = 0
        for mem in extracted:
            if not isinstance(mem, dict):
                continue
            mem_content = mem.get("content", "").strip()
            if not mem_content:
                continue

            # Dedup: check if a very similar memory already exists (skip in dry_run)
            if not dry_run and self._is_duplicate_memory(mem_content):
                logger.debug("Skipping duplicate: %s", mem.get("title", mem_content[:40]))
                skipped += 1
                continue

            try:
                node_type = NodeType(mem.get("type", "fact"))
            except ValueError:
                node_type = NodeType.fact

            mem_title = mem.get("title") or _generate_title(mem_content)

            # Default confidence for auto-ingested memories: 0.7
            confidence = mem.get("confidence", 0.7)

            tags = mem.get("tags", []) + ["auto-ingested"] + (extra_tags or [])

            if dry_run:
                created.append({
                    "title": mem_title,
                    "content": mem_content,
                    "type": node_type.value,
                    "tags": tags,
                    "about_self": mem.get("about_self", False),
                    "confidence": confidence,
                })
                continue

            req = CreateNodeRequest(
                content=mem_content,
                type=node_type,
                title=mem_title,
                tags=tags,
                space=space,
                about_self=mem.get("about_self", False),
                confidence=confidence,
            )
            node_id, _ = self.remember(req, agent_id=agent_id or "ingester")
            created.append({
                "node_id": node_id,
                "title": mem_title,
            })

        if skipped:
            logger.debug("Ingestion: skipped %d duplicates", skipped)
        if created and not dry_run:
            logger.info("Ingested %d memories from conversation", len(created))
        return created

    def _extract_memories_llm(self, content: str) -> list[dict] | str:
        """Use configured LLM to extract memories from conversation text.

        Returns a list of memory dicts on success, or an error string
        if the LLM is unavailable.
        """
        try:
            from ormah.background.llm_client import llm_generate

            max_chars = self.settings.ingest_max_content_chars
            prompt = _INGEST_LLM_PROMPT.format(conversation=content[:max_chars])
            raw = llm_generate(self.settings, prompt, json_mode=True)
            if raw is None:
                return (
                    "No LLM available for server-side extraction. "
                    "Pass pre-extracted memories via the 'memories' parameter instead."
                )

            # Extract JSON from response — handle markdown fences and surrounding prose
            stripped = _extract_json(raw)
            logger.debug("LLM raw (%d chars), extracted JSON (%d chars): %.300s",
                         len(raw), len(stripped), stripped)
            result = json.loads(stripped)
            # Unwrap: support {"memories": [...]}, {"memories": {"memories": [...]}}, or bare list
            memories = result
            while isinstance(memories, dict) and "memories" in memories:
                memories = memories["memories"]
            if isinstance(memories, list):
                return memories
            return []
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return (
                "LLM extraction failed. "
                "Pass pre-extracted memories via the 'memories' parameter instead."
            )

    def _is_duplicate_memory(self, content: str) -> bool:
        """Check if a very similar memory already exists using vector search."""
        try:
            from ormah.embeddings.encoder import get_encoder
            from ormah.embeddings.vector_store import VectorStore

            encoder = get_encoder(self.settings)
            vec_store = VectorStore(self.db)

            query_vec = encoder.encode(content)
            results = vec_store.search(query_vec, limit=1)

            if results and results[0]["similarity"] >= self.settings.auto_merge_threshold:
                return True
        except Exception as e:
            logger.debug("Dedup check failed: %s", e)

        return False


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Extract JSON from an LLM response that may contain markdown fences or prose."""
    # Try direct parse first
    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        return stripped

    # Look for ```json ... ``` fenced block
    m = _FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()

    # Last resort: find first { or [ to last matching } or ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end > start:
            return raw[start : end + 1]

    return stripped


_INGEST_LLM_PROMPT = """\
You are a memory curator for a persistent knowledge graph. Your job: read a conversation and extract memories that will be valuable in future sessions — days or weeks later, when all context is gone.

These memories are stored as typed nodes in a graph with semantic search. They will be retrieved by an AI assistant to provide context it wouldn't otherwise have. Every memory you extract should pass this test: "Would an AI assistant benefit from knowing this when helping the user on a related task in the future?"

## Quality bar

A good memory is **specific, self-contained, and searchable**. It must:
1. Be understood in complete isolation — no "this", "the above", "as discussed"
2. Include concrete details — names, versions, paths, specific choices, numbers
3. Have a title that distinguishes it from related memories (titles are weighted 10x in search)

BAD: "The project uses SQLite for storage"
GOOD: "Chose SQLite over PostgreSQL for ormah's index because the system is local-first and single-user. The DB stores FTS5 indexes, vector embeddings (via sqlite-vec), and node metadata. File-based markdown is the source of truth; SQLite is a derived index."

BAD: "User prefers dark mode"
GOOD: "User prefers dark mode in all editors and terminals. Specifically mentioned VS Code, iTerm2, and Obsidian."

BAD: "Discussed authentication approach"
GOOD: "Decided to use JWT tokens over session cookies for the API because the client is a CLI tool, not a browser. Tokens are stored in ~/.config/app/auth.json with 7-day expiry."

## What to extract (priority order)

1. **Decisions with reasoning** (type: "decision") — The most valuable memory type. What was chosen, what was rejected, and WHY. The reasoning prevents re-litigating the same decision in future sessions. Always name the alternatives.

2. **User corrections and "no" moments** (type: "preference" or "decision") — When the user pushed back, said "no", or corrected the AI. These reveal unstated preferences. Set about_self=true.

3. **Preferences and opinions** (type: "preference") — Must be specific. "Prefers map/filter over for loops, avoids classes unless modeling state" not "prefers functional style". Set about_self=true — this links the memory to the user's identity profile.

4. **Architecture and design patterns** (type: "fact" or "concept") — HOW the system works, not just what it does. Include the constraints that shaped the design. Name specific files, modules, patterns.

5. **Procedures discovered through effort** (type: "procedure") — Steps that weren't obvious. Include the exact commands, flags, paths. These save future sessions from re-discovering the same process.

6. **Goals and strategic direction** (type: "goal") — What the user is trying to achieve long-term. Include context on why this goal matters to them.

7. **Surprising findings** (type: "observation") — Bugs with non-obvious causes, unexpected library behavior, performance discoveries. These are high-value because they prevent repeat mistakes.

8. **Personal identity facts** (type: "person" or "fact") — Name, role, email, location, team. Set about_self=true — person nodes get promoted to core tier.

## What NOT to extract

- Vague summaries that could apply to any project
- Generic technical knowledge the AI already has
- Intermediate debugging steps that led nowhere
- Routine code changes with no decision or learning behind them
- Information already captured more specifically by another memory in your output
- The same fact restated at different granularities — pick the most specific version
- **Code read-throughs where the user is just tracing existing logic** — if the user reads code and the AI confirms what it says, this information is already in the codebase. Only extract things that go BEYOND what's already in the code: decisions, preferences, surprises, corrections. "I traced through the pipeline and it works as expected" is NOT worth storing.

## Deduplication within your output

Before adding a memory to your list, check if you've already extracted something that covers the same ground. Prefer fewer, richer memories over many thin ones. If a decision memory already explains the architecture, you don't need a separate architecture fact.

## Output format

For each memory:
- "content": 2-5 sentences. Be specific — include names, versions, paths, flags. For decisions, always state what was rejected and why. Write as if explaining to a knowledgeable colleague who has no context about this conversation.
- "type": One of: fact, decision, preference, event, person, project, concept, procedure, goal, observation. Choose carefully — type affects how the memory is weighted, stored, and retrieved.
- "title": 5-12 words. Must be specific enough to distinguish this memory from related ones. The title is heavily weighted in search — make it count. BAD: "Database choice". GOOD: "Chose SQLite over Postgres for local-first single-user index".
- "tags": 2-5 tags. Include the project name if mentioned, technology names, and domain terms. Tags are indexed for search.
- "about_self": true if about the user's identity, preferences, or personal information. This triggers special handling: person types get promoted to core memory; preferences are linked to the user's identity profile.
- "confidence": 0.0-1.0. Use 1.0 for explicit statements by the user. Use 0.7-0.9 for clear but unstated implications. Use 0.4-0.6 for inferences you're less sure about. Low confidence memories are penalized in search ranking, so be honest.

Return: {{"memories": [...]}}
Return {{"memories": []}} if nothing worth remembering was discussed.

## Conversation

{conversation}
"""
