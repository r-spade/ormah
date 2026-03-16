"""Hybrid search combining FTS5 + vector search with Reciprocal Rank Fusion.

Uses RRF (Cormack et al. 2009) to fuse ranked lists from FTS5 and vector
retrievers.  RRF scores by rank position rather than raw score magnitude,
making it immune to scale differences between retrievers and naturally
boosting nodes found by both sources.

Additional scoring signals (recency, access frequency, tier) are blended
after the base fusion to personalize results without distorting retrieval.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ormah.config import Settings

if TYPE_CHECKING:
    from ormah.index.db import Database

logger = logging.getLogger(__name__)

_QUESTION_PATTERN = re.compile(
    r"^\s*(where|what|who|whom|when|why|how|which|is|are|was|were|do|does|did|can|could|should|would|will)\b",
    re.IGNORECASE,
)


def _is_question_query(query: str) -> bool:
    """Detect whether a query is a natural language question."""
    return _QUESTION_PATTERN.search(query) is not None
from ormah.embeddings.encoder import get_encoder
from ormah.embeddings.vector_store import VectorStore
from ormah.index.graph import GraphIndex


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    weights: list[float],
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked lists using weighted Reciprocal Rank Fusion.

    Each list contributes ``weight / (k + rank)`` for every node it contains
    (rank is 1-based).  Nodes appearing in multiple lists accumulate higher
    scores, naturally rewarding agreement between retrievers.

    Args:
        ranked_lists: Ordered lists of node IDs (highest relevance first).
        weights: Per-list weight factors (must match length of *ranked_lists*).
        k: Smoothing constant (default 60).  Higher values reduce the
           influence of high ranks; lower values sharpen the distinction.

    Returns:
        Mapping of node ID → fused RRF score.
    """
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, node_id in enumerate(ranked, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + weight / (k + rank)
    return scores


class HybridSearch:
    """Combines FTS5 full-text search with sqlite-vec vector search."""

    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.conn = db.conn
        self.settings = settings
        self.graph = GraphIndex(db.conn)
        self.vec_store = VectorStore(db)
        self.encoder = get_encoder(settings)

    def search(
        self,
        query: str,
        limit: int = 10,
        types: list[str] | None = None,
        tiers: list[str] | None = None,
        spaces: list[str] | None = None,
        tags: list[str] | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search with Reciprocal Rank Fusion."""
        sim_threshold = self.settings.similarity_threshold

        # Widen candidate pool when temporal filters are active so the
        # post-filter has enough candidates to work with.
        has_temporal = created_after is not None or created_before is not None
        candidate_multiplier = 10 if has_temporal else 3

        # FTS results — use raw FTS5 scores (higher = better match)
        fts_results = self.graph.fts_search(query, limit=limit * candidate_multiplier)
        fts_scores = {r["id"]: r["score"] for r in fts_results}

        # Vector results — use cosine similarity, drop below threshold
        vec_scores: dict[str, float] = {}
        try:
            query_vec = self.encoder.encode_query(query)
            vec_results = self.vec_store.search(query_vec, limit=limit * candidate_multiplier)
            vec_scores = {
                r["id"]: r["similarity"]
                for r in vec_results
                if r["similarity"] >= sim_threshold
            }
        except Exception as e:
            logger.warning("Vector search failed: %s", e)

        # Build ranked lists (ordered by descending score/similarity)
        fts_ranked = sorted(fts_scores, key=fts_scores.get, reverse=True)
        vec_ranked = sorted(vec_scores, key=vec_scores.get, reverse=True)

        # Scale RRF weights for question queries
        is_question = _is_question_query(query)
        fts_w = self.settings.fts_weight
        vec_w = self.settings.vector_weight
        if is_question:
            fts_w *= self.settings.question_fts_weight_scale
            vec_w *= self.settings.question_vector_weight_scale

        # Reciprocal Rank Fusion
        rrf_scores = _reciprocal_rank_fusion(
            ranked_lists=[fts_ranked, vec_ranked],
            weights=[fts_w, vec_w],
            k=self.settings.rrf_k,
        )

        # Normalize RRF scores to 0-1 range so they're comparable with similarity.
        # Without this, RRF (~0.01) is drowned by raw similarity (~0.4).
        # Use min-max normalization when scores are spread enough; fall back to
        # max-normalization when scores are too close to avoid over-amplification.
        if rrf_scores:
            max_rrf = max(rrf_scores.values())
            min_rrf = min(rrf_scores.values())
            spread = max_rrf - min_rrf

            if max_rrf > 0 and spread > max_rrf * self.settings.rrf_min_spread_ratio:
                # Min-max: [min, max] → [0, 1]
                for node_id in rrf_scores:
                    rrf_scores[node_id] = (rrf_scores[node_id] - min_rrf) / spread
            elif max_rrf > 0:
                # Fallback: max-normalization (scores too close for min-max)
                for node_id in rrf_scores:
                    rrf_scores[node_id] /= max_rrf

        # Content length penalty — dampen vector similarity for long documents.
        # Long docs get moderate vector similarity with everything because their
        # embeddings average over many topics.
        length_threshold = self.settings.length_penalty_threshold
        content_lengths: dict[str, int] = {}
        if length_threshold > 0:
            candidate_ids_for_len = list(rrf_scores.keys())
            if candidate_ids_for_len:
                placeholders = ",".join("?" for _ in candidate_ids_for_len)
                rows = self.conn.execute(
                    f"SELECT id, length(content) as len FROM nodes WHERE id IN ({placeholders})",
                    candidate_ids_for_len,
                ).fetchall()
                content_lengths = {r["id"]: r["len"] for r in rows}

        # Blend normalized RRF with raw vector similarity.
        # RRF captures rank-agreement between FTS and vector retrievers.
        # Similarity blend restores magnitude that RRF discards.
        # For questions, lean heavily on raw similarity — it's the only
        # absolute quality signal after stop-word removal guts FTS precision.
        similarity_blend_weight = (
            self.settings.question_similarity_blend_weight
            if is_question
            else self.settings.similarity_blend_weight
        )
        rrf_weight = 1.0 - similarity_blend_weight
        fts_only_dampening = self.settings.fts_only_dampening
        for node_id in rrf_scores:
            raw_sim = vec_scores.get(node_id, 0.0)
            if raw_sim > 0:
                # Apply length penalty before blending
                if length_threshold > 0:
                    content_len = content_lengths.get(node_id, 200)
                    if content_len > length_threshold:
                        penalty = max(0.1, length_threshold / content_len)
                        raw_sim *= penalty
                rrf_scores[node_id] = rrf_weight * rrf_scores[node_id] + similarity_blend_weight * raw_sim
            else:
                # FTS-only result with no semantic match — dampen
                rrf_scores[node_id] *= fts_only_dampening

        scored = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Batch-fetch all candidate nodes and tags in single queries
        candidate_ids = [node_id for node_id, _ in scored]
        node_map = self.graph.get_nodes_batch(candidate_ids)
        tags_map = self.graph.get_tags_batch(candidate_ids) if tags else {}

        # Precompute boost parameters
        now = datetime.now(timezone.utc)
        recency_boost = self.settings.recency_boost
        half_life = self.settings.recency_half_life_days
        access_boost = self.settings.access_boost
        tier_boosts = {
            "core": self.settings.tier_boost_core,
            "working": self.settings.tier_boost_working,
            "archival": self.settings.tier_boost_archival,
        }

        # Apply filters, compute boosted scores, build results
        results = []
        for node_id, base_score in scored:
            if len(results) >= limit:
                break

            node = node_map.get(node_id)
            if node is None:
                continue

            # Apply filters
            if types and node["type"] not in types:
                continue
            if tiers and node["tier"] not in tiers:
                continue
            if spaces and node.get("space") not in spaces:
                continue
            if tags:
                node_tags = tags_map.get(node_id, set())
                if not set(tags) & node_tags:
                    continue

            # Temporal filters
            if created_after and (node.get("created") or "") < created_after:
                continue
            if created_before and (node.get("created") or "") > created_before:
                continue

            # Title match bonus — query terms in title get a significant boost
            # Disabled for question queries where residual tokens cause false positives
            title_match_boost = 0.0 if is_question else self.settings.title_match_boost
            if title_match_boost > 0:
                title = re.sub(r'[^\w\s]', '', (node.get("title") or "")).lower()
                query_clean = re.sub(r'[^\w\s]', '', query).lower()
                query_tokens = set(query_clean.split())
                title_tokens = set(title.split())
                overlap = sum(1 for t in query_tokens if len(t) > 2 and t in title_tokens)
                if overlap > 0:
                    title_bonus = title_match_boost * (overlap / max(len(query_tokens), 1))
                    base_score *= (1.0 + title_bonus)

            # --- Multiplicative factors from enrichment fields ---
            confidence = node.get("confidence")
            if confidence is None:
                confidence = 1.0
            confidence_factor = 0.4 + 0.6 * confidence  # [0.4, 1.0]

            # Hard-filter expired nodes
            valid_until_raw = node.get("valid_until")
            if valid_until_raw:
                try:
                    valid_until = datetime.fromisoformat(valid_until_raw)
                    if now >= valid_until:
                        continue  # Hard-filter: expired nodes excluded from results
                except (ValueError, TypeError):
                    pass

            adjusted_score = base_score * confidence_factor

            # --- Proportional tiebreakers (scale with base score) ---
            # Recency boost: proportional to adjusted score with exponential decay
            r_boost = 0.0
            if recency_boost > 0 and half_life > 0:
                try:
                    last_accessed = datetime.fromisoformat(node["last_accessed"])
                    days_ago = max((now - last_accessed).total_seconds() / 86400, 0)
                    decay_factor = math.exp(-days_ago * math.log(2) / half_life)
                    r_boost = adjusted_score * recency_boost * decay_factor
                except (ValueError, KeyError):
                    pass

            # Access frequency boost: proportional to adjusted score, logarithmic scale
            a_boost = 0.0
            if access_boost > 0:
                count = node.get("access_count", 0) or 0
                a_boost = adjusted_score * access_boost * math.log1p(count) / math.log1p(20)

            # Tier boost: multiplicative factor on adjusted score
            tier_factor = 1.0 + tier_boosts.get(node.get("tier", "working"), 0.0)

            final_score = adjusted_score * tier_factor + r_boost + a_boost

            results.append({"node": node, "score": round(final_score, 6), "source": "hybrid"})

        # Re-sort by boosted score since boosts may reorder results
        results.sort(key=lambda x: x["score"], reverse=True)

        # Drop results below minimum score threshold
        min_score = self.settings.min_result_score
        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]

        return results
