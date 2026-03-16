"""Builds the 'always loaded' core context for system prompt injection."""

from __future__ import annotations

import logging
import re

import numpy as np

from ormah.engine.traversal import (
    format_context,
    format_context_with_project,
    format_identity_section,
)
from ormah.index.graph import GraphIndex

logger = logging.getLogger(__name__)

_WHISPER_FRAMING = (
    "# Whispered memories (ormah)\n"
    "Memories from the user's knowledge graph, selected for relevance to the current prompt. "
    "Use them naturally if they add useful context; ignore if not relevant. "
    "If a memory is truncated and you need the full content, you can use the recall tool with its node ID."
)


def _first_sentence_truncate(content: str, max_len: int) -> str:
    """Return the first sentence of content, capped to max_len."""
    content = content.strip()
    if len(content) <= max_len:
        return content
    # Find first sentence boundary
    for end in ('. ', '.\n', '; ', '\n'):
        idx = content.find(end)
        if 0 < idx < max_len:
            return content[:idx + 1]
    return content[:max_len]


class ContextBuilder:
    """Builds agent context from core memories."""

    def __init__(self, graph: GraphIndex, engine=None) -> None:
        self.graph = graph
        self.engine = engine
        self._classifier = None  # lazy-init PromptClassifier

    def _get_classifier(self):
        """Get or create the prompt intent classifier (uses engine's encoder)."""
        if self._classifier is not None:
            return self._classifier
        if not self.engine:
            return None
        try:
            from ormah.engine.prompt_classifier import PromptClassifier

            hybrid_search = self.engine._get_hybrid_search()
            if hybrid_search is None:
                return None
            encoder = hybrid_search.encoder
            threshold = getattr(self.engine, "settings", None)
            threshold = (
                threshold.whisper_intent_threshold if threshold else 0.65
            )
            self._classifier = PromptClassifier(encoder, threshold=threshold)
            return self._classifier
        except Exception as e:
            logger.warning("Failed to create prompt classifier: %s", e)
            return None

    def _get_tags(self, node_id: str) -> set[str]:
        """Get tags for a node from the DB."""
        rows = self.graph.conn.execute(
            "SELECT tag FROM node_tags WHERE node_id = ?", (node_id,)
        ).fetchall()
        return {r["tag"] for r in rows}

    def build_core_context(
        self,
        space: str | None = None,
        user_node_id: str | None = None,
        task_hint: str | None = None,
        max_nodes: int | None = None,
    ) -> str:
        """Get core-tier nodes + project working nodes formatted for system prompt.

        When *task_hint* is provided, candidate memories are scored against
        the hint embedding and only the top-N most relevant are returned.
        Identity nodes are always included regardless of scoring.
        """
        core_nodes = self.graph.get_nodes_by_tier("core")

        # Separate identity nodes from other core nodes
        identity_nodes: list[dict] = []
        identity_ids: set[str] = set()
        if user_node_id:
            identity_ids.add(user_node_id)
            identity_nodes = self.graph.get_neighbors(
                user_node_id, depth=1, edge_types=["defines"]
            )
            identity_ids.update(n["id"] for n in identity_nodes)

            for n in core_nodes:
                if n["id"] in identity_ids:
                    continue
                tags = self._get_tags(n["id"])
                if "about_self" in tags:
                    identity_nodes.append(n)
                    identity_ids.add(n["id"])

        other_core = [n for n in core_nodes if n["id"] not in identity_ids]

        # Gather working nodes for the project
        working_nodes: list[dict] = []
        if space is not None:
            working_nodes = [
                n for n in self.graph.get_nodes_by_tier("working")
                if n.get("space") == space
            ]

        # Adaptive filtering when task_hint is given
        if task_hint and self.engine:
            candidate_ids = {n["id"] for n in other_core + working_nodes}
            try:
                search_results = self.engine.recall_search_structured(
                    query=task_hint,
                    limit=max_nodes or 20,
                    default_space=space,
                    tiers=["core", "working"],
                    touch_access=False,
                )
                # Keep only results that are in our candidate set
                filtered = [r["node"] for r in search_results if r["node"]["id"] in candidate_ids]
                core_ids = {n["id"] for n in other_core}
                other_core = [n for n in filtered if n["id"] in core_ids]
                working_nodes = [n for n in filtered if n["id"] not in core_ids]
            except Exception as e:
                logger.warning("Hybrid context filtering failed, falling back to capped context: %s", e)
                effective_max = max_nodes or 20
                other_core, working_nodes = self._cap_by_space(
                    other_core, working_nodes, space, effective_max
                )
        elif task_hint:
            # Fallback: engine not available, use old vector-only method
            candidates = other_core + working_nodes
            filtered = self._filter_by_hint(
                candidates, task_hint, max_nodes=max_nodes or 20
            )
            if filtered is not None:
                core_ids = {n["id"] for n in other_core}
                other_core = [n for n in filtered if n["id"] in core_ids]
                working_nodes = [n for n in filtered if n["id"] not in core_ids]
            else:
                effective_max = max_nodes or 20
                other_core, working_nodes = self._cap_by_space(
                    other_core, working_nodes, space, effective_max
                )
        else:
            # No task_hint: space-partition and cap to max_nodes
            effective_max = max_nodes or 20
            other_core, working_nodes = self._cap_by_space(
                other_core, working_nodes, space, effective_max
            )

        # Filter identity nodes when task_hint is given
        if task_hint and identity_nodes:
            identity_nodes = self._filter_identity(identity_nodes, task_hint)

        # Deduplicate: remove nodes from core/working that duplicate identity
        # entries (by ID or by title). This prevents the same preference from
        # appearing in both "About the User" and the core/project sections.
        identity_titles = {n.get("title") for n in identity_nodes if n.get("title")}
        other_core = [
            n for n in other_core
            if n["id"] not in identity_ids and n.get("title") not in identity_titles
        ]
        working_nodes = [
            n for n in working_nodes
            if n["id"] not in identity_ids and n.get("title") not in identity_titles
        ]

        # Build identity section
        identity_text = format_identity_section(identity_nodes) if identity_nodes else ""

        if space is None and not working_nodes:
            core_text = format_context(other_core)
        else:
            core_text = format_context_with_project(
                other_core, working_nodes, space or "default"
            )

        if identity_text:
            return identity_text + "\n\n" + core_text
        return core_text

    def build_whisper_context(
        self,
        prompt: str,
        space: str | None = None,
        user_node_id: str | None = None,
        max_nodes: int = 8,
        min_score: float = 0.45,
        identity_max: int = 5,
        max_content_len: int = 150,
        reranker_enabled: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_min_score: float = 0.0,
        reranker_blend_alpha: float = 0.4,
        reranker_max_doc_chars: int = 512,
        recent_prompts: list[str] | None = None,
        topic_shift_enabled: bool = False,
        topic_shift_threshold: float = 0.75,
        injection_gate: float = 0.50,
        content_total_budget: int = 0,
        content_min_per_node: int = 100,
        content_max_per_node: int = 500,
    ) -> str:
        """Build compact whisper context for involuntary recall injection.

        Key differences from build_core_context:
        - Hard min-score threshold: results below min_score are dropped.
        - Tighter identity cap.
        - Compact formatting (shorter content truncation).
        - Returns empty string on failure instead of full dump.
        """
        if not prompt.strip():
            return ""

        # Short prompts (≤2 alphanumeric chars) are navigational ("y", "ok",
        # "...", "---") — skip search
        stripped = re.sub(r'[^a-zA-Z0-9]', '', prompt.strip())
        if len(stripped) <= 2:
            return ""

        if not self.engine:
            return ""

        # Topic-shift detection: skip injection when topic hasn't changed
        if topic_shift_enabled and recent_prompts and len(recent_prompts) >= 1:
            try:
                hybrid_search = self.engine._get_hybrid_search()
                if hybrid_search is not None:
                    encoder = hybrid_search.encoder
                    current_vec = encoder.encode(prompt)
                    recent_vecs = encoder.encode_batch(recent_prompts[-3:])
                    centroid = np.mean(recent_vecs, axis=0)
                    norm_current = np.linalg.norm(current_vec)
                    norm_centroid = np.linalg.norm(centroid)
                    if norm_current > 0 and norm_centroid > 0:
                        similarity = float(
                            np.dot(current_vec, centroid)
                            / (norm_current * norm_centroid)
                        )
                        if similarity > topic_shift_threshold:
                            return ""  # same topic, skip injection
            except Exception as e:
                logger.warning("Topic-shift detection failed, proceeding with whisper: %s", e)

        # Classify prompt intent before searching
        intent = None
        classifier = self._get_classifier()
        if classifier is not None:
            try:
                intent = classifier.classify(prompt)
            except Exception as e:
                logger.warning("Prompt classification failed, using default search: %s", e)

        # conversational-only → inject nothing
        if intent is not None and intent.categories == ["conversational"]:
            return ""

        # identity-only → skip general search, use existing identity path below
        identity_only = intent is not None and intent.categories == ["identity"]

        # Build context-enhanced search query from recent prompts
        search_query = prompt
        if recent_prompts:
            # Join last few prompts with current to give embedding model
            # topic context for vague follow-ups like "continue" or "more"
            context_parts = recent_prompts[-3:] + [prompt]
            search_query = " ".join(context_parts)

        # Build search kwargs, merging any intent-derived params
        search_kwargs: dict = {
            "query": search_query,
            "limit": max_nodes + identity_max,
            "default_space": space,
            "tiers": ["core", "working"],
            "touch_access": False,
        }
        if intent is not None:
            # Extract search_query override before merging (it's not a
            # recall_search_structured kwarg — it overrides our local query).
            intent_search_query = intent.search_params.pop("search_query", None)
            search_kwargs.update(intent.search_params)
            if intent_search_query is not None:
                search_kwargs["query"] = intent_search_query

        # Always run search — even for identity-only queries, search finds
        # location/work/study nodes that graph neighbors alone miss.
        try:
            search_results = self.engine.recall_search_structured(**search_kwargs)
        except Exception as e:
            logger.warning("Whisper search failed: %s", e)
            return ""

        # Per-intent adjustments: temporal queries rely on the created_after
        # filter for relevance rather than semantic similarity, so we relax
        # both the min-score threshold and the reranker threshold.
        has_temporal = intent is not None and "temporal" in intent.categories

        # Apply min-score threshold (relaxed for temporal queries whose
        # vague phrasing like "what did we do today" scores poorly against
        # specific memory content — the created_after filter already ensures
        # temporal relevance).  Temporal-supplement results (source="temporal")
        # are always kept — they were fetched by SQL recency, not semantic
        # similarity, so their low base score (0.001) is not meaningful.
        effective_min_score = min(min_score, 0.30) if has_temporal else min_score
        search_results = [
            r for r in search_results
            if r.get("score", 0) >= effective_min_score or r.get("source") == "temporal"
        ]
        effective_reranker_min = 0.0 if has_temporal else reranker_min_score

        # Cross-encoder reranking
        if reranker_enabled and search_results:
            try:
                from ormah.embeddings.reranker import rerank

                search_results = rerank(
                    query=prompt,
                    candidates=search_results,
                    model_name=reranker_model,
                    min_score=effective_reranker_min,
                    blend_alpha=reranker_blend_alpha,
                    max_doc_chars=reranker_max_doc_chars,
                )

            except Exception as e:
                logger.warning("Whisper reranker failed, using embedding scores: %s", e)

        # Injection gate: require at least one result with a strong enough
        # blended score to justify injection.  Temporal queries are exempt
        # (they rely on time filtering, not semantic relevance).
        if not has_temporal and search_results:
            max_blended = max(r.get("score", 0.0) for r in search_results)
            if max_blended < injection_gate:
                search_results = []
            else:
                # Score-floor: only keep results that individually clear the
                # injection gate.  Weak queries naturally get fewer results
                # instead of padding to max_nodes with marginal matches.
                search_results = [r for r in search_results if r.get("score", 0) >= injection_gate]

        # Temporal queries: re-sort by recency (most recent first).
        # Semantic scores already filtered noise via the 0.45 threshold,
        # but users expect chronological ordering for "what did we do today".
        if has_temporal and search_results:
            search_results.sort(
                key=lambda r: r["node"].get("created") or "",
                reverse=True,
            )

        # Separate identity nodes from search results.
        # Identity nodes compete on search merit like everything else —
        # no proactive graph-neighbor fetch.  We query identity neighbor
        # *IDs* (cheap) to classify search results, but don't inject
        # neighbors that didn't score well in search.
        identity_ids: set[str] = set()
        if user_node_id:
            identity_ids.add(user_node_id)

            # Collect identity-linked IDs from graph edges (lightweight ID query)
            neighbor_rows = self.graph.conn.execute(
                "SELECT target_id FROM edges WHERE source_id = ? AND edge_type = 'defines'",
                (user_node_id,),
            ).fetchall()
            identity_ids.update(r["target_id"] for r in neighbor_rows)

            # Also check about_self tags on search results
            for r in search_results:
                node = r["node"]
                if node["id"] in identity_ids:
                    continue
                tags = self._get_tags(node["id"])
                if "about_self" in tags:
                    identity_ids.add(node["id"])

        # Split results into identity vs non-identity, tracking top identity score
        identity_results: list[dict] = []
        other_results: list[dict] = []
        top_identity_score: float = 0.0
        for r in search_results:
            node = r["node"]
            if node["id"] in identity_ids:
                identity_results.append(node)
                top_identity_score = max(top_identity_score, r.get("score", 0))
            else:
                other_results.append(node)

        # Don't inject identity if no topical results survived AND
        # identity results scored low — prevents identity dump for
        # vague/off-topic prompts while keeping high-confidence identity
        # matches (e.g. "where does X live" → score 0.99).
        if not identity_only and not other_results and top_identity_score < min_score:
            identity_results = []

        # For identity-only intent, fall back to graph neighbors when
        # search didn't surface any identity nodes (e.g. "who am I?").
        if identity_only and not identity_results and user_node_id:
            identity_neighbors = self.graph.get_neighbors(
                user_node_id, depth=1, edge_types=["defines"]
            )
            if identity_neighbors:
                identity_results = identity_neighbors

        # Cap identity nodes
        if identity_results:
            # Always keep person/preference, filter the rest
            always_keep = [n for n in identity_results if n.get("type") in self._ALWAYS_KEEP_TYPES]
            rest = [n for n in identity_results if n.get("type") not in self._ALWAYS_KEEP_TYPES]
            remaining = max(identity_max - len(always_keep), 0)
            identity_results = always_keep + rest[:remaining]

        # Cap non-identity to remaining budget
        non_identity_budget = max(max_nodes - len(identity_results), 0)
        other_results = other_results[:non_identity_budget]

        # Split other results into core vs working for formatting
        core_nodes = [n for n in other_results if n.get("tier") == "core"]
        working_nodes = [n for n in other_results if n.get("tier") != "core"]

        # Dynamic content budget: distribute total chars across results
        effective_content_len = max_content_len
        if content_total_budget > 0:
            total_results = len(identity_results) + len(other_results)
            if total_results > 0:
                per_node = content_total_budget // total_results
                effective_content_len = max(
                    content_min_per_node,
                    min(content_max_per_node, per_node),
                )

        # Truncate content for whisper nodes (defensive copy to avoid
        # mutating objects that may be referenced elsewhere)
        for node_list in (identity_results, core_nodes, working_nodes):
            for i, node in enumerate(node_list):
                content = node.get("content", "")
                if content and len(content) > effective_content_len:
                    node_list[i] = node = dict(node)
                    node["content"] = _first_sentence_truncate(content, effective_content_len)

        # Format sections (whisper uses ## headers and includes node IDs)
        identity_text = (
            format_identity_section(
                identity_results,
                max_content_len=effective_content_len,
                header_prefix="##",
                include_ids=True,
            )
            if identity_results
            else ""
        )

        if space is not None and working_nodes:
            memories_text = format_context_with_project(
                core_nodes, working_nodes, space,
                max_content_len=effective_content_len,
                header_prefix="##",
                include_ids=True,
            )
        elif core_nodes or working_nodes:
            memories_text = format_context(
                core_nodes + working_nodes,
                max_content_len=effective_content_len,
                header_prefix="##",
                include_ids=True,
            )
        else:
            memories_text = ""

        parts = [p for p in (identity_text, memories_text) if p]
        body = "\n\n".join(parts)
        if not body:
            return ""
        return _WHISPER_FRAMING + "\n\n" + body

    @staticmethod
    def _cap_by_space(
        core_nodes: list[dict],
        working_nodes: list[dict],
        space: str | None,
        max_nodes: int,
    ) -> tuple[list[dict], list[dict]]:
        """Space-partition and cap non-identity nodes to *max_nodes*.

        Buckets (in priority order):
        1. Current-space core nodes
        2. Global (space=None) core nodes
        3. Other-project core nodes
        4. Working nodes (already scoped to current space)

        Within each bucket, nodes are sorted by importance descending.
        """

        def _by_importance(n: dict) -> float:
            return -(n.get("importance") or 0.0)

        current_core: list[dict] = []
        global_core: list[dict] = []
        other_core: list[dict] = []

        for n in core_nodes:
            ns = n.get("space")
            if ns == space and space is not None:
                current_core.append(n)
            elif ns is None:
                global_core.append(n)
            else:
                other_core.append(n)

        current_core.sort(key=_by_importance)
        global_core.sort(key=_by_importance)
        other_core.sort(key=_by_importance)
        working_nodes = sorted(working_nodes, key=_by_importance)

        # Fill up to max_nodes in priority order
        selected: list[dict] = []
        for bucket in (current_core, global_core, other_core, working_nodes):
            for n in bucket:
                if len(selected) >= max_nodes:
                    break
                selected.append(n)
            if len(selected) >= max_nodes:
                break

        # Split back into core vs working
        core_ids = {n["id"] for n in core_nodes}
        capped_core = [n for n in selected if n["id"] in core_ids]
        capped_working = [n for n in selected if n["id"] not in core_ids]
        return capped_core, capped_working

    def _filter_by_hint(
        self, candidates: list[dict], task_hint: str, max_nodes: int = 20
    ) -> list[dict] | None:
        """Score candidates against a task hint and return the top-N.

        Returns None on failure (caller should fall back to full dump).
        """
        try:
            from ormah.config import settings as default_settings
            from ormah.embeddings.encoder import get_encoder
            from ormah.embeddings.vector_store import VectorStore

            encoder = get_encoder(default_settings)
            if self.engine is None:
                return None
            vec_store = VectorStore(self.engine.db)

            hint_vec = encoder.encode_query(task_hint)

            scored: list[tuple[dict, float]] = []
            for node in candidates:
                node_vec = vec_store.get(node["id"])
                if node_vec is None:
                    # No embedding — give neutral score based on importance alone
                    importance = node.get("importance") or 0.5
                    scored.append((node, 0.3 * importance))
                    continue

                similarity = float(np.dot(hint_vec, node_vec) / (
                    np.linalg.norm(hint_vec) * np.linalg.norm(node_vec) + 1e-9
                ))
                importance = node.get("importance") or 0.5
                score = 0.7 * similarity + 0.3 * importance
                scored.append((node, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [n for n, _ in scored[:max_nodes]]

        except Exception as e:
            logger.warning("Adaptive context filtering failed, falling back to capped context: %s", e)
            return None

    # Types that are always kept regardless of relevance score
    _ALWAYS_KEEP_TYPES = {"person", "preference"}

    def _filter_identity(
        self, identity_nodes: list[dict], task_hint: str, max_nodes: int = 10
    ) -> list[dict]:
        """Filter identity nodes by relevance to task_hint.

        Always keeps person and preference nodes (the agent needs to know
        who it's talking to and how they want to communicate). Other
        identity facts are scored against the hint and only the top-N
        are included.
        """
        always: list[dict] = []
        scorable: list[dict] = []
        for n in identity_nodes:
            if n.get("type") in self._ALWAYS_KEEP_TYPES:
                always.append(n)
            else:
                scorable.append(n)

        if not scorable:
            return always

        # Score the rest against the hint
        remaining_slots = max(max_nodes - len(always), 0)
        if remaining_slots == 0:
            return always

        scored = self._filter_by_hint(scorable, task_hint, max_nodes=remaining_slots)
        if scored is None:
            # Scoring failed — return all rather than lose identity
            return identity_nodes

        return always + scored
