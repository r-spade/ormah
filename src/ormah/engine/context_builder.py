"""Builds whisper context for involuntary recall injection."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import numpy as np

from ormah.engine.maintenance_signal import MAINTENANCE_DUE_SIGNAL
from ormah.index.graph import GraphIndex
from ormah.text.tokens import distinctive_tokens

logger = logging.getLogger(__name__)

_WHISPER_FRAMING = (
    "# Ormah whispers\n"
    "The most relevant memories are shown in full. The rest are titles only. "
    "If any memory looks relevant or interesting, use recall with its node ID "
    "to get the full content and related memories."
)


_REVIEW_FRAMING = (
    "\n\n## Ormah: one thing to review when you get a chance\n"
    "In a recent session, the user was working on:\n"
    "\"{prompt_snippet}\"\n\n"
    "Ormah held back this memory because it wasn't confident it was relevant:\n"
    "\"{title}\" — {content}  (node: {node_id})\n\n"
    "When you can judge it, call submit_feedback(node_id=\"{node_id}\", signal=1 for yes, "
    "signal=-1 for no, source=\"implicit\"). Skip if it's not a good moment — "
    "this won't be surfaced again for 14 days."
)

def _truncate_at_word_boundary(text: str, max_len: int = 300) -> str:
    """Return *text* truncated to *max_len* characters at a word boundary."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space == -1:
        return truncated + "…"
    return truncated[:last_space] + "…"


def _prompt_log_snippet(text: str, max_len: int = 80) -> str:
    """Compact single-line prompt snippet for diagnostics."""
    compact = " ".join(text.split())
    return _truncate_at_word_boundary(compact, max_len=max_len)


def _topic_tokens(text: str) -> set[str]:
    """Extract meaningful topical tokens from text."""
    return distinctive_tokens(text)
def _has_topical_overlap(prompt_tokens: set[str], node: dict) -> bool:
    """Return True when prompt tokens overlap node title/content tokens."""
    if not prompt_tokens:
        return False
    node_text = " ".join(
        part for part in (node.get("title"), node.get("content")) if isinstance(part, str)
    )
    return bool(prompt_tokens & _topic_tokens(node_text))


def _find_review_candidate(conn, threshold: float) -> dict | None:
    """Find a gated-out whisper candidate eligible for session-start review.

    Applies three Python-side filters after SQL eligibility query:
    1. No strong affinity signal (cosine sim < threshold against existing affinity rows)
    2. Not recently surfaced (no review_log row within 14 days)
    3. Not exhausted (fewer than 3 unanswered review_log rows)
    """
    try:
        rows = conn.execute(
            """
            WITH ranked AS (
              SELECT
                wl.node_id, wl.score, wl.session_id, wl.space, wl.prompt_text,
                wl.prompt_vec,
                n.title, n.content,
                ROW_NUMBER() OVER (PARTITION BY wl.node_id ORDER BY wl.score DESC) AS rn
              FROM whisper_log wl
              JOIN nodes n ON n.id = wl.node_id
              WHERE wl.was_injected = 0
                AND wl.logged_at > datetime('now', '-7 days')
                AND NOT EXISTS (
                  SELECT 1 FROM whisper_log wl2
                  WHERE wl2.node_id = wl.node_id
                    AND wl2.was_injected = 1
                    AND wl2.logged_at > datetime('now', '-7 days')
                )
            )
            SELECT node_id, score, session_id, space, prompt_text, prompt_vec, title, content
            FROM ranked
            WHERE rn = 1
            ORDER BY score DESC
            LIMIT 20
            """
        ).fetchall()
    except Exception as e:
        logger.warning("_find_review_candidate SQL failed: %s", e)
        return None

    for row in rows:
        node_id = row["node_id"]
        candidate_prompt_vec_blob = row["prompt_vec"]

        # Step 2a: No strong affinity signal
        try:
            affinity_rows = conn.execute(
                "SELECT prompt_vec FROM affinity WHERE node_id = ?", (node_id,)
            ).fetchall()
            if affinity_rows and candidate_prompt_vec_blob:
                candidate_vec = np.frombuffer(candidate_prompt_vec_blob, dtype=np.float32)
                candidate_norm = float(np.linalg.norm(candidate_vec))
                skip = False
                if candidate_norm > 0:
                    for arow in affinity_rows:
                        aff_vec = np.frombuffer(arow["prompt_vec"], dtype=np.float32)
                        aff_norm = float(np.linalg.norm(aff_vec))
                        if aff_norm > 0:
                            sim = float(np.dot(candidate_vec, aff_vec) / (candidate_norm * aff_norm))
                            if sim >= threshold:
                                skip = True
                                break
                if skip:
                    continue
        except Exception as e:
            logger.warning("Affinity check failed for node %s: %s", node_id, e)

        # Step 2b: Not recently surfaced
        try:
            recently = conn.execute(
                "SELECT 1 FROM review_log WHERE node_id = ? AND surfaced_at > datetime('now', '-14 days') LIMIT 1",
                (node_id,),
            ).fetchone()
            if recently:
                continue
        except Exception as e:
            logger.warning("review_log recency check failed for node %s: %s", node_id, e)
            continue

        # Step 2c: Not exhausted
        try:
            unanswered_count = conn.execute(
                "SELECT COUNT(*) FROM review_log WHERE node_id = ? AND answered = 0",
                (node_id,),
            ).fetchone()[0]
            if unanswered_count >= 3:
                continue
        except Exception as e:
            logger.warning("review_log exhaustion check failed for node %s: %s", node_id, e)
            continue

        return dict(row)

    return None


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

    def build_whisper_context(
        self,
        prompt: str,
        space: str | None = None,
        user_node_id: str | None = None,
        max_nodes: int = 8,
        min_score: float = 0.45,
        full_content_count: int = 2,
        reranker_enabled: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_min_score: float = 0.0,
        reranker_blend_alpha: float = 0.4,
        reranker_max_doc_chars: int = 512,
        recent_prompts: list[str] | None = None,
        topic_shift_enabled: bool = False,
        topic_shift_threshold: float = 0.75,
        injection_gate: float = 0.55,
        session_id: str | None = None,
        _return_debug: bool = False,
    ) -> str | tuple[str, list[str]]:
        """Build compact whisper context for involuntary recall injection.

        Key differences from build_core_context:
        - Hard min-score threshold: results below min_score are dropped.
        - Returns empty string on failure instead of full dump.
        """
        prompt_snippet = _prompt_log_snippet(prompt)
        _injected_ids: list[str] = []

        if not prompt.strip():
            logger.info("Whisper diagnostics: empty prompt -> skip")
            if _return_debug:
                return "", _injected_ids
            return ""

        # Short prompts (≤2 alphanumeric chars) are navigational ("y", "ok",
        # "...", "---") — skip search
        stripped = re.sub(r'[^a-zA-Z0-9]', '', prompt.strip())
        if len(stripped) <= 2:
            logger.info(
                "Whisper diagnostics: prompt=%r short_prompt -> skip",
                prompt_snippet,
            )
            if _return_debug:
                return "", _injected_ids
            return ""

        if not self.engine:
            logger.info(
                "Whisper diagnostics: prompt=%r no_engine -> skip",
                prompt_snippet,
            )
            if _return_debug:
                return "", _injected_ids
            return ""

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
            logger.info(
                "Whisper diagnostics: prompt=%r conversational_intent -> skip",
                prompt_snippet,
            )
            if _return_debug:
                return "", _injected_ids
            return ""

        follow_up_mode = bool(recent_prompts) and (
            intent is not None and "continuation" in intent.categories
        )

        # Topic-shift detection: skip injection when topic hasn't changed
        if (
            topic_shift_enabled
            and recent_prompts
            and len(recent_prompts) >= 1
            and not follow_up_mode
        ):
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
                            logger.info(
                                "Whisper diagnostics: prompt=%r topic_shift_skip similarity=%.3f threshold=%.3f",
                                prompt_snippet,
                                similarity,
                                topic_shift_threshold,
                            )
                            if _return_debug:
                                return "", _injected_ids
                            return ""  # same topic, skip injection
            except Exception as e:
                logger.warning("Topic-shift detection failed, proceeding with whisper: %s", e)

        # identity-only → skip general search, use existing identity path below
        identity_only = intent is not None and intent.categories == ["identity"]
        identity_linked_ids: set[str] = set()
        if user_node_id:
            try:
                identity_linked_ids = {
                    node["id"]
                    for node in self.graph.get_neighbors(
                        user_node_id,
                        depth=1,
                        edge_types=["defines"],
                    )
                    if node.get("id")
                }
            except Exception as e:
                logger.warning("Failed to load identity-linked nodes: %s", e)

        # Compute prompt_vec early — needed for affinity boost and whisper_log
        prompt_vec: np.ndarray | None = None
        try:
            hybrid_search = self.engine._get_hybrid_search()
            if hybrid_search is not None:
                prompt_vec = hybrid_search.encoder.encode(prompt)
        except Exception as e:
            logger.warning("Failed to compute prompt_vec for affinity: %s", e)

        # Build context-enhanced search query from recent prompts
        search_query = prompt
        if recent_prompts and follow_up_mode:
            # Use recent context only for underspecified follow-ups so we
            # improve ambiguous prompts without polluting explicit ones.
            context_parts = recent_prompts[-2:] + [prompt]
            search_query = " ".join(context_parts)

        # Build search kwargs, merging any intent-derived params
        search_kwargs: dict = {
            "query": search_query,
            "limit": max_nodes,
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
            if _return_debug:
                return "", _injected_ids
            return ""

        initial_candidate_count = len(search_results)
        reranker_applied = False
        reranker_before_count = 0
        reranker_after_count = 0

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
        if has_temporal:
            effective_min_score = min(min_score, 0.30)
        else:
            effective_min_score = min_score
        search_results = [
            r for r in search_results
            if r.get("score", 0) >= effective_min_score or r.get("source") == "temporal"
        ]
        post_min_score_count = len(search_results)
        # Cross-encoder reranking — always pass min_score=0.0 so affinity boost
        # can rescue candidates before any floor is applied (spec: reranker_min_score
        # is now applied as a post-boost floor, not inside rerank())
        if reranker_enabled and search_results and not identity_only:
            try:
                from ormah.embeddings.reranker import rerank

                reranker_applied = True
                reranker_before_count = len(search_results)
                search_results = rerank(
                    query=prompt,
                    candidates=search_results,
                    model_name=reranker_model,
                    min_score=0.0,
                    blend_alpha=reranker_blend_alpha,
                    max_doc_chars=reranker_max_doc_chars,
                )
                reranker_after_count = len(search_results)

            except Exception as e:
                logger.warning("Whisper reranker failed, using embedding scores: %s", e)

        # Affinity boost (adaptive feedback loop): apply personalised score
        # adjustments based on historical signals before any floor filtering.
        # pre_gate_candidates captures the full set after boost (used by
        # exploration slot and whisper_log logging below).
        pre_gate_candidates: list[dict] = []
        if not has_temporal and reranker_enabled and search_results and prompt_vec is not None:
            try:
                from ormah.engine.affinity import batch_fetch_affinity, compute_affinity_boost

                node_ids = [r["node"]["id"] for r in search_results]
                affinity_rows_map = batch_fetch_affinity(self.graph.conn, node_ids)
                boosted = []
                for r in search_results:
                    nid = r["node"]["id"]
                    rows = affinity_rows_map.get(nid, [])
                    boost = compute_affinity_boost(prompt_vec, nid, rows, self.engine.settings)
                    boosted.append({**r, "score": r["score"] + boost, "_pre_boost_score": r["score"]})
                # Apply 0.40 floor AFTER boost (spec: reranker_min_score is now a post-boost floor).
                # Use the reranker_min_score parameter (passed from engine.settings); fall back to 0.40.
                effective_floor = reranker_min_score if reranker_min_score > 0.0 else 0.40
                pre_gate_candidates = [r for r in boosted if r["score"] >= effective_floor]
                search_results = pre_gate_candidates
            except Exception as e:
                logger.warning("Affinity boost failed, using unmodified scores: %s", e)

        if search_results:
            if identity_only:
                global_results = [
                    r for r in search_results
                    if r["node"].get("space") in (None, "null")
                ]
                if global_results:
                    search_results = global_results
                    pre_gate_candidates = [
                        r for r in pre_gate_candidates
                        if r["node"].get("space") in (None, "null")
                    ] or pre_gate_candidates

            prompt_tokens = _topic_tokens(prompt)
            if space:
                prompt_tokens.discard(space.lower())
            overlapping_ids = {
                r["node"]["id"]
                for r in search_results
                if _has_topical_overlap(prompt_tokens, r["node"])
            }
            if overlapping_ids:
                search_results = [
                    r for r in search_results
                    if (
                        r["node"]["id"] in overlapping_ids
                        or r["node"]["id"] in identity_linked_ids
                        or (identity_only and r["node"].get("space") in (None, "null"))
                    )
                ]
                if pre_gate_candidates:
                    pre_gate_candidates = [
                        r for r in pre_gate_candidates
                        if (
                            r["node"]["id"] in overlapping_ids
                            or r["node"]["id"] in identity_linked_ids
                            or (identity_only and r["node"].get("space") in (None, "null"))
                        )
                    ]

        # Injection gate: require at least one result with a strong enough
        # blended score to justify injection.  Temporal queries are exempt
        # (they rely on time filtering, not semantic relevance).
        if not has_temporal and search_results:
            max_blended = max(r.get("score", 0.0) for r in search_results)
            if max_blended < injection_gate:
                logger.info(
                    "Whisper diagnostics: prompt=%r gate_reject max_score=%.3f gate=%.3f",
                    prompt_snippet,
                    max_blended,
                    injection_gate,
                )
                search_results = []
            else:
                # Score-floor: only keep results that individually clear the
                # injection gate.  Weak queries naturally get fewer results
                # instead of padding to max_nodes with marginal matches.
                search_results = [r for r in search_results if r.get("score", 0) >= injection_gate]

        # Exploration slot: inject one unconfirmed gated-out candidate to
        # surface false negatives and collect affinity signal for them.
        # CE gate: skip candidates the cross-encoder strongly rejected
        # (ce < -8 means "definitely not relevant") to prevent noise injection.
        if (not has_temporal
                and getattr(self.engine.settings, "whisper_exploration_enabled", True)
                and prompt_vec is not None
                and pre_gate_candidates):
            try:
                from ormah.engine.affinity import batch_fetch_affinity

                injected_ids = {r["node"]["id"] for r in search_results}
                # Gated-out candidates that cleared the 0.40 floor but not the gate
                gated_out = [
                    r for r in pre_gate_candidates
                    if r["node"]["id"] not in injected_ids
                ]
                if gated_out:
                    explore_node_ids = [r["node"]["id"] for r in gated_out]
                    affinity_map = batch_fetch_affinity(self.graph.conn, explore_node_ids)
                    explore_threshold = getattr(
                        self.engine.settings, "affinity_similarity_threshold", 0.70
                    )
                    for candidate in sorted(gated_out, key=lambda r: r["score"], reverse=True):
                        # CE gate: don't explore candidates the CE strongly rejected
                        ce_score = candidate.get("cross_encoder_score")
                        if ce_score is not None and ce_score < -8.0:
                            continue
                        nid = candidate["node"]["id"]
                        rows = affinity_map.get(nid, [])
                        # Only explore nodes with no existing affinity signal for similar prompts
                        has_signal = False
                        for arow in rows:
                            row_vec = np.frombuffer(arow["prompt_vec"], dtype=np.float32)
                            row_norm = float(np.linalg.norm(row_vec))
                            prompt_norm = float(np.linalg.norm(prompt_vec))
                            if row_norm > 0 and prompt_norm > 0:
                                sim = float(np.dot(prompt_vec, row_vec) / (prompt_norm * row_norm))
                                if sim >= explore_threshold:
                                    has_signal = True
                                    break
                        if not has_signal:
                            search_results.append(candidate)
                            break  # one exploration slot only
            except Exception as e:
                logger.warning("Exploration slot failed: %s", e)

        # Temporal queries: re-sort by recency (most recent first).
        # Semantic scores already filtered noise via the 0.45 threshold,
        # but users expect chronological ordering for "what did we do today".
        if has_temporal and search_results:
            search_results.sort(
                key=lambda r: r["node"].get("created") or "",
                reverse=True,
            )

        # Cap to max_nodes (already ordered by relevance score, or by recency for temporal queries)
        search_results = search_results[:max_nodes]
        final_candidate_count = len(search_results)
        _injected_ids = [r["node"]["id"] for r in search_results]

        # Build flat ranked list — top full_content_count get full content,
        # rest get title + type + node ID only.
        lines = []
        for i, r in enumerate(search_results):
            node = r["node"]
            node_id = node.get("id", "")
            short_id = node_id[:8] if node_id else ""
            content_preview = node.get("content", "")
            title = node.get("title") or (content_preview[:60].strip() + ("…" if len(content_preview) > 60 else ""))
            node_type = node.get("type", "fact")
            id_suffix = f" (id: {short_id})" if short_id else ""

            lines.append(f"- **[{node_type}]** {title}{id_suffix}")

            if i < full_content_count:
                content = node.get("content", "").strip()
                if content and content != title:
                    lines.append(f"  {content}")

            lines.append("")

        body = "\n".join(lines).rstrip()

        # Write whisper_log: one row per non-temporal candidate with boosted_score >= 0.40.
        # Uses pre_gate_candidates (all above the 0.40 floor) so gated-out candidates
        # are also logged. was_injected reflects the final post-gate, post-exploration decision.
        if session_id and prompt_vec is not None and self.engine is not None:
            try:
                import hashlib

                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                now_iso = datetime.now(timezone.utc).isoformat()
                vec_blob = prompt_vec.astype(np.float32).tobytes()
                # Use pre_gate_candidates when available (after affinity boost),
                # otherwise fall back to the current search_results
                candidates_to_log = pre_gate_candidates if pre_gate_candidates else search_results
                injected_ids = {r["node"]["id"] for r in search_results}
                with self.engine.db.transaction() as conn:
                    for r in candidates_to_log:
                        if r.get("source") == "temporal":
                            continue
                        boosted_score = r["score"]
                        if boosted_score < 0.40:
                            continue  # below noise floor, skip
                        pre_boost_score = r.get("_pre_boost_score", r["score"])
                        was_injected = 1 if r["node"]["id"] in injected_ids else 0
                        conn.execute(
                            "INSERT INTO whisper_log "
                            "(session_id, space, prompt_hash, prompt_text, prompt_vec, "
                            "node_id, score, was_injected, logged_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (session_id, space, prompt_hash, prompt, vec_blob,
                             r["node"]["id"], pre_boost_score, was_injected, now_iso),
                        )
            except Exception as e:
                logger.warning("whisper_log write failed: %s", e)

        if not body:
            result = ""
        else:
            result = _WHISPER_FRAMING + "\n\n" + body

        logger.info(
            "Whisper diagnostics: prompt=%r intent=%s identity_only=%s temporal=%s "
            "candidates=%d post_min_score=%d reranker_enabled=%s reranker_applied=%s "
            "reranker_before=%d reranker_after=%d final=%d injected=%s",
            prompt_snippet,
            intent.categories if intent is not None else None,
            identity_only,
            has_temporal,
            initial_candidate_count,
            post_min_score_count,
            reranker_enabled,
            reranker_applied,
            reranker_before_count,
            reranker_after_count,
            final_candidate_count,
            bool(result),
        )

        # Maintenance due signal: fires once per interval regardless of node creation rate.
        # Self-limiting: apply_maintenance_results records last_maintenance_run, silencing
        # the signal for claude_maintenance_interval_hours.
        if self.engine is not None:
            settings = getattr(self.engine, "settings", None)
            if settings and getattr(settings, "claude_maintenance_enabled", False):
                interval_hours = getattr(settings, "claude_maintenance_interval_hours", 24)
                try:
                    row = self.graph.conn.execute(
                        "SELECT value FROM meta WHERE key = 'last_maintenance_run'"
                    ).fetchone()
                    last_run = row[0] if row else None
                    due = True
                    if last_run:
                        parsed_last_run = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
                        if parsed_last_run.tzinfo is None:
                            parsed_last_run = parsed_last_run.replace(tzinfo=timezone.utc)
                        elapsed = datetime.now(timezone.utc) - parsed_last_run.astimezone(timezone.utc)
                        due = elapsed.total_seconds() > interval_hours * 3600
                    if due:
                        result = (
                            f"{result}\n{MAINTENANCE_DUE_SIGNAL}"
                            if result
                            else MAINTENANCE_DUE_SIGNAL
                        )
                except Exception as e:
                    logger.warning("Failed to compute maintenance_due: %s", e)

        # First-message review: surface a gated-out whisper candidate for feedback.
        # recent_prompts is None only on the first message of a session (buffer just created).
        if recent_prompts is None and self.engine is not None:
            settings = getattr(self.engine, "settings", None)
            try:
                threshold = getattr(settings, "affinity_similarity_threshold", 0.70) if settings else 0.70
                candidate = _find_review_candidate(self.graph.conn, threshold)
                if candidate:
                    current_session_id = session_id or ""
                    with self.engine.db.transaction() as conn:
                        conn.execute(
                            "INSERT INTO review_log (node_id, session_id, surfaced_at) VALUES (?, ?, datetime('now'))",
                            (candidate["node_id"], current_session_id),
                        )
                    prompt_snippet = _truncate_at_word_boundary(
                        candidate["prompt_text"] or "", max_len=300
                    )
                    space_label = candidate["space"] or "global"
                    review_block = _REVIEW_FRAMING.format(
                        space=space_label,
                        prompt_snippet=prompt_snippet,
                        title=candidate["title"],
                        content=candidate["content"],
                        node_id=candidate["node_id"],
                    )
                    result = result + review_block
            except Exception as e:
                logger.warning("Review mechanism failed: %s", e)

        if _return_debug:
            return result, _injected_ids
        return result
