"""Builds whisper context for involuntary recall injection."""

from __future__ import annotations

import hashlib
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
        # Ellipsis counts against the budget: never exceed max_len.
        return truncated[: max_len - 1] + "…"
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
                AND wl.decision_stage IN ('injection_gate', 'candidate_cap', 'legacy')
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


def _gate_score(r: dict) -> float:
    """Absolute relevance signal for gating decisions.

    Gates answer "is anything here relevant?" and need a score whose meaning
    does not change per query. The blended `score` is rank-relative (RRF is
    min-max normalized per query, so any query's best candidate scores ~1.0)
    — an absolute threshold on it cannot reject a weak query's least-bad
    match. Prefer the cross-encoder's rescaled score; fall back to raw cosine
    when the reranker didn't run. The affinity delta is included so learned
    feedback can still lift a candidate over the gate. The blended `score`
    remains the ordering key.

    The raw absolute signals (ce_absolute, raw_cosine) are pure relevance and
    carry none of the *suppression* factors the pre-contract blended gate
    applied: cross-space demotion (_apply_space_scores) and the confidence
    factor. Both are folded back in here so a wrong-project or low-confidence
    memory must clear a higher bar — they are <= 1.0, so they can only push a
    candidate below the gate, never lift noise over it. The blended fallback
    already contains both, so it is returned unscaled.
    """
    affinity = r.get("_affinity_boost", 0.0)
    ce = r.get("ce_absolute")
    cos = r.get("raw_cosine")
    if ce is None and cos is None:
        # Legacy/FTS-only/spread-activation results carry neither absolute
        # signal; fall back to the blended score (which already contains the
        # suppression factors and any affinity boost) so behavior degrades to
        # the pre-contract gate rather than rejecting.
        return r.get("score", 0.0)
    node = r.get("node", {})
    confidence = node.get("confidence")
    confidence_factor = 0.4 + 0.6 * (1.0 if confidence is None else confidence)
    space_factor = r.get("_space_factor", 1.0)
    signal = ce if ce is not None else cos
    return signal * confidence_factor * space_factor + affinity


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

    def _topic_was_served(
        self,
        session_id: str | None,
        prompt_vec: np.ndarray,
        threshold: float,
    ) -> bool:
        """True when this session already had an injection on a similar topic.

        Reads whisper_log (session_id + prompt_vec + was_injected) instead of
        holding in-process state, so it works on every entry path and across
        restarts. Without a session_id there is no history to consult —
        return True to preserve the plain topic-shift skip behavior.
        """
        if not session_id:
            return True
        try:
            rows = self.graph.conn.execute(
                "SELECT DISTINCT prompt_vec FROM whisper_log "
                "WHERE session_id = ? AND was_injected = 1",
                (session_id,),
            ).fetchall()
            norm_current = float(np.linalg.norm(prompt_vec))
            if norm_current == 0:
                return True
            for row in rows:
                served_vec = np.frombuffer(row["prompt_vec"], dtype=np.float32)
                norm_served = float(np.linalg.norm(served_vec))
                if norm_served == 0:
                    continue
                sim = float(np.dot(prompt_vec, served_vec) / (norm_current * norm_served))
                if sim >= threshold:
                    return True
            return False
        except Exception as e:
            logger.warning("_topic_was_served check failed: %s", e)
            return True  # fail toward the historical skip behavior

    def _log_decision(
        self,
        *,
        session_id: str | None,
        space: str | None,
        prompt: str,
        intent,
        outcome: str,
        candidate_count: int = 0,
        injected_count: int = 0,
        max_gate_score: float | None = None,
    ) -> None:
        """Write one whisper_decisions row per whisper call — including silence.

        whisper_log records candidates; this records the per-prompt outcome so
        silence rate has a denominator. Never raises: instrumentation must not
        break whisper.
        """
        if not self.engine:
            return
        try:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            intent_str = ",".join(intent.categories) if intent is not None else None
            with self.engine.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO whisper_decisions "
                    "(session_id, space, prompt_hash, intent, outcome, "
                    "candidate_count, injected_count, max_gate_score, logged_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, space, prompt_hash, intent_str, outcome,
                     candidate_count, injected_count, max_gate_score,
                     datetime.now(timezone.utc).isoformat()),
                )
        except Exception as e:
            logger.warning("whisper_decisions write failed: %s", e)

    def build_whisper_context(
        self,
        prompt: str,
        space: str | None = None,
        user_node_id: str | None = None,
        max_nodes: int = 8,
        min_score: float = 0.45,
        full_content_count: int = 2,
        candidate_pool_multiplier: int = 5,
        injected_content_max_chars: int = 600,
        reranker_enabled: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_min_score: float = 0.0,
        reranker_blend_alpha: float = 0.4,
        reranker_max_doc_chars: int = 512,
        recent_prompts: list[str] | None = None,
        topic_shift_enabled: bool = False,
        topic_shift_threshold: float = 0.75,
        injection_gate: float = 0.55,
        no_overlap_ce_floor: float = 0.45,
        no_overlap_cosine_floor: float = 0.70,
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
            self._log_decision(
                session_id=session_id, space=space, prompt=prompt,
                intent=None, outcome="silent_short",
            )
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
            self._log_decision(
                session_id=session_id, space=space, prompt=prompt,
                intent=None, outcome="silent_short",
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
            self._log_decision(
                session_id=session_id, space=space, prompt=prompt,
                intent=intent, outcome="silent_conversational",
            )
            if _return_debug:
                return "", _injected_ids
            return ""

        follow_up_mode = bool(recent_prompts) and (
            intent is not None and "continuation" in intent.categories
        )

        # Reuse the prompt vector PromptClassifier already computed (it encodes
        # the same raw prompt string with the same encoder) instead of
        # encoding again — this vector is then reused by topic-shift
        # detection, the affinity boost, and whisper_log. Only falls back to
        # a fresh encode when there's no classifier or it hit the degenerate
        # zero-vector case.
        prompt_vec: np.ndarray | None = intent.prompt_vec if intent is not None else None
        if prompt_vec is None:
            try:
                hybrid_search = self.engine._get_hybrid_search()
                if hybrid_search is not None:
                    prompt_vec = hybrid_search.encoder.encode(prompt)
            except Exception as e:
                logger.warning("Failed to compute prompt_vec: %s", e)

        # Topic-shift detection: skip injection when topic hasn't changed
        if (
            topic_shift_enabled
            and recent_prompts
            and len(recent_prompts) >= 1
            and not follow_up_mode
            and prompt_vec is not None
        ):
            try:
                hybrid_search = self.engine._get_hybrid_search()
                if hybrid_search is not None:
                    encoder = hybrid_search.encoder
                    current_vec = prompt_vec
                    recent_vecs = encoder.encode_batch(recent_prompts[-3:])
                    centroid = np.mean(recent_vecs, axis=0)
                    norm_current = np.linalg.norm(current_vec)
                    norm_centroid = np.linalg.norm(centroid)
                    if norm_current > 0 and norm_centroid > 0:
                        similarity = float(
                            np.dot(current_vec, centroid)
                            / (norm_current * norm_centroid)
                        )
                        # Suppress only topics that were actually SERVED: if
                        # every earlier prompt on this topic produced silence
                        # (gate reject, conversational, …), skipping again
                        # would starve the topic for the whole session — the
                        # first miss must not condemn the conversation.
                        if similarity > topic_shift_threshold and self._topic_was_served(
                            session_id, current_vec, topic_shift_threshold
                        ):
                            logger.info(
                                "Whisper diagnostics: prompt=%r topic_shift_skip similarity=%.3f threshold=%.3f",
                                prompt_snippet,
                                similarity,
                                topic_shift_threshold,
                            )
                            self._log_decision(
                                session_id=session_id, space=space, prompt=prompt,
                                intent=intent, outcome="silent_topic_shift",
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

        # Build context-enhanced search query from recent prompts
        search_query = prompt
        if recent_prompts and follow_up_mode:
            # Use recent context only for underspecified follow-ups so we
            # improve ambiguous prompts without polluting explicit ones.
            context_parts = recent_prompts[-2:] + [prompt]
            search_query = " ".join(context_parts)

        # Build search kwargs, merging any intent-derived params
        # Fetch a deep candidate pool so the reranker/gate can rescue memories
        # the bi-encoder under-ranked; the final injected set is capped at
        # max_nodes after gating.
        search_kwargs: dict = {
            "query": search_query,
            "limit": max_nodes * max(candidate_pool_multiplier, 1),
            "default_space": space,
            "tiers": ["core", "working"],
            "touch_access": False,
            # Whisper needs the raw pool — it applies its own floors and an
            # absolute-signal gate; the deliberate-recall floor would drop
            # length-penalized long docs before the reranker can rescue them.
            "min_relevance": 0.0,
        }
        if intent is not None:
            # Extract search_query override before merging (it's not a
            # recall_search_structured kwarg — it overrides our local query).
            intent_search_query = intent.search_params.pop("search_query", None)
            search_kwargs.update(intent.search_params)
            if intent_search_query is not None:
                search_kwargs["query"] = intent_search_query

        # The effective query is what search actually ran on: the bare prompt,
        # the follow-up context-enhanced query, or an intent override (e.g. a
        # temporal-stripped query). The reranker's ce_absolute drives the
        # injection gate, so it must judge candidates against this same query —
        # scoring the bare prompt would gate-reject memories that only make
        # sense with the session context ("and the second one?").
        effective_query = search_kwargs["query"]

        # Always run search — even for identity-only queries, search finds
        # location/work/study nodes that graph neighbors alone miss.
        try:
            search_results = self.engine.recall_search_structured(**search_kwargs)
        except Exception as e:
            logger.warning("Whisper search failed: %s", e)
            self._log_decision(
                session_id=session_id, space=space, prompt=prompt,
                intent=intent, outcome="silent_error",
            )
            if _return_debug:
                return "", _injected_ids
            return ""

        candidate_trace: dict[str, dict] = {}

        def _record_candidate_versions(results: list[dict]) -> None:
            for rank, result in enumerate(results, start=1):
                if result.get("source") == "temporal":
                    continue
                node_id = result.get("node", {}).get("id")
                if not node_id:
                    continue
                trace = candidate_trace.setdefault(
                    node_id,
                    {
                        "retrieval_score": result.get("score", 0.0),
                        "retrieval_rank": rank,
                        "decision_stage": "candidate",
                    },
                )
                trace["latest"] = result

        def _mark_removed(before: list[dict], after: list[dict], stage: str) -> None:
            after_ids = {r.get("node", {}).get("id") for r in after}
            for result in before:
                node_id = result.get("node", {}).get("id")
                trace = candidate_trace.get(node_id)
                if (
                    trace is not None
                    and node_id not in after_ids
                    and trace["decision_stage"] == "candidate"
                ):
                    trace["decision_stage"] = stage

        _record_candidate_versions(search_results)
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
        # A candidate reaches the reranker if EITHER signal clears the floor:
        # the blended score is rank-relative and the length penalty can bury a
        # long-but-relevant node's blend while its raw cosine stays high (the
        # strongest match can otherwise be the lowest-blended candidate).
        # The cross-encoder + absolute gate downstream handle any noise this
        # lets through.
        before_min_score = search_results
        search_results = [
            r for r in search_results
            if r.get("score", 0) >= effective_min_score
            or r.get("raw_cosine", 0.0) >= effective_min_score
            or r.get("source") == "temporal"
        ]
        _mark_removed(before_min_score, search_results, "pre_rerank_floor")
        post_min_score_count = len(search_results)
        # Cross-encoder reranking — always pass min_score=0.0 so affinity boost
        # can rescue candidates before any floor is applied (spec: reranker_min_score
        # is now applied as a post-boost floor, not inside rerank())
        if reranker_enabled and search_results and not identity_only:
            try:
                from ormah.embeddings.reranker import rerank

                reranker_applied = True
                reranker_before_count = len(search_results)
                before_rerank = search_results
                search_results = rerank(
                    query=effective_query,
                    candidates=search_results,
                    model_name=reranker_model,
                    min_score=0.0,
                    blend_alpha=reranker_blend_alpha,
                    max_doc_chars=reranker_max_doc_chars,
                )
                _record_candidate_versions(search_results)
                _mark_removed(before_rerank, search_results, "reranker_floor")
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
                    boosted.append({
                        **r,
                        "score": r["score"] + boost,
                        "_pre_boost_score": r["score"],
                        # Tagged separately so the (absolute-signal) gate can
                        # include learned feedback without inheriting the
                        # rank-relative blended score.
                        "_affinity_boost": boost,
                    })
                # Apply 0.40 floor AFTER boost (spec: reranker_min_score is now a post-boost floor).
                # Use the reranker_min_score parameter (passed from engine.settings); fall back to 0.40.
                effective_floor = reranker_min_score if reranker_min_score > 0.0 else 0.40
                _record_candidate_versions(boosted)
                pre_gate_candidates = [r for r in boosted if r["score"] >= effective_floor]
                _mark_removed(boosted, pre_gate_candidates, "post_rerank_floor")
                search_results = pre_gate_candidates
            except Exception as e:
                logger.warning("Affinity boost failed, using unmodified scores: %s", e)

        if search_results:
            if identity_only:
                before_identity_filter = search_results
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
                    _mark_removed(
                        before_identity_filter,
                        search_results,
                        "identity_space_filter",
                    )

            prompt_tokens = _topic_tokens(prompt)
            if space:
                prompt_tokens.discard(space.lower())
            overlapping_ids = {
                r["node"]["id"]
                for r in search_results
                if _has_topical_overlap(prompt_tokens, r["node"])
            }

            # Fail CLOSED: a candidate sharing no token with the prompt is
            # the maximally suspicious case (embedding false-friends), so it
            # needs a voucher — identity protection, or a strong absolute
            # relevance signal. Previously the filter was skipped entirely
            # when nothing overlapped, passing everything through in exactly
            # the situation it exists to catch.
            def _keep(r: dict) -> bool:
                nid = r["node"]["id"]
                if nid in overlapping_ids:
                    return True
                if nid in identity_linked_ids:
                    return True
                if identity_only and r["node"].get("space") in (None, "null"):
                    return True
                # Recency-vouched results: relevance comes from the time
                # filter, not semantics — never demand a semantic voucher.
                if r.get("source") == "temporal":
                    return True
                # Temporal and follow-up prompts are underspecified by design
                # ("what did we do today", "and the second one?") — the CE
                # judged them against the raw prompt, so its verdict is not a
                # fair voucher. Keep pre-contract behavior for them: pass
                # when nothing overlapped, drop when other candidates did.
                if has_temporal or follow_up_mode:
                    return not overlapping_ids
                ce = r.get("ce_absolute")
                if ce is not None:
                    return ce >= no_overlap_ce_floor
                cos = r.get("raw_cosine")
                if cos is not None:
                    return cos >= no_overlap_cosine_floor
                # Results without absolute signals (spread-activation
                # neighbors, legacy callers): preserve pre-contract behavior —
                # dropped when other candidates overlap, passed when nothing
                # overlapped.
                return not overlapping_ids

            before_topical_filter = search_results
            search_results = [r for r in search_results if _keep(r)]
            _mark_removed(before_topical_filter, search_results, "topical_filter")
            if pre_gate_candidates:
                pre_gate_candidates = [r for r in pre_gate_candidates if _keep(r)]

        # Injection gate: require at least one result with a strong enough
        # ABSOLUTE relevance signal to justify injection (see _gate_score —
        # the blended score is rank-relative and cannot reject a weak query's
        # least-bad match; it stays the ordering key only). Temporal queries
        # are exempt (they rely on time filtering, not semantic relevance).
        max_gate_score: float | None = None
        if search_results:
            max_gate_score = max(_gate_score(r) for r in search_results)
        if not has_temporal and search_results:
            before_injection_gate = search_results
            if max_gate_score < injection_gate:
                logger.info(
                    "Whisper diagnostics: prompt=%r gate_reject max_gate_score=%.3f gate=%.3f",
                    prompt_snippet,
                    max_gate_score,
                    injection_gate,
                )
                search_results = []
            else:
                # Score-floor: only keep results that individually clear the
                # injection gate.  Weak queries naturally get fewer results
                # instead of padding to max_nodes with marginal matches.
                search_results = [r for r in search_results if _gate_score(r) >= injection_gate]
            _mark_removed(before_injection_gate, search_results, "injection_gate")

        # Exploration slot: inject one unconfirmed gated-out candidate to
        # surface false negatives and collect affinity signal for them.
        # Piggybacks on real injections only (`search_results` non-empty):
        # when the gate decided on silence, silence stands — exploration must
        # never manufacture an injection from nothing.
        # CE gate: skip candidates the cross-encoder strongly rejected
        # (ce < -8 means "definitely not relevant") to prevent noise injection.
        if (not has_temporal
                and getattr(self.engine.settings, "whisper_exploration_enabled", True)
                and prompt_vec is not None
                and search_results
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
                            # Label it: the agent should weigh a deliberate
                            # long-shot differently from a confident whisper,
                            # and feedback on it stays honest.
                            search_results.append({**candidate, "_exploration": True})
                            candidate_trace[nid]["decision_stage"] = "candidate"
                            break  # one exploration slot only
            except Exception as e:
                logger.warning("Exploration slot failed: %s", e)

        # Temporal queries: re-sort by (space priority, recency).
        # Semantic scores already filtered noise via the 0.45 threshold,
        # but users expect chronological ordering for "what did we do today".
        # Space priority stays the primary key so a newer other-project memory
        # cannot outrank an older current-project one purely by recency — both
        # semantic hits and temporal supplements carry _space_factor from the
        # recall layer.
        if has_temporal and search_results:
            search_results.sort(
                key=lambda r: (r.get("_space_factor", 1.0), r["node"].get("created") or ""),
                reverse=True,
            )

        # Cap to max_nodes (already ordered by relevance score, or by recency for temporal queries)
        before_candidate_cap = search_results
        search_results = search_results[:max_nodes]
        _mark_removed(before_candidate_cap, search_results, "candidate_cap")
        final_candidate_count = len(search_results)
        _injected_ids = [r["node"]["id"] for r in search_results]
        for final_rank, result in enumerate(search_results, start=1):
            node_id = result["node"]["id"]
            trace = candidate_trace.get(node_id)
            if trace is None:
                continue
            trace["latest"] = result
            trace["final_rank"] = final_rank
            trace["decision_stage"] = (
                "exploration_injected" if result.get("_exploration") else "injected"
            )

        # Per-prompt outcome row (silence instrumentation): exactly one row
        # per whisper call so silence rate has a denominator.
        if final_candidate_count > 0:
            outcome = "injected"
        elif post_min_score_count == 0:
            outcome = "silent_no_candidates"
        else:
            outcome = "silent_gate"
        self._log_decision(
            session_id=session_id, space=space, prompt=prompt, intent=intent,
            outcome=outcome,
            candidate_count=initial_candidate_count,
            injected_count=final_candidate_count,
            max_gate_score=max_gate_score,
        )

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
            marker = "[exploring]" if r.get("_exploration") else f"[{node_type}]"

            lines.append(f"- **{marker}** {title}{id_suffix}")

            if i < full_content_count:
                content = node.get("content", "").strip()
                if content and content != title:
                    content = _truncate_at_word_boundary(
                        content, max_len=injected_content_max_chars
                    )
                    lines.append(f"  {content}")

            lines.append("")

        body = "\n".join(lines).rstrip()

        # Write one diagnostic row for every non-temporal retrieved candidate.
        # Absolute signals and the first destructive stage are retained so live
        # replays can distinguish retrieval misses from floor, topical, and gate
        # rejections. Existing feedback consumers still key off was_injected.
        if session_id and prompt_vec is not None and self.engine is not None:
            try:
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                now_iso = datetime.now(timezone.utc).isoformat()
                vec_blob = prompt_vec.astype(np.float32).tobytes()
                injected_ids = {r["node"]["id"] for r in search_results}
                with self.engine.db.transaction() as conn:
                    for node_id, trace in candidate_trace.items():
                        r = trace["latest"]
                        score = r.get("_pre_boost_score", r.get("score", 0.0))
                        was_injected = 1 if node_id in injected_ids else 0
                        conn.execute(
                            "INSERT INTO whisper_log "
                            "(session_id, space, prompt_hash, prompt_text, prompt_vec, "
                            "node_id, score, retrieval_score, raw_cosine, "
                            "cross_encoder_score, ce_absolute, gate_score, source, "
                            "retrieval_rank, final_rank, decision_stage, was_injected, logged_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (session_id, space, prompt_hash, prompt, vec_blob,
                             node_id, score, trace["retrieval_score"],
                             r.get("raw_cosine"), r.get("cross_encoder_score"),
                             r.get("ce_absolute"), _gate_score(r), r.get("source"),
                             trace["retrieval_rank"], trace.get("final_rank"),
                             trace["decision_stage"], was_injected, now_iso),
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
