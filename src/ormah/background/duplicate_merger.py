"""Detect near-duplicate memories and create merge proposals."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from ormah.background.memory_lock import serialized_memory_job

logger = logging.getLogger(__name__)

# Multi-signal weights
_W_EMBEDDING = 0.6
_W_TITLE = 0.2
_W_TOKEN = 0.2
_COMPOSITE_THRESHOLD = 0.60

_LLM_DUPLICATE_PROMPT = """\
You are deciding whether two memories in a knowledge graph are duplicates that should be merged into one. If merged, the resulting memory replaces both originals — one is kept (updated), one is deleted. All edges from the deleted node are remapped to the kept node. This is irreversible (though undoable), so be conservative.

Memory A:
- Title: {title_a}
- Type: {type_a}
- Content: {content_a}

Memory B:
- Title: {title_b}
- Type: {type_b}
- Content: {content_b}

## Are they duplicates?

**YES — merge** if they describe the same core fact, decision, or preference and keeping both would create redundancy. One might be more detailed than the other, or they might phrase the same information differently. The test: if an AI assistant retrieved both while helping the user, would it think "these are saying the same thing"?

**NO — keep separate** if ANY of these apply:
- Different aspects of the same topic ("chose SQLite" vs "SQLite schema has 4 tables" — same subject, different information)
- Different granularity levels that are BOTH useful (architecture overview + specific implementation detail)
- Same system at different points in time, where both versions are still informative
- One is a decision, the other is a consequence or implementation of that decision
- Same project but different components, features, or modules
- One has context or reasoning the other lacks, and merging would lose the narrative
- **An observation/evaluation and the decision it led to** — "evaluated X and found problems" vs "chose Y over X because of those problems" contain different information (the evaluation findings vs the final choice). Merging loses the evaluation detail.
- **A config/parameter at one value and a later change to a different value with reasoning** — "set threshold to 0.38" vs "lowered threshold to 0.35 because 0.38 broke temporal queries" are NOT duplicates. The second contains the reasoning for why the first value was wrong. Keeping both preserves the decision history.

**When in doubt, keep separate.** Two similar-but-distinct memories are better than one merged memory that lost nuance. The graph's "supports" edges already connect related memories — you don't need to merge them to show they're related.

## If merging (is_duplicate=true)

The merged content must be the UNION of both memories — every concrete detail preserved:
- Names, versions, paths, specific numbers, dates
- Reasoning and "why" explanations
- Rejected alternatives in decisions
- Write 2-5 clear sentences as if this is the single canonical memory on this topic
- The title must be specific and searchable (titles get 10x weight in full-text search)

Return JSON:
{{
  "is_duplicate": true or false,
  "merged_title": "specific searchable title, 5-12 words (only if is_duplicate=true)",
  "merged_content": "merged content preserving ALL unique details from both (only if is_duplicate=true)",
  "reason": "one sentence referencing the actual memory titles"
}}"""


def _title_similarity(title_a: str | None, title_b: str | None) -> float:
    """Levenshtein-ratio similarity between two titles. Returns 0.0 if either is None."""
    if not title_a or not title_b:
        return 0.0
    a, b = title_a.lower().strip(), title_b.lower().strip()
    if a == b:
        return 1.0
    # Levenshtein ratio without external deps
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0
    # Build DP table (space-optimized)
    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[len_b]
    return 1.0 - dist / max(len_a, len_b)


def _token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity on lowercased word sets."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _composite_score(embedding_sim: float, title_sim: float, token_sim: float) -> float:
    """Weighted composite duplicate score."""
    return _W_EMBEDDING * embedding_sim + _W_TITLE * title_sim + _W_TOKEN * token_sim


def _llm_check_duplicate(settings, node_row, other_row) -> dict | None:
    """Ask LLM whether two nodes are duplicates and get merged content.

    Returns parsed dict with keys is_duplicate, merged_title, merged_content,
    reason — or None if the LLM is unavailable or returns invalid output.
    """
    from ormah.background.llm_client import extract_json, llm_generate

    prompt = _LLM_DUPLICATE_PROMPT.format(
        title_a=node_row["title"] or "(untitled)",
        type_a=node_row["type"],
        content_a=node_row["content"][:2000],
        title_b=other_row["title"] or "(untitled)",
        type_b=other_row["type"],
        content_b=other_row["content"][:2000],
    )

    raw = llm_generate(settings, prompt, json_mode=True)
    if raw is None:
        return None

    try:
        result = json.loads(extract_json(raw))
        if "is_duplicate" not in result:
            return None
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for duplicate check")
        return None


def _find_merge_candidates(
    engine,
    limit: int = 8,
    *,
    max_seeds: int | None = None,
    delta: bool = False,
    respect_checked: bool = True,
):
    """Find node pairs that might be duplicates.

    ``delta=False`` (default — agent path): today's ``ORDER BY RANDOM()``
    selection, unchanged; returns a candidate list.

    ``delta=True`` (background run only, #81): seeds are nodes with ``seq``
    above the ``duplicate_check_watermark``, oldest-first, bounded by
    *max_seeds* (default: ``duplicate_check_max_nodes_per_run``). Vector
    neighbors are NOT age-filtered. Returns ``(candidates, drained_seeds)``;
    candidates carry ``seed_seq``. Only ``run_duplicate_detection`` advances
    the watermark. ``limit`` stays pair-denominated in both modes.

    ``respect_checked`` (default ``True``): skip pairs already present in
    ``auto_link_checked``. Background dedup (``run_duplicate_detection``)
    passes ``False`` — that table records auto_linker's LINK decisions
    (including "no link"), not dedup verdicts, and relies on its own
    seq-watermark for convergence instead.

    Does NOT call the LLM — just applies the same pre-filters as
    ``run_duplicate_detection`` (same type, composite score threshold).
    """
    drained_seeds: list[tuple[str, int]] = []
    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore, stored_or_encoded

        settings = engine.settings
        encoder = get_encoder(settings)
        vec_store = VectorStore(engine.db)
        user_node_id = getattr(engine, "user_node_id", None)

        if delta:
            from ormah.background.watermark import DUPLICATE_WATERMARK_KEY, get_watermark

            if max_seeds is None:
                max_seeds = settings.duplicate_check_max_nodes_per_run
            watermark = get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY)
            nodes = engine.db.conn.execute(
                "SELECT id, content, title, type, seq FROM nodes "
                "WHERE seq > ? ORDER BY seq ASC LIMIT ?",
                (watermark, max_seeds),
            ).fetchall()
        else:
            # Legacy selection — byte-for-byte today's query (agent path).
            nodes = engine.db.conn.execute(
                "SELECT id, content, title, type, seq FROM nodes ORDER BY RANDOM()"
            ).fetchall()

        checked: set[tuple[str, str]] = set()
        candidates: list[dict] = []
        barrier_hit = False

        for node in nodes:
            if len(candidates) >= limit:
                break  # pair budget hit before this seed: not drained
            if node["id"] == user_node_id:
                if not barrier_hit:
                    drained_seeds.append((node["id"], node["seq"]))
                continue

            text = f"{node['title'] or ''} {node['content']}".strip()
            if not text:
                if not barrier_hit:
                    drained_seeds.append((node["id"], node["seq"]))
                continue

            # DRAIN BARRIER (overview invariant, mirrors upstream
            # auto_linker.py): a seed with text but no persisted vector must
            # not let the cursor advance past it — its pairs would be
            # permanently skipped once the vector is backfilled. `continue`,
            # not `break`: later seeds are still PROCESSED (liveness, mirrors
            # auto_linker) but no further seed drains once the barrier is hit.
            if delta and vec_store.get(node["id"]) is None:
                if not barrier_hit:
                    logger.warning(
                        "duplicate delta stalled: node %s has no persisted vector (embedding "
                        "backfill pending?); cursor parked at seq %s until it embeds",
                        node["id"][:8], node["seq"],
                    )
                barrier_hit = True
                continue

            query_vec = stored_or_encoded(
                vec_store,
                encoder,
                node["id"],
                node["title"],
                node["content"],
                settings.embedding_max_content_chars,
            )
            similar = vec_store.search(query_vec, limit=6)

            for match in similar:
                if len(candidates) >= limit:
                    break
                if match["id"] == node["id"]:
                    continue
                if match["id"] == user_node_id:
                    continue

                pair = tuple(sorted([node["id"], match["id"]]))
                if pair in checked:
                    continue
                checked.add(pair)

                if respect_checked:
                    already_checked = engine.db.conn.execute(
                        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
                    ).fetchone()
                    if already_checked:
                        continue

                embedding_sim = match["similarity"]
                if embedding_sim < 0.25:
                    continue

                other = engine.db.conn.execute(
                    "SELECT id, type, title, content FROM nodes WHERE id = ?", (match["id"],)
                ).fetchone()
                if other is None or other["type"] != node["type"]:
                    continue

                title_sim = _title_similarity(node["title"], other["title"])
                other_text = f"{other['title'] or ''} {other['content']}".strip()
                token_sim = _token_overlap(text, other_text)
                score = _composite_score(embedding_sim, title_sim, token_sim)

                if score < _COMPOSITE_THRESHOLD:
                    continue

                def _nd(row) -> dict:
                    return {
                        "id": row["id"],
                        "title": row["title"] or "",
                        "type": row["type"] or "",
                        "space": "",
                        "content": (row["content"] or "")[:400],
                    }

                candidates.append({
                    "node_a": _nd(node),
                    "node_b": _nd(other),
                    "similarity": round(embedding_sim, 3),
                    "score": round(score, 3),
                    "embedding_sim": round(embedding_sim, 3),
                    "title_sim": round(title_sim, 3),
                    "token_sim": round(token_sim, 3),
                    "seed_seq": node["seq"],
                })

            if len(candidates) >= limit:
                break  # pair budget hit mid-seed: possibly partial, not drained
            if not barrier_hit:
                drained_seeds.append((node["id"], node["seq"]))

        if delta:
            return candidates, drained_seeds
        return candidates

    except Exception as e:
        logger.warning("_find_merge_candidates failed: %s", e)
        return ([], []) if delta else []


@serialized_memory_job
def run_duplicate_detection(engine) -> None:
    """Find near-duplicate nodes and create merge proposals.

    Uses a multi-signal approach: embedding similarity (primary),
    title similarity, and token overlap for candidate generation.
    LLM confirmation is mandatory — no merges happen without LLM
    saying ``is_duplicate: true``.
    """
    try:
        settings = engine.settings

        if not settings.llm_enabled:
            logger.debug("Duplicate detection skipped: LLM not enabled")
            return

        from ormah.background.watermark import (
            DUPLICATE_WATERMARK_KEY, get_watermark, set_watermark,
        )

        candidates, drained_seeds = _find_merge_candidates(
            engine, limit=10_000, delta=True, respect_checked=False,
        )
        failed_seed_seqs: set[int] = set()
        proposals_created = 0

        for cand in candidates:
            # Re-fetch FULL rows: finder previews are truncated to 400 chars and
            # merged_content must never be generated from truncated text.
            node = engine.db.conn.execute(
                "SELECT id, content, title, type FROM nodes WHERE id = ?",
                (cand["node_a"]["id"],),
            ).fetchone()
            other = engine.db.conn.execute(
                "SELECT id, content, title, type FROM nodes WHERE id = ?",
                (cand["node_b"]["id"],),
            ).fetchone()
            if node is None or other is None:
                continue  # merged/deleted earlier in this same run: drained

            embedding_sim = cand["embedding_sim"]
            title_sim = cand["title_sim"]
            token_sim = cand["token_sim"]
            score = cand["score"]

            # --- LLM confirmation (mandatory) ---
            llm_result = _llm_check_duplicate(settings, node, other)
            if llm_result is None:
                # LLM unavailable for this pair — skip, park the seed
                failed_seed_seqs.add(cand["seed_seq"])
                continue
            if not llm_result.get("is_duplicate"):
                logger.debug(
                    "LLM rejected duplicate for %s / %s: %s",
                    node["id"][:8], other["id"][:8],
                    llm_result.get("reason", ""),
                )
                continue

            # Extract LLM-generated merge content
            merged_content = llm_result.get("merged_content")
            merged_title = llm_result.get("merged_title")
            reason = llm_result.get("reason", "LLM confirmed duplicate")
            reason += f" (score={score:.3f}, embed={embedding_sim:.2f}, title={title_sim:.2f}, token={token_sim:.2f})"

            # Auto-merge for high-confidence duplicates
            if score >= engine.settings.auto_merge_threshold:
                try:
                    result = engine.execute_merge(
                        node["id"], other["id"],
                        merged_content=merged_content,
                        merged_title=merged_title,
                    )
                    logger.info("Auto-merged: %s", result)
                    proposals_created += 1
                    continue
                except Exception as e:
                    logger.warning("Auto-merge failed for %s / %s: %s",
                                   node["id"][:8], other["id"][:8], e)

            # Check no existing merge proposal
            existing = engine.db.conn.execute(
                "SELECT 1 FROM proposals WHERE type = 'merge' AND status = 'pending' "
                "AND (source_nodes LIKE ? OR source_nodes LIKE ?)",
                (f"%{node['id']}%", f"%{other['id']}%"),
            ).fetchone()

            if existing:
                continue

            # Build proposed_action — include merged content preview when available
            proposed_action = f"Merge two {node['type']} memories into one"
            if merged_content is not None:
                proposed_action += (
                    f"\n\nMerged content preview:\n---\n"
                    f"{merged_title or ''}\n{merged_content}\n---"
                )

            proposal_id = str(uuid.uuid4())
            with engine.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO proposals (id, type, status, source_nodes, proposed_action, reason, created) "
                    "VALUES (?, 'merge', 'pending', ?, ?, ?, ?)",
                    (
                        proposal_id,
                        json.dumps([node["id"], other["id"]]),
                        proposed_action,
                        reason,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            proposals_created += 1
        if proposals_created:
            logger.info("Duplicate merger created %d proposals/auto-merges", proposals_created)

        # ponytail: contiguous-prefix advance; deterministic failure parks the
        # cursor — dead-letter escape hatch is upstream #122.
        new_watermark = get_watermark(engine.db.conn, DUPLICATE_WATERMARK_KEY)
        for _seed_id, seed_seq in drained_seeds:  # ascending seq
            if seed_seq in failed_seed_seqs:
                break
            new_watermark = seed_seq
        set_watermark(engine, DUPLICATE_WATERMARK_KEY, new_watermark)

    except Exception as e:
        logger.warning("Duplicate detection failed: %s", e)
