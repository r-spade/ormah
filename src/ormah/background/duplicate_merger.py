"""Detect near-duplicate memories and create merge proposals."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

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
    from ormah.background.llm_client import llm_generate

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
        result = json.loads(raw)
        if "is_duplicate" not in result:
            return None
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for duplicate check")
        return None


def run_duplicate_detection(engine) -> None:
    """Find near-duplicate nodes and create merge proposals.

    Uses a multi-signal approach: embedding similarity (primary),
    title similarity, and token overlap for candidate generation.
    LLM confirmation is mandatory — no merges happen without LLM
    saying ``is_duplicate: true``.
    """
    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore

        settings = engine.settings
        encoder = get_encoder(settings)
        vec_store = VectorStore(engine.db)

        if not settings.llm_enabled:
            logger.debug("Duplicate detection skipped: LLM not enabled")
            return

        user_node_id = getattr(engine, "user_node_id", None)

        nodes = engine.db.conn.execute("SELECT id, content, title, type FROM nodes").fetchall()
        checked = set()
        proposals_created = 0

        for node in nodes:
            if node["id"] == user_node_id:
                continue
            text = f"{node['title'] or ''} {node['content']}".strip()
            if not text:
                continue

            query_vec = encoder.encode(text)
            # Fetch more candidates since we use a lower embedding pre-filter
            similar = vec_store.search(query_vec, limit=6)

            for match in similar:
                if match["id"] == node["id"]:
                    continue
                if match["id"] == user_node_id:
                    continue

                pair = tuple(sorted([node["id"], match["id"]]))
                if pair in checked:
                    continue
                checked.add(pair)

                embedding_sim = match["similarity"]
                # Pre-filter: skip very dissimilar pairs to avoid wasted work
                if embedding_sim < 0.25:
                    continue

                # Same type only
                other = engine.db.conn.execute(
                    "SELECT type, title, content FROM nodes WHERE id = ?", (match["id"],)
                ).fetchone()
                if other is None or other["type"] != node["type"]:
                    continue

                # Compute multi-signal score
                title_sim = _title_similarity(node["title"], other["title"])
                other_text = f"{other['title'] or ''} {other['content']}".strip()
                token_sim = _token_overlap(text, other_text)
                score = _composite_score(embedding_sim, title_sim, token_sim)

                if score < _COMPOSITE_THRESHOLD:
                    continue

                # --- LLM confirmation (mandatory) ---
                llm_result = _llm_check_duplicate(settings, node, other)
                if llm_result is None:
                    # LLM unavailable for this pair — skip
                    continue
                if not llm_result.get("is_duplicate"):
                    logger.debug(
                        "LLM rejected duplicate for %s / %s: %s",
                        node["id"][:8], match["id"][:8],
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
                            node["id"], match["id"],
                            merged_content=merged_content,
                            merged_title=merged_title,
                        )
                        logger.info("Auto-merged: %s", result)
                        proposals_created += 1
                        continue
                    except Exception as e:
                        logger.warning("Auto-merge failed for %s / %s: %s",
                                       node["id"][:8], match["id"][:8], e)

                # Check no existing merge proposal
                existing = engine.db.conn.execute(
                    "SELECT 1 FROM proposals WHERE type = 'merge' AND status = 'pending' "
                    "AND (source_nodes LIKE ? OR source_nodes LIKE ?)",
                    (f"%{node['id']}%", f"%{match['id']}%"),
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
                            json.dumps([node["id"], match["id"]]),
                            proposed_action,
                            reason,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                proposals_created += 1
        if proposals_created:
            logger.info("Duplicate merger created %d proposals/auto-merges", proposals_created)

    except Exception as e:
        logger.warning("Duplicate detection failed: %s", e)
