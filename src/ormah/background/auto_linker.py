"""Automatic edge creation based on embedding similarity."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ormah.background.llm import normalize_link_type

logger = logging.getLogger(__name__)

_LLM_LINK_PROMPT = """\
You are classifying the relationship between two memories in a knowledge graph. These edges power spreading activation during search — when a user finds Memory A, the system traverses edges to surface related context. Bad edges inject noise; good edges dilute the signal from good ones.

Memory A:
- Title: {title_a}
- Type: {type_a}
- Space: {space_a}
- Content: {content_a}

Memory B:
- Title: {title_b}
- Type: {type_b}
- Space: {space_b}
- Content: {content_b}

## Decision process

**Step 1: Would surfacing B actually help someone reading A?**
Imagine a developer just found Memory A via search. Would showing them Memory B provide a genuine "aha" moment, fill in a missing piece, or change how they think about A? If not, return "none". Shared keywords or same project is NOT enough.

**Step 2: Is the relationship structural or just topical?**
- Structural = one literally builds on, contains, or requires the other
- Topical = they mention similar things but are independently useful

Topical similarity alone → "none". Do NOT use "related_to" as a catch-all for "these are about the same project" or "these share a domain".

**Step 3: Pick the most specific edge type that fits.**

Edge types (ordered by specificity — prefer higher):

- **"supports"** (1.0x weight): A provides evidence, reasoning, or implementation detail that makes B more useful. Reading B without A would miss important context. Examples: observation that motivated a decision; a fact that validates a preference.

- **"part_of"** (1.0x): A is literally a component or sub-topic within B's scope. Examples: "FTS5 config" is part_of "search architecture"; a specific endpoint is part_of "API design".

- **"depends_on"** (1.0x): A cannot function or be implemented without B. Hard dependency. Examples: a client that calls a server depends_on the server existing; a feature that uses a specific library depends_on that library choice. Not just topical relevance — one literally requires the other to work.

- **"contradicts"** (0.4x): A and B make genuinely incompatible claims that are BOTH believed to be true RIGHT NOW. Both must be current beliefs held simultaneously. If one supersedes or replaces the other (e.g., "uses tool A" followed by "switched to tool B"), that is NOT a contradiction — it's temporal succession where the older fact is simply outdated. A general preference with a justified exception is also NOT a contradiction — engineers routinely make specific tradeoffs against general principles.

- **"related_to"** (0.7x): Use ONLY when ALL three conditions are met: (1) knowing A would genuinely change how you act on B, (2) no stronger edge type fits, AND (3) the connection is specific enough that you could explain it in one concrete sentence (not "they're both about X").

- **"none"**: Default. Use when memories share a project, domain, or keywords but are independently useful. Two features of the same project with no dependency → "none". Two preferences about different domains → "none". A decision and an unrelated implementation detail → "none".

## Common traps — these are ALL "none"

- **General preference + specific exception**: "prefers minimal dependencies" and "added library X because it was worth the tradeoff" — this is normal engineering judgment, not a contradiction or tension. A general preference doesn't create an edge to every decision that weighed against it.
- **Superseded fact + newer decision**: "system uses tool A" and "switched to tool B" — this is temporal succession. The fact is outdated, but that doesn't create a structural relationship. The conflict detector handles evolutions, not the linker.
- **Same technology in different projects**: "Project X uses Redis" and "Project Y uses Redis" — shared technology is not a relationship. Each memory is independently useful.
- **Sequential observations about the same metric**: "accuracy was 60%" and "accuracy improved to 88%" — these are snapshots in time, not structurally related. One does not help you understand the other.
- **Same domain, different aspects**: Two memories about search (one about FTS config, one about vector scoring) that don't literally build on each other — they share a topic but are independently useful.

**Step 4: If you're about to return "contradicts", STOP and verify ALL of these:**
- Neither memory uses words like "switched", "changed", "migrated", "moved to", "replaced", "updated" → if either does, it's temporal succession → "none"
- Neither memory is a justified exception to the other (e.g., a general rule + a specific case where the tradeoff was worth it) → if it is, that's normal engineering judgment → "none"
- A reasonable engineer could NOT hold both views simultaneously without being inconsistent → if they could (e.g., "prefer X in general" + "chose Y here because of Z"), return "none"
- The claims are logically incompatible like "use tabs" vs "use spaces" for the same context, or "library X is best" vs "library X is terrible"

If ANY check fails, return "none". The conflict detector (a separate system) handles temporal evolutions and tensions — the linker should NOT.

**The bar for "none" is LOW. The bar for any edge is HIGH.** A sparse graph with 50 high-quality edges beats a dense graph with 500 mediocre ones. Every edge is traversed during search — noise edges actively hurt retrieval quality.

Return JSON:
{{
  "relationship": "supports|contradicts|part_of|depends_on|related_to|none",
  "reason": "one concrete sentence: what specific insight does finding A give you about B?"
}}"""


def _llm_classify_link(settings, node_row, other_row) -> dict | None:
    """Ask LLM to classify the relationship between two nodes.

    Returns parsed dict with keys relationship, reason — or None if the LLM
    is unavailable or returns invalid output.
    """
    from ormah.background.llm_client import llm_generate

    def _get(row, key, default="unknown"):
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    prompt = _LLM_LINK_PROMPT.format(
        title_a=node_row["title"] or "(untitled)",
        type_a=_get(node_row, "type"),
        space_a=_get(node_row, "space", "global"),
        content_a=node_row["content"][:2000],
        title_b=other_row["title"] or "(untitled)",
        type_b=_get(other_row, "type"),
        space_b=_get(other_row, "space", "global"),
        content_b=other_row["content"][:2000],
    )

    raw = llm_generate(settings, prompt, json_mode=True)
    if raw is None:
        return None

    try:
        result = json.loads(raw)
        if "relationship" not in result:
            return None
        result["relationship"] = normalize_link_type(result["relationship"])
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for link classification")
        return None


def _node_dict(row, content_limit: int = 400) -> dict:
    """Convert a DB row to a plain node dict for candidate lists."""
    return {
        "id": row["id"],
        "title": row["title"] or "",
        "type": row["type"] or "",
        "space": row["space"] or "",
        "content": (row["content"] or "")[:content_limit],
    }


def _find_link_candidates(engine, limit: int = 8) -> list[dict]:
    """Find node pairs that need link classification.

    Returns up to *limit* pairs as
    ``[{"node_a": {...}, "node_b": {...}, "similarity": float}]``.
    Does NOT call the LLM — just applies the same pre-filters as
    ``run_auto_linker`` (similarity threshold, cross-space penalty,
    not in auto_link_checked, no existing edge).
    """
    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore

        settings = engine.settings
        encoder = get_encoder(settings)
        vec_store = VectorStore(engine.db)

        conn = engine.db.conn
        nodes = conn.execute("SELECT id, content, title, type, space FROM nodes").fetchall()
        threshold = settings.auto_link_similarity_threshold
        cross_space_penalty = settings.auto_link_cross_space_penalty

        candidates: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()

        for node in nodes:
            if len(candidates) >= limit:
                break

            text = f"{node['title'] or ''} {node['content']}".strip()
            if not text:
                continue

            query_vec = encoder.encode(text)
            similar = vec_store.search(query_vec, limit=6)

            for match in similar:
                if len(candidates) >= limit:
                    break
                if match["id"] == node["id"]:
                    continue

                similarity = match["similarity"]

                other_space = conn.execute(
                    "SELECT space FROM nodes WHERE id = ?", (match["id"],)
                ).fetchone()
                if other_space is not None:
                    src_space = node["space"] or ""
                    tgt_space = other_space["space"] or ""
                    if src_space != tgt_space:
                        similarity -= cross_space_penalty

                if similarity < threshold:
                    continue

                pair = tuple(sorted([node["id"], match["id"]]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                already_checked = conn.execute(
                    "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?",
                    pair,
                ).fetchone()
                if already_checked:
                    continue

                existing = conn.execute(
                    "SELECT 1 FROM edges WHERE "
                    "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                    (node["id"], match["id"], match["id"], node["id"]),
                ).fetchone()
                if existing:
                    continue

                other = conn.execute(
                    "SELECT id, title, content, type, space FROM nodes WHERE id = ?",
                    (match["id"],),
                ).fetchone()
                if other is None:
                    continue

                candidates.append({
                    "node_a": _node_dict(node),
                    "node_b": _node_dict(other),
                    "similarity": round(similarity, 3),
                })

        return candidates

    except Exception as e:
        logger.warning("_find_link_candidates failed: %s", e)
        return []


def _apply_edge(
    engine,
    node_a_id: str,
    node_b_id: str,
    edge_type: str,
    reason: str,
    similarity: float = 0.0,
) -> None:
    """Record a link decision: write to auto_link_checked and optionally create an edge.

    ``edge_type="none"`` records the pair as checked without creating an edge.
    """
    from ormah.models.node import Connection, EdgeType

    pair = tuple(sorted([node_a_id, node_b_id]))
    now = datetime.now(timezone.utc).isoformat()

    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, ?, ?)",
            (*pair, edge_type, now),
        )

        if edge_type != "none":
            conn.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_a_id, node_b_id, edge_type, round(similarity, 3), now, reason),
            )

    if edge_type != "none":
        try:
            mem_node = engine.file_store.load(node_a_id)
            if mem_node is not None:
                md_conn = Connection(
                    target=node_b_id,
                    edge=EdgeType(edge_type),
                    weight=round(similarity, 2),
                )
                mem_node.connections.append(md_conn)
                engine.file_store.save(mem_node)
        except Exception as e:
            logger.debug("Failed to persist connection to markdown for %s: %s", node_a_id[:8], e)


def run_auto_linker(engine) -> None:
    """Find similar nodes and create edges between them."""
    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore

        settings = engine.settings
        encoder = get_encoder(settings)
        vec_store = VectorStore(engine.db)

        if not settings.llm_enabled:
            logger.debug("Auto-linker skipped: LLM not enabled")
            return

        conn = engine.db.conn
        nodes = conn.execute("SELECT id, content, title, type, space FROM nodes").fetchall()
        threshold = settings.auto_link_similarity_threshold
        cross_space_penalty = settings.auto_link_cross_space_penalty
        max_edges = settings.auto_link_max_edges_per_run
        created = 0

        for node in nodes:
            if created >= max_edges:
                break

            text = f"{node['title'] or ''} {node['content']}".strip()
            if not text:
                continue

            query_vec = encoder.encode(text)
            similar = vec_store.search(query_vec, limit=6)

            for match in similar:
                if match["id"] == node["id"]:
                    continue

                similarity = match["similarity"]

                other_space = conn.execute(
                    "SELECT space FROM nodes WHERE id = ?", (match["id"],)
                ).fetchone()
                if other_space is not None:
                    src_space = node["space"] or ""
                    tgt_space = other_space["space"] or ""
                    if src_space != tgt_space:
                        similarity -= cross_space_penalty

                if similarity < threshold:
                    continue

                pair = tuple(sorted([node["id"], match["id"]]))

                already_checked = conn.execute(
                    "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?",
                    pair,
                ).fetchone()
                if already_checked:
                    continue

                existing = conn.execute(
                    "SELECT 1 FROM edges WHERE "
                    "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
                    (node["id"], match["id"], match["id"], node["id"]),
                ).fetchone()
                if existing:
                    continue

                other = conn.execute(
                    "SELECT title, content, type, space FROM nodes WHERE id = ?",
                    (match["id"],),
                ).fetchone()
                if other is None:
                    continue

                llm_result = _llm_classify_link(settings, node, other)
                if llm_result is None:
                    # LLM unavailable for this pair — skip without recording
                    continue

                relationship = llm_result["relationship"]
                reason = llm_result.get("reason", "")
                _apply_edge(engine, node["id"], match["id"], relationship, reason, similarity)

                if relationship != "none":
                    created += 1

        if created:
            logger.info("Auto-linker created %d edges", created)

    except Exception as e:
        logger.warning("Auto-linker failed: %s", e)
