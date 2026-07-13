"""Automatic edge creation based on embedding similarity."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from ormah.background.llm import normalize_link_type

logger = logging.getLogger(__name__)

_WATERMARK_KEY = "auto_link_watermark"


def _get_watermark(conn) -> int:
    """Return the seq of the last fully-processed node, or 0 if unset."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (_WATERMARK_KEY,)).fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except (ValueError, TypeError):
        return 0


def _set_watermark(engine, seq: int) -> None:
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (_WATERMARK_KEY, str(seq)),
        )


def _select_nodes_after(conn, watermark: int, limit: int) -> list:
    """Nodes with seq strictly greater than the watermark, ascending, bounded."""
    return conn.execute(
        "SELECT id, content, title, type, space, seq FROM nodes "
        "WHERE seq > ? ORDER BY seq ASC LIMIT ?",
        (watermark, limit),
    ).fetchall()

_LLM_LINK_INTRO = """\
You are classifying the relationship between two memories in a knowledge graph. These edges power spreading activation during search — when a user finds Memory A, the system traverses edges to surface related context. Bad edges inject noise; good edges dilute the signal from good ones."""

_LLM_LINK_PAIR = """\
Memory A:
- Title: {title_a}
- Type: {type_a}
- Space: {space_a}
- Content: {content_a}

Memory B:
- Title: {title_b}
- Type: {type_b}
- Space: {space_b}
- Content: {content_b}"""

_LLM_LINK_RULES = """\
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

_LLM_LINK_PROMPT = _LLM_LINK_INTRO + "\n\n" + _LLM_LINK_PAIR + "\n\n" + _LLM_LINK_RULES
_LLM_LINK_INSTRUCTIONS = _LLM_LINK_INTRO + "\n\n" + _LLM_LINK_RULES  # fixed block, sent once per batch


def _render_link_pair(pair: dict) -> str:
    """Render one candidate pair for a batched link prompt (#87)."""
    node, other = pair["node"], pair["other"]

    def _get(row, key, default="unknown"):
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    return _LLM_LINK_PAIR.format(
        title_a=node["title"] or "(untitled)", type_a=_get(node, "type"),
        space_a=_get(node, "space", "global"), content_a=node["content"][:2000],
        title_b=other["title"] or "(untitled)", type_b=_get(other, "type"),
        space_b=_get(other, "space", "global"), content_b=other["content"][:2000],
    )


def _llm_classify_link(settings, node_row, other_row) -> dict | None:
    """Ask LLM to classify the relationship between two nodes.

    Returns a dict with keys relationship, reason. On invalid/unparseable
    output the relationship is the "error" sentinel (poison content — the
    node is still marked resolved so it doesn't block the watermark, but no
    edge is created). Returns raw None only when the LLM itself is
    unavailable — a transient condition the caller retries.
    """
    from ormah.background.llm_client import extract_json, llm_generate

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
        return None  # LLM UNAVAILABLE — transient; caller leaves the node unresolved

    try:
        result = json.loads(extract_json(raw))
        if "relationship" not in result:
            return {"relationship": "error", "reason": "missing relationship field"}
        result["relationship"] = normalize_link_type(result["relationship"])
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for link classification")
        return {"relationship": "error", "reason": "invalid LLM output"}


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
    from ormah.embeddings.encoder import get_encoder
    from ormah.embeddings.vector_store import VectorStore, stored_or_encoded

    settings = engine.settings
    encoder = get_encoder(settings)
    vec_store = VectorStore(engine.db)

    conn = engine.db.conn
    watermark = _get_watermark(conn)
    nodes = _select_nodes_after(conn, watermark, settings.auto_link_max_nodes_per_run)
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

        if edge_type not in ("none", "error"):
            conn.execute(
                "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (node_a_id, node_b_id, edge_type, round(similarity, 3), now, reason),
            )

    if edge_type not in ("none", "error"):
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


def run_auto_linker(engine) -> dict | None:
    """Incrementally link nodes with seq above the watermark, judging candidate
    pairs in K-sized LLM calls (#87) and advancing only past fully-resolved nodes.

    One streaming loop for every K: pairs are collected across nodes into a window
    of size K; when it fills it is judged (via pair_batch) and applied. At K=1 the
    window is a single pair, so the flow is exactly the interleaved single-pair
    loop — one judgment applied before the next pair is collected — which the
    existing suite pins as the K=1 regression net. `active` holds the node states
    whose pairs are still in flight, in seq order; a node advances the watermark
    only once it is fully collected, drained, and resolved, with every earlier
    node already advanced.
    """
    t0 = time.monotonic()
    try:
        from ormah.background.llm.pair_batch import judge_pairs
        from ormah.embeddings.vector_store import VectorStore

        settings = engine.settings
        if not settings.llm_enabled:
            logger.debug("Auto-linker skipped: LLM not enabled")
            return {"skipped": "llm_disabled"}

        vec_store = VectorStore(engine.db)
        conn = engine.db.conn
        threshold = settings.auto_link_similarity_threshold
        cross_space_penalty = settings.auto_link_cross_space_penalty
        max_edges = settings.auto_link_max_edges_per_run
        max_pairs = settings.auto_link_max_pairs_per_run      # 0 = unbounded (default)
        k = max(settings.auto_link_pairs_per_call or settings.maintenance_pairs_per_call, 1)

        watermark = _get_watermark(conn)
        nodes = _select_nodes_after(conn, watermark, settings.auto_link_max_nodes_per_run)

        seen_pairs: set[tuple[str, str]] = set()
        window: list[dict] = []          # pending {state, pair}, at most K, across nodes
        active: list[dict] = []          # node states with pairs in flight, seq order
        created = 0
        pairs_attempted = 0
        pairs_evaluated = 0
        last_complete: int | None = None
        cap_hit = False
        stopped = False

        def _advance_watermark() -> None:
            """Pop fully-collected, drained, resolved nodes off the front of `active`;
            each sets the watermark. Stop at the first node not yet done — it and
            everything after it must be reprocessed next run."""
            nonlocal last_complete
            while active:
                head = active[0]
                if not head["collected"] or head["pending"] > 0 or not head["resolved"]:
                    break
                last_complete = head["node"]["seq"]
                active.pop(0)

        def _flush() -> bool:
            """Judge + apply the pairs in `window` (one K-chunk). Returns False when
            the run must stop (LLM outage or edge budget spent)."""
            nonlocal created, pairs_attempted, pairs_evaluated
            verdicts = judge_pairs(
                settings, _LLM_LINK_INSTRUCTIONS, [slot["pair"] for slot in window],
                _render_link_pair,
                judge_single=lambda p: _llm_classify_link(settings, p["node"], p["other"]),
                k=k,
            )
            ok = True
            for slot, v in zip(window, verdicts):
                state = slot["state"]
                state["pending"] -= 1
                if not ok:
                    continue          # already stopping — just drain pending counts
                if created >= max_edges:
                    state["resolved"] = False   # budget spent -> don't advance past this node
                    ok = False
                    continue
                pairs_attempted += 1
                if v is None:                   # unavailable/unjudged -> fail closed
                    state["resolved"] = False
                    ok = False
                    continue
                # The "error" sentinel (unparseable/missing output) must never be
                # normalized — normalize_link_type maps unknowns to related_to, which
                # would forge an edge from poison content. Real values are normalized
                # (matches _llm_classify_link's single-pair path and #90).
                raw_rel = v.get("relationship", "error")
                relationship = "error" if raw_rel == "error" else normalize_link_type(raw_rel)
                if relationship != "error":
                    pairs_evaluated += 1
                _apply_edge(engine, state["node"]["id"], slot["pair"]["match_id"],
                            relationship, v.get("reason", ""), slot["pair"]["similarity"])
                # 'error' (poison content) is recorded in auto_link_checked by _apply_edge
                # and the node still counts as resolved → watermark advances (council v2 crit#2).
                if relationship not in ("none", "error"):
                    created += 1
            window.clear()
            return ok

        for node in nodes:
            if stopped or cap_hit:
                break
            text = f"{node['title'] or ''} {node['content']}".strip()
            node_vec = vec_store.get(node["id"]) if text else None
            if text and node_vec is None:
                # Vectorless node (store mid-backfill). Don't abort the whole run —
                # that froze the cursor for the entire store. Park the watermark here
                # (resolved=False → never advance past it, fail-closed) but keep
                # processing later nodes so they still get edges. A later run advances
                # once this node's vector lands. A PERMANENTLY vectorless node parks
                # the cursor for good — hence the WARNING (deferred: retry set
                # decoupled from the cursor).
                logger.warning(
                    "auto_linker: node %s (seq %s) has no vector; watermark parked, "
                    "continuing with later nodes",
                    node["id"][:8], node["seq"],
                )
                active.append({"node": node, "resolved": False, "pending": 0, "collected": True})
                continue
            state = {"node": node, "resolved": True, "pending": 0, "collected": False}
            active.append(state)
            if text:
                for match in vec_store.search(node_vec, limit=6):
                    if created >= max_edges:
                        state["resolved"] = False   # budget spent mid-node -> interrupted
                        stopped = True
                        break
                    if match["id"] == node["id"]:
                        continue
                    similarity = match["similarity"]
                    other_space = conn.execute(
                        "SELECT space FROM nodes WHERE id = ?", (match["id"],)).fetchone()
                    if other_space is not None and (node["space"] or "") != (other_space["space"] or ""):
                        similarity -= cross_space_penalty
                    if similarity < threshold:
                        continue
                    pair_key = tuple(sorted([node["id"], match["id"]]))
                    if pair_key in seen_pairs:
                        continue
                    if conn.execute("SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?",
                                    pair_key).fetchone():
                        continue
                    if conn.execute(
                            "SELECT 1 FROM edges WHERE (source_id = ? AND target_id = ?) "
                            "OR (source_id = ? AND target_id = ?)",
                            (node["id"], match["id"], match["id"], node["id"])).fetchone():
                        continue
                    other = conn.execute(
                        "SELECT title, content, type, space FROM nodes WHERE id = ?",
                        (match["id"],)).fetchone()
                    if other is None:
                        continue
                    if max_pairs and (pairs_attempted + len(window)) >= max_pairs:
                        cap_hit = True
                        state["resolved"] = False   # partially collected -> never advance past
                        break
                    seen_pairs.add(pair_key)
                    state["pending"] += 1
                    window.append({"state": state, "pair": {
                        "node": node, "other": other,
                        "match_id": match["id"], "similarity": similarity}})
                    if len(window) >= k:
                        if not _flush():
                            stopped = True
                            break
            state["collected"] = True
            if not stopped and not cap_hit:
                _advance_watermark()

        if window and not stopped and not cap_hit:
            _flush()
        _advance_watermark()

        if last_complete is not None:
            _set_watermark(engine, last_complete)

        # `seq` is 1-based (meta.node_seq_next starts at 1), so a real node's seq is
        # never 0 and `last_complete or watermark` cannot silently fall back.
        backlog = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE seq > ?", (last_complete or watermark,)
        ).fetchone()[0]
        duration = time.monotonic() - t0
        stats = {
            "nodes_scanned": len(nodes),
            "pairs_attempted": pairs_attempted,
            "pairs_evaluated": pairs_evaluated,
            "edges_created": created,
            "cap_hit": cap_hit,
            "backlog_nodes": backlog,
            "duration_s": round(duration, 3),
            "pairs_per_s": round(pairs_evaluated / duration, 2) if duration > 0 else 0.0,
        }
        logger.info("auto_linker run: %s", stats)
        return stats

    except Exception as e:
        logger.warning("Auto-linker failed: %s", e)
        return {"error": str(e)}
