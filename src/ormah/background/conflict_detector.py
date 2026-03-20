"""Detect contradictions between memory nodes."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ormah.background.llm import normalize_conflict_type
from ormah.models.node import Connection, EdgeType

logger = logging.getLogger(__name__)

_LLM_CONFLICT_PROMPT = """\
You are checking whether two memories genuinely contradict each other. False positives create noise in the graph and waste the user's time resolving non-conflicts. Be very conservative — only flag true contradictions.

Memory A (created: {created_a}, space: {space_a}): "{title_a}"
{content_a}

Memory B (created: {created_b}, space: {space_b}): "{title_b}"
{content_b}

## Decision process (stop at first "no")

**1. Same space AND same specific subject?**
- Different spaces/projects → NEVER a conflict. Full stop. A choice in project X says nothing about project Y.
- Same project but different components, features, or layers → NOT a conflict. Systems have many parts.
- Same project and same specific topic (e.g., both about "the database choice for project X's API") → continue.

**2. Are they genuinely incompatible?**
Both claims CANNOT be true simultaneously in the same context:
- "Uses Postgres for the API" + "Uses SQLite for the API" → YES, incompatible (same role, same system)
- "Dislikes tabs" + "Prefers 2-space indent" → NO, compatible (both anti-tab)
- "Prefers REST" + "Chose GraphQL for project X" → NO, preference vs specific decision coexist
- Architectural overview + pivot/redesign of that architecture → NOT a contradiction. The pivot supersedes the overview — that's an evolution, not a tension. An overview describing the system and a later decision changing the system are compatible in time.
- Implementation detail + high-level description of the same system → NOT a contradiction. Different granularity levels coexist.
- Config change (e.g., "intervals changed to daily") + description of what a job does → NOT a contradiction. Changing how often something runs doesn't conflict with what it does.

**3. If genuinely incompatible: evolution or tension?**
- **Evolution**: The person's view changed over time. The newer memory supersedes the older. Use creation dates. Example: "Uses Vim" (2024) → "Switched to VS Code" (2025). This is the most common case.
- **Tension**: Both are simultaneously held beliefs that genuinely pull in opposite directions. Rare. Example: "prefers minimal dependencies" + "keeps adding npm packages".

## Critical: these are NOT conflicts
- A project overview and a later architectural pivot → evolution at most, never tension
- A description of system behavior and a config change to that system → not a conflict
- Two memories about the same technology used in DIFFERENT projects
- A general preference and a project-specific exception
- Two memories at different abstraction levels (architecture doc vs implementation detail)

Return JSON only:
{{
  "same_subject": true or false,
  "conflict": true or false,
  "type": "evolution" or "tension" or "none",
  "evolved_node": "a" or "b" (the NEWER view per creation dates — only if type=evolution),
  "explanation": "one sentence using actual memory titles"
}}"""


def _llm_check_conflict(settings, node_row, other_row) -> dict | None:
    """Ask LLM whether two nodes contradict each other.

    Returns parsed dict or None if the LLM is unavailable or returns
    invalid output.
    """
    from ormah.background.llm_client import llm_generate

    def _get(row, key, default="unknown"):
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    prompt = _LLM_CONFLICT_PROMPT.format(
        title_a=node_row["title"] or "(untitled)",
        content_a=node_row["content"][:500],
        created_a=_get(node_row, "created"),
        space_a=_get(node_row, "space", "global"),
        title_b=other_row["title"] or "(untitled)",
        content_b=other_row["content"][:500],
        created_b=_get(other_row, "created"),
        space_b=_get(other_row, "space", "global"),
    )

    raw = llm_generate(settings, prompt, json_mode=True)
    if raw is None:
        return None

    try:
        result = json.loads(raw)
        if "conflict" not in result:
            return None
        if "type" in result:
            result["type"] = normalize_conflict_type(result["type"])
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM returned invalid JSON for conflict check")
        return None


_BELIEF_TYPES = ('preference', 'fact', 'observation', 'goal')


def _find_conflict_candidates(engine, limit: int = 8) -> list[dict]:
    """Find node pairs that might contradict each other.

    Returns up to *limit* pairs as
    ``[{"node_a": {...}, "node_b": {...}, "similarity": float}]``.
    Node dicts include ``created`` so they can be passed directly to the
    LLM conflict-check prompt.  Does NOT call the LLM.
    """
    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore

        settings = engine.settings
        encoder = get_encoder(settings)
        vec_store = VectorStore(engine.db)

        if settings.conflict_check_all_spaces:
            nodes = engine.db.conn.execute(
                "SELECT id, content, title, type, created, space FROM nodes "
                "WHERE type IN (?, ?, ?, ?)",
                _BELIEF_TYPES,
            ).fetchall()
        else:
            nodes = engine.db.conn.execute(
                "SELECT id, content, title, type, created, space FROM nodes "
                "WHERE type IN (?, ?, ?, ?) AND (space IS NULL OR space = 'null')",
                _BELIEF_TYPES,
            ).fetchall()

        checked: set[tuple[str, str]] = set()
        candidates: list[dict] = []

        for node in nodes:
            if len(candidates) >= limit:
                break

            text = f"{node['title'] or ''} {node['content']}".strip()
            if not text:
                continue

            query_vec = encoder.encode(text)
            similar = vec_store.search(query_vec, limit=15)

            for match in similar:
                if len(candidates) >= limit:
                    break
                if match["id"] == node["id"]:
                    continue

                pair = tuple(sorted([node["id"], match["id"]]))
                if pair in checked:
                    continue
                checked.add(pair)

                similarity = match["similarity"]
                if similarity < 0.4:
                    continue

                other = engine.db.conn.execute(
                    "SELECT id, title, content, type, created, space FROM nodes WHERE id = ?",
                    (match["id"],),
                ).fetchone()
                if other is None:
                    continue
                if other["type"] not in _BELIEF_TYPES:
                    continue

                has_edge = engine.db.conn.execute(
                    "SELECT 1 FROM edges WHERE edge_type IN ('contradicts', 'evolved_from') AND "
                    "((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))",
                    (node["id"], match["id"], match["id"], node["id"]),
                ).fetchone()
                if has_edge:
                    continue

                def _nd(row) -> dict:
                    return {
                        "id": row["id"],
                        "title": row["title"] or "",
                        "type": row["type"] or "",
                        "space": row["space"] or "",
                        "content": (row["content"] or "")[:400],
                        "created": row["created"] or "",
                    }

                candidates.append({
                    "node_a": _nd(node),
                    "node_b": _nd(other),
                    "similarity": round(similarity, 3),
                })

        return candidates

    except Exception as e:
        logger.warning("_find_conflict_candidates failed: %s", e)
        return []


def run_conflict_detection(engine) -> None:
    """Find potentially contradicting nodes and create edges."""
    try:
        settings = engine.settings

        if not settings.llm_enabled:
            logger.debug("Conflict detection skipped: LLM not enabled")
            return

        candidates = _find_conflict_candidates(engine, limit=10000)
        edges_created = 0
        dirty_nodes: dict[str, list[Connection]] = {}

        for candidate in candidates:
            node_a = candidate["node_a"]
            node_b = candidate["node_b"]

            llm_result = _llm_check_conflict(settings, node_a, node_b)
            if llm_result is None:
                continue
            if not llm_result.get("conflict"):
                continue
            if not llm_result.get("same_subject", True):
                continue

            explanation = llm_result.get("explanation", "")
            now = datetime.now(timezone.utc).isoformat()
            conflict_type = llm_result.get("type", "tension")

            with engine.db.transaction() as db_conn:
                if conflict_type == "evolution":
                    evolved = llm_result.get("evolved_node", "b")
                    if evolved == "a":
                        newer_id, older_id = node_a["id"], node_b["id"]
                    else:
                        newer_id, older_id = node_b["id"], node_a["id"]

                    db_conn.execute(
                        "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
                        "VALUES (?, ?, 'evolved_from', 0.9, ?, ?)",
                        (newer_id, older_id, now, explanation),
                    )
                    edge_type_str = "evolved_from"
                    source_id, target_id = newer_id, older_id
                else:
                    db_conn.execute(
                        "INSERT INTO edges (source_id, target_id, edge_type, weight, created, reason) "
                        "VALUES (?, ?, 'contradicts', 0.9, ?, ?)",
                        (node_a["id"], node_b["id"], now, explanation),
                    )
                    edge_type_str = "contradicts"
                    source_id, target_id = node_a["id"], node_b["id"]

            md_conn = Connection(
                target=target_id,
                edge=EdgeType(edge_type_str),
                weight=0.9,
            )
            dirty_nodes.setdefault(source_id, []).append(md_conn)
            edges_created += 1

        # Persist new connections to markdown files
        for nid, new_connections in dirty_nodes.items():
            try:
                mem_node = engine.file_store.load(nid)
                if mem_node is None:
                    continue
                mem_node.connections.extend(new_connections)
                engine.file_store.save(mem_node)
            except Exception as e:
                logger.debug("Failed to persist conflict edge to markdown for %s: %s", nid[:8], e)

        if edges_created:
            logger.info("Conflict detector created %d edges", edges_created)

    except Exception as e:
        logger.warning("Conflict detection failed: %s", e)
