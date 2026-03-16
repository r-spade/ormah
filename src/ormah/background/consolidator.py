"""Background job: consolidate clusters of similar working-tier memories via LLM."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Maximum number of clusters to consolidate per run.
_MAX_CLUSTERS_PER_RUN = 10

# Minimum cluster size to justify consolidation.
_MIN_CLUSTER_SIZE = 2

# Cosine similarity threshold for clustering.
_CLUSTER_THRESHOLD = 0.6


def run_consolidation(engine) -> None:
    """Find clusters of similar working memories and consolidate via LLM."""
    settings = engine.settings
    if not settings.llm_enabled:
        return

    try:
        from ormah.embeddings.encoder import get_encoder
        from ormah.embeddings.vector_store import VectorStore
    except ImportError:
        return

    conn = engine.db.conn

    # Get all working-tier nodes
    rows = conn.execute(
        "SELECT id, title, content, space FROM nodes WHERE tier = 'working'"
    ).fetchall()
    if len(rows) < _MIN_CLUSTER_SIZE:
        return

    encoder = get_encoder(settings)
    vec_store = VectorStore(engine.db)

    # Greedy clustering
    clustered_ids: set[str] = set()
    clusters: list[list[dict]] = []

    for row in rows:
        nid = row["id"]
        if nid in clustered_ids:
            continue

        # Get embedding and find similar nodes
        node_vec = vec_store.get(nid)
        if node_vec is None:
            continue

        similar = vec_store.search(node_vec, limit=20)
        cluster = [dict(row)]
        clustered_ids.add(nid)

        for match in similar:
            mid = match["id"]
            if mid == nid or mid in clustered_ids:
                continue
            if match["similarity"] < _CLUSTER_THRESHOLD:
                continue
            # Must also be working tier
            m_row = conn.execute(
                "SELECT id, title, content, space, tier FROM nodes WHERE id = ?",
                (mid,),
            ).fetchone()
            if m_row is None or m_row["tier"] != "working":
                continue
            cluster.append(dict(m_row))
            clustered_ids.add(mid)

        if len(cluster) >= _MIN_CLUSTER_SIZE:
            clusters.append(cluster)

    if not clusters:
        return

    consolidated_count = 0
    for cluster in clusters[:_MAX_CLUSTERS_PER_RUN]:
        try:
            _consolidate_cluster(engine, cluster)
            consolidated_count += 1
        except Exception as e:
            logger.warning("Failed to consolidate cluster: %s", e)

    if consolidated_count:
        logger.info("Consolidated %d cluster(s)", consolidated_count)


def _consolidate_cluster(engine, cluster: list[dict]) -> None:
    """Consolidate a single cluster using LLM summarization."""
    from ormah.background.llm_client import llm_generate
    from ormah.models.node import (
        CreateNodeRequest,
        EdgeType,
        ConnectRequest,
        Tier,
    )

    # Build prompt
    items = []
    for node in cluster:
        title = node.get("title") or "Untitled"
        content = node.get("content", "")
        items.append(f"- [{title}]: {content[:300]}")
    items_text = "\n".join(items)

    prompt = f"""\
You are consolidating a cluster of semantically similar memories into a single, richer memory. The consolidated memory will be stored as a new working-tier node, and the originals will be demoted to archival (still searchable but deprioritized). This means your output becomes the PRIMARY representation of this knowledge — it must be complete.

Memories to consolidate:
{items_text}

## Your task

Synthesize these into ONE memory that is better than any individual original. The result should read as a well-written knowledge base entry — not a list of bullet points stitched together, but a coherent narrative.

## Rules

1. **Preserve every concrete detail**: Names, versions, paths, numbers, dates, specific choices, file locations, command flags. If any original says "bge-base-en-v1.5" or "port 8787", the consolidated version must include it. Never generalize specifics — "chose SQLite" must stay "chose SQLite", not "chose a database".

2. **Preserve all reasoning**: If any memory explains WHY something was decided, what alternative was rejected, or what constraint drove the design, that reasoning MUST appear. Reasoning is the most valuable part of memory — it prevents re-litigating decisions.

3. **Eliminate only true redundancy**: If three memories all state "uses FastAPI", say it once. But if they say different things about FastAPI (routing patterns, middleware choices, deployment config), keep ALL of those distinct facts.

4. **Choose the most valuable type**: Priority order: decision > procedure > observation > concept > fact. If the cluster contains a decision and supporting facts, the consolidated type should be "decision" because that's what makes the memory actionable.

5. **Title for searchability**: Titles get 10x weight in full-text search. Make it specific and keyword-rich. BAD: "Project architecture". GOOD: "Ormah uses FastAPI + SQLite with hybrid FTS/vector search". The title should let someone find this memory by searching for any of its key topics.

6. **Content length**: 3-8 sentences. Long enough to be self-contained, short enough to be scannable. This will be displayed to an AI assistant as context — it needs to absorb it quickly.

Return a JSON object:
{{
  "title": "specific, keyword-rich title, 5-15 words",
  "summary": "consolidated content as a coherent narrative, preserving all unique details",
  "type": "fact|decision|preference|event|person|project|concept|procedure|goal|observation"
}}"""

    raw = llm_generate(engine.settings, prompt, json_mode=True)
    if raw is None:
        return

    result = json.loads(raw)
    title = result.get("title", "Consolidated memory")
    summary = result.get("summary", "")
    node_type = result.get("type", "fact")

    if not summary:
        return

    # Determine space by majority vote
    space_counts: dict[str | None, int] = {}
    for node in cluster:
        sp = node.get("space")
        space_counts[sp] = space_counts.get(sp, 0) + 1
    space = max(space_counts, key=space_counts.get)  # type: ignore[arg-type]

    # Create consolidated node
    req = CreateNodeRequest(
        content=summary,
        type=node_type,
        title=title,
        space=space,
        tags=["consolidated"],
    )
    new_id, _ = engine.remember(req, agent_id="consolidator")

    # Transfer identity edges: if any original was an identity node (linked via
    # 'defines' from user node), link the new consolidated node to the user node
    # and carry over the about_self tag.
    if engine.user_node_id:
        conn = engine.db.conn
        original_ids = [n["id"] for n in cluster]
        placeholders = ",".join("?" * len(original_ids))
        has_identity = conn.execute(
            f"SELECT 1 FROM edges WHERE source_id = ? AND edge_type = 'defines' "
            f"AND target_id IN ({placeholders}) LIMIT 1",
            [engine.user_node_id] + original_ids,
        ).fetchone()
        if has_identity:
            try:
                engine.connect(ConnectRequest(
                    source_id=engine.user_node_id,
                    target_id=new_id,
                    edge=EdgeType.defines,
                    weight=1.0,
                ))
            except Exception:
                pass
            # Ensure about_self tag on the new node
            new_node = engine.file_store.load(new_id)
            if new_node and "about_self" not in new_node.tags:
                new_node.tags.append("about_self")
                engine.file_store.save(new_node)
                with engine.db.transaction() as tx_conn:
                    tx_conn.execute(
                        "INSERT OR IGNORE INTO node_tags (node_id, tag) VALUES (?, 'about_self')",
                        (new_id,),
                    )

    # Create derived_from edges to originals and demote originals to archival
    for node in cluster:
        try:
            engine.connect(ConnectRequest(
                source_id=new_id,
                target_id=node["id"],
                edge=EdgeType.derived_from,
                weight=1.0,
            ))
        except Exception:
            pass

        # Demote original to archival
        from ormah.models.node import UpdateNodeRequest
        engine.update_node(node["id"], UpdateNodeRequest(tier=Tier.archival))
