"""Background job: consolidate clusters of similar working-tier memories via LLM."""

from __future__ import annotations

import json
import logging

from ormah.background.memory_lock import RestoredUnderfoot, restore_aware_job

logger = logging.getLogger(__name__)


def _find_consolidation_clusters(engine, limit: int = 4) -> list[list[dict]]:
    """Find clusters of similar working-tier nodes for consolidation.

    Returns up to *limit* clusters, each a list of node dicts (max
    ``engine.settings.consolidation_max_cluster_nodes`` nodes).
    Does NOT call the LLM — pure similarity-based clustering.
    """
    try:
        from ormah.embeddings.vector_store import VectorStore
    except ImportError:
        return []

    conn = engine.db.conn
    s = engine.settings
    min_size = s.consolidation_min_cluster_size
    max_nodes = s.consolidation_max_cluster_nodes
    threshold = s.consolidation_cluster_threshold

    if max_nodes < min_size:
        logger.warning(
            "consolidation_max_cluster_nodes (%d) < consolidation_min_cluster_size (%d); "
            "no cluster can ever be emitted",
            max_nodes, min_size,
        )
        return []

    rows = conn.execute(
        "SELECT id, title, content, space FROM nodes WHERE tier = 'working'"
    ).fetchall()
    if len(rows) < min_size:
        return []

    try:
        vec_store = VectorStore(engine.db)
    except Exception:
        return []

    clustered_ids: set[str] = set()
    clusters: list[list[dict]] = []

    for row in rows:
        if len(clusters) >= limit:
            break

        nid = row["id"]
        if nid in clustered_ids:
            continue

        node_vec = vec_store.get(nid)
        if node_vec is None:
            continue

        similar = vec_store.search(node_vec, limit=20)
        cluster = [dict(row)]
        clustered_ids.add(nid)

        for match in similar:
            if len(cluster) >= max_nodes:
                break
            mid = match["id"]
            if mid == nid or mid in clustered_ids:
                continue
            if match["similarity"] < threshold:
                continue
            m_row = conn.execute(
                "SELECT id, title, content, space, tier FROM nodes WHERE id = ?",
                (mid,),
            ).fetchone()
            if m_row is None or m_row["tier"] != "working":
                continue
            cluster.append(dict(m_row))
            clustered_ids.add(mid)

        if len(cluster) >= min_size:
            clusters.append(cluster)

    return clusters


def _apply_consolidation(
    engine,
    node_ids: list[str],
    title: str,
    content: str,
    node_type: str,
    *,
    epoch: int | None = None,
) -> str:
    """Create a consolidated node, link originals, and demote them to archival.

    Returns the new node's ID. When *epoch* is given the whole operation is one
    exclusive apply step (#240): a restore landing mid-consolidation must not be
    able to observe the new node without the originals demoted, or vice versa.
    ``apply_maintenance_results`` passes no epoch — it already holds L_mem.
    """
    from contextlib import nullcontext

    from ormah.models.node import (
        ConnectRequest,
        CreateNodeRequest,
        EdgeType,
        Tier,
        UpdateNodeRequest,
    )

    guard = engine.memory_operation_at(epoch) if epoch is not None else nullcontext()
    with guard:
        conn = engine.db.conn
        placeholders = ",".join("?" * len(node_ids))

        # Fetch cluster nodes for space determination and identity transfer
        cluster_rows = conn.execute(
            f"SELECT id, space FROM nodes WHERE id IN ({placeholders})",
            node_ids,
        ).fetchall()
        cluster = [dict(r) for r in cluster_rows]

        # Determine space by majority vote
        space_counts: dict[str | None, int] = {}
        for node in cluster:
            sp = node.get("space")
            space_counts[sp] = space_counts.get(sp, 0) + 1
        space = max(space_counts, key=space_counts.get)  # type: ignore[arg-type]

        # Create consolidated node
        req = CreateNodeRequest(
            content=content,
            type=node_type,
            title=title,
            space=space,
            tags=["consolidated"],
        )
        new_id, _ = engine.remember(req, agent_id="consolidator")

        # Transfer identity edges
        if engine.user_node_id:
            has_identity = conn.execute(
                f"SELECT 1 FROM edges WHERE source_id = ? AND edge_type = 'defines' "
                f"AND target_id IN ({placeholders}) LIMIT 1",
                [engine.user_node_id] + node_ids,
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
                new_node = engine.file_store.load(new_id)
                if new_node and "about_self" not in new_node.tags:
                    new_node.tags.append("about_self")
                    new_node.touch_updated()
                    engine.file_store.save(new_node)
                    with engine.db.transaction() as tx_conn:
                        tx_conn.execute(
                            "INSERT OR IGNORE INTO node_tags (node_id, tag) VALUES (?, 'about_self')",
                            (new_id,),
                        )

        # Create derived_from edges and demote originals to archival
        for node_id in node_ids:
            try:
                engine.connect(ConnectRequest(
                    source_id=new_id,
                    target_id=node_id,
                    edge=EdgeType.derived_from,
                    weight=1.0,
                ))
            except Exception:
                pass
            # Revalidate before demoting: node_ids was snapshotted before the LLM call,
            # so a foreground promotion may have landed since. Skip the item, do not abort
            # the run — one node changing invalidates that node, not the whole cluster (#240).
            current = conn.execute(
                "SELECT tier FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if current is None or current["tier"] != "working":
                logger.debug(
                    "Consolidation left %s alone: tier changed since the snapshot", node_id[:8])
                continue
            engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))

        return new_id


@restore_aware_job
def run_consolidation(engine, epoch: int) -> None:
    """Find clusters of similar working memories and consolidate via LLM."""
    settings = engine.settings
    if not settings.llm_enabled:
        return

    clusters = _find_consolidation_clusters(
        engine, limit=settings.consolidation_max_clusters_per_run
    )
    if not clusters:
        return

    consolidated_count = 0
    for cluster in clusters:
        try:
            _consolidate_cluster(engine, cluster, epoch)
            consolidated_count += 1
        except RestoredUnderfoot:
            raise
        except Exception as e:
            logger.warning("Failed to consolidate cluster: %s", e)

    if consolidated_count:
        logger.info("Consolidated %d cluster(s)", consolidated_count)


def _consolidate_cluster(engine, cluster: list[dict], epoch: int) -> None:
    """Consolidate a single cluster using LLM summarization."""
    from ormah.background.llm_client import extract_json, llm_generate

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

    result = json.loads(extract_json(raw))
    title = result.get("title", "Consolidated memory")
    summary = result.get("summary", "")
    node_type = result.get("type", "fact")

    if not summary:
        return

    node_ids = [n["id"] for n in cluster]
    _apply_consolidation(engine, node_ids, title, summary, node_type, epoch=epoch)
