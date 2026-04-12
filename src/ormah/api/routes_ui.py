"""UI API routes for the web graph explorer."""

from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ui", tags=["ui"])


@router.get("/graph")
async def get_graph(request: Request):
    """Get full graph data for visualization."""
    engine = request.app.state.engine
    nodes = engine.db.conn.execute("SELECT * FROM nodes").fetchall()
    edges = engine.graph.get_all_edges()

    return {
        "nodes": [dict(n) for n in nodes],
        "edges": edges,
        "user_node_id": getattr(engine, "user_node_id", None),
    }


@router.get("/graph/node/{node_id}")
async def get_node_detail(node_id: str, request: Request):
    """Get detailed node info for the side panel."""
    engine = request.app.state.engine
    node = engine.graph.get_node(node_id)
    if node is None:
        return {"error": "not found"}

    edges = engine.graph.get_edges_for(node_id)
    neighbors = engine.graph.get_neighbors(node_id, depth=1)
    tags = [
        r["tag"]
        for r in engine.db.conn.execute(
            "SELECT tag FROM node_tags WHERE node_id = ?", (node_id,)
        ).fetchall()
    ]

    return {
        "node": node,
        "edges": edges,
        "neighbors": [dict(n) for n in neighbors],
        "tags": tags,
    }


@router.get("/search")
async def search_nodes(q: str, request: Request, limit: int = 20):
    """Search nodes for the UI, returning structured results.

    Uses the same hybrid search (FTS + vector) as the MCP agent path
    so that results are consistent everywhere.
    """
    engine = request.app.state.engine
    if not q.strip():
        return []

    results = engine.recall_search_structured(q, limit=limit)
    # Flatten: return node dicts with _score for the UI
    out = []
    for r in results:
        node = r["node"]
        node["_score"] = r.get("score", 0)
        out.append(node)
    return out


@router.get("/insights")
async def get_insights(request: Request):
    """Get belief evolutions and unresolved tensions for the insights panel."""
    engine = request.app.state.engine
    conn = engine.db.conn

    # Fetch evolved_from edges joined with both nodes
    evolutions_rows = conn.execute(
        """
        SELECT
            e.source_id AS newer_id, e.target_id AS older_id, e.reason,
            n1.title AS newer_title, n1.type AS newer_type, n1.tier AS newer_tier,
            n1.content AS newer_content, n1.created AS newer_created,
            n2.title AS older_title, n2.type AS older_type, n2.tier AS older_tier,
            n2.content AS older_content, n2.created AS older_created
        FROM edges e
        JOIN nodes n1 ON n1.id = e.source_id
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE e.edge_type = 'evolved_from'
        ORDER BY n1.created DESC
        """
    ).fetchall()

    evolutions = [
        {
            "newer": {
                "id": r["newer_id"], "title": r["newer_title"], "type": r["newer_type"],
                "tier": r["newer_tier"], "content": r["newer_content"], "created": r["newer_created"],
            },
            "older": {
                "id": r["older_id"], "title": r["older_title"], "type": r["older_type"],
                "tier": r["older_tier"], "content": r["older_content"], "created": r["older_created"],
            },
            "explanation": r["reason"] or "",
        }
        for r in evolutions_rows
    ]

    # Fetch contradicts edges joined with both nodes
    tensions_rows = conn.execute(
        """
        SELECT
            e.source_id, e.target_id, e.reason,
            n1.title AS title_a, n1.type AS type_a, n1.tier AS tier_a,
            n1.content AS content_a, n1.created AS created_a,
            n2.title AS title_b, n2.type AS type_b, n2.tier AS tier_b,
            n2.content AS content_b, n2.created AS created_b
        FROM edges e
        JOIN nodes n1 ON n1.id = e.source_id
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE e.edge_type = 'contradicts'
        ORDER BY e.created DESC
        """
    ).fetchall()

    tensions = [
        {
            "node_a": {
                "id": r["source_id"], "title": r["title_a"], "type": r["type_a"],
                "tier": r["tier_a"], "content": r["content_a"], "created": r["created_a"],
            },
            "node_b": {
                "id": r["target_id"], "title": r["title_b"], "type": r["type_b"],
                "tier": r["tier_b"], "content": r["content_b"], "created": r["created_b"],
            },
            "explanation": r["reason"] or "",
        }
        for r in tensions_rows
    ]

    return {"evolutions": evolutions, "tensions": tensions}


@router.get("/blind-spots")
async def get_blind_spots(request: Request):
    """Surface memory coverage gaps: isolated nodes, sparse spaces, uncovered types."""
    engine = request.app.state.engine
    conn = engine.db.conn

    # Nodes with zero edges (completely isolated)
    isolated = conn.execute(
        """
        SELECT n.id, n.title, n.type, n.tier, n.space, n.created
        FROM nodes n
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e
            WHERE e.source_id = n.id OR e.target_id = n.id
        )
        ORDER BY n.created DESC
        LIMIT 20
        """
    ).fetchall()

    # Spaces with node counts (NULL space = global)
    sparse_spaces = conn.execute(
        """
        SELECT
            COALESCE(space, '') AS space,
            COUNT(*) AS node_count,
            (
                SELECT COUNT(*) FROM edges e
                JOIN nodes n2 ON (n2.id = e.source_id OR n2.id = e.target_id)
                WHERE COALESCE(n2.space, '') = COALESCE(nodes.space, '')
            ) AS edge_count
        FROM nodes
        GROUP BY space
        HAVING node_count < 5
        ORDER BY node_count ASC
        LIMIT 10
        """
    ).fetchall()

    # Node types that have no core-tier members
    all_types = [
        "fact", "decision", "preference", "event", "person",
        "project", "concept", "procedure", "goal", "observation",
    ]
    type_rows = conn.execute(
        """
        SELECT type, COUNT(*) AS total,
               SUM(CASE WHEN tier = 'core' THEN 1 ELSE 0 END) AS core_count
        FROM nodes
        GROUP BY type
        """
    ).fetchall()
    type_map = {r["type"]: dict(r) for r in type_rows}
    uncovered_types = [
        {"type": t, "total": type_map.get(t, {}).get("total", 0),
         "core_count": type_map.get(t, {}).get("core_count", 0)}
        for t in all_types
        if type_map.get(t, {}).get("core_count", 0) == 0
        and type_map.get(t, {}).get("total", 0) > 0
    ]

    return {
        "isolated_nodes": [dict(r) for r in isolated],
        "sparse_spaces": [dict(r) for r in sparse_spaces],
        "uncovered_types": uncovered_types,
    }


@router.get("/recall-debug")
async def get_recall_debug(request: Request, limit: int = 30):
    """Return recent whisper injection log for debugging recall behaviour."""
    engine = request.app.state.engine
    conn = engine.db.conn

    rows = conn.execute(
        """
        SELECT
            wl.id, wl.session_id, wl.space, wl.prompt_text,
            wl.node_id, wl.score, wl.was_injected, wl.logged_at,
            n.title AS node_title, n.type AS node_type
        FROM whisper_log wl
        LEFT JOIN nodes n ON n.id = wl.node_id
        ORDER BY wl.logged_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    return {"entries": [dict(r) for r in rows]}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time graph updates."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Future: handle real-time subscriptions
            await websocket.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        pass
