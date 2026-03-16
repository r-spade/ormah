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
