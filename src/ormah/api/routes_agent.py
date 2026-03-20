"""Agent-facing API routes."""

from __future__ import annotations

import json
import logging
import time
from collections import deque

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from ormah.models.node import ConnectRequest, CreateNodeRequest, UpdateNodeRequest
from ormah.models.proposals import ResolveProposalRequest
from ormah.models.search import SearchQuery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# Per-session prompt ring buffer for context-enhanced whisper search.
# Key: session_id, Value: deque of (prompt, timestamp) tuples.
_session_buffers: dict[str, deque[tuple[str, float]]] = {}


class TextResponse(BaseModel):
    text: str
    node_id: str | None = None


@router.post("/remember", response_model=TextResponse)
async def remember(
    req: CreateNodeRequest,
    request: Request,
    default_space: str | None = Query(None, description="Fallback space if not set in body"),
):
    """Store a new memory."""
    if default_space and not req.space:
        req.space = default_space
    engine = request.app.state.engine
    agent_id = getattr(request.state, "agent_id", None)
    node_id, text = engine.remember(req, agent_id=agent_id)
    return TextResponse(text=text, node_id=node_id)


@router.post("/recall", response_model=TextResponse)
async def recall_search(
    req: SearchQuery,
    request: Request,
    default_space: str | None = Query(None, description="Default space for result prioritization"),
):
    """Search memories by query."""
    engine = request.app.state.engine
    text = engine.recall_search(
        req.query,
        limit=req.limit,
        default_space=default_space,
        types=[t.value for t in req.types] if req.types else None,
        tiers=[t.value for t in req.tiers] if req.tiers else None,
        spaces=req.spaces,
        tags=req.tags,
        created_after=req.created_after,
        created_before=req.created_before,
    )
    return TextResponse(text=text)


@router.get("/recall/{node_id}", response_model=TextResponse)
async def recall_node(node_id: str, request: Request):
    """Get a specific memory with its connections."""
    engine = request.app.state.engine
    text = engine.recall_node(node_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return TextResponse(text=text, node_id=node_id)


@router.post("/update/{node_id}", response_model=TextResponse)
async def update_node(node_id: str, req: UpdateNodeRequest, request: Request):
    """Update an existing memory."""
    engine = request.app.state.engine
    text = engine.update_node(node_id, req)
    if text is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return TextResponse(text=text, node_id=node_id)


@router.delete("/recall/{node_id}", response_model=TextResponse)
async def delete_node(node_id: str, request: Request):
    """Delete a memory by ID."""
    engine = request.app.state.engine
    text = engine.delete_node(node_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return TextResponse(text=text, node_id=node_id)


@router.post("/connect", response_model=TextResponse)
async def connect(req: ConnectRequest, request: Request):
    """Create a connection between two memories."""
    engine = request.app.state.engine
    text = engine.connect(req)
    return TextResponse(text=text)


@router.get("/self", response_model=TextResponse)
async def get_self(request: Request):
    """Get the user's identity profile — name, preferences, and personal facts."""
    engine = request.app.state.engine
    text = engine.get_self()
    return TextResponse(text=text)


@router.get("/context", response_model=TextResponse)
async def get_context(
    request: Request,
    space: str | None = Query(None, description="Space to scope context to"),
    task_hint: str | None = Query(None, description="Task hint for adaptive context filtering"),
):
    """Get core memories for system prompt injection."""
    engine = request.app.state.engine
    text = engine.get_context(space=space, task_hint=task_hint)
    return TextResponse(text=text)



@router.post("/whisper", response_model=TextResponse)
async def whisper(request: Request):
    """Build compact whisper context for involuntary recall injection."""
    body = await request.json()
    prompt = body.get("prompt", "")
    space = body.get("space")
    session_id = body.get("session_id", "")
    engine = request.app.state.engine

    # Build recent_prompts from session buffer
    recent_prompts: list[str] | None = None
    if session_id and prompt.strip():
        from ormah.config import settings

        now = time.time()
        gap_seconds = settings.whisper_session_gap_minutes * 60
        buf_size = settings.whisper_context_buffer_size

        buf = _session_buffers.get(session_id)
        if buf is None:
            buf = deque(maxlen=buf_size)
            _session_buffers[session_id] = buf

        # Prune entries older than session gap
        while buf and (now - buf[0][1]) > gap_seconds:
            buf.popleft()

        # Collect recent prompts (excluding the current one)
        if buf:
            recent_prompts = [p for p, _ in buf]

        # Append current prompt to the buffer
        buf.append((prompt.strip(), now))

    text = engine.get_whisper_context(
        prompt=prompt, space=space, recent_prompts=recent_prompts,
    )
    return TextResponse(text=text)


class MarkOutdatedBody(BaseModel):
    reason: str | None = None


@router.post("/outdated/{node_id}", response_model=TextResponse)
async def mark_outdated(node_id: str, request: Request, body: MarkOutdatedBody | None = None):
    """Mark a memory as outdated."""
    engine = request.app.state.engine
    reason = body.reason if body else None
    text = engine.mark_outdated(node_id, reason=reason)
    if text is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return TextResponse(text=text, node_id=node_id)


@router.get("/insights")
async def get_insights(request: Request):
    """Get belief evolutions and conflicting ideas detected by the system."""
    engine = request.app.state.engine
    conn = engine.db.conn

    evolutions = conn.execute(
        """
        SELECT
            e.source_id AS newer_id, e.target_id AS older_id, e.reason,
            n1.title AS newer_title, n1.content AS newer_content, n1.created AS newer_created,
            n2.title AS older_title, n2.content AS older_content, n2.created AS older_created
        FROM edges e
        JOIN nodes n1 ON n1.id = e.source_id
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE e.edge_type = 'evolved_from'
        ORDER BY n1.created DESC
        """
    ).fetchall()

    tensions = conn.execute(
        """
        SELECT
            e.source_id, e.target_id, e.reason,
            n1.title AS title_a, n1.content AS content_a,
            n2.title AS title_b, n2.content AS content_b
        FROM edges e
        JOIN nodes n1 ON n1.id = e.source_id
        JOIN nodes n2 ON n2.id = e.target_id
        WHERE e.edge_type = 'contradicts'
        ORDER BY e.created DESC
        """
    ).fetchall()

    return {
        "evolutions": [dict(r) for r in evolutions],
        "tensions": [dict(r) for r in tensions],
    }


@router.get("/proposals")
async def get_proposals(request: Request):
    """Get pending proposals with enriched source node details."""
    engine = request.app.state.engine
    rows = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE status = 'pending' ORDER BY created DESC"
    ).fetchall()

    result = []
    for r in rows:
        proposal = dict(r)
        # Enrich with source node details
        try:
            node_ids = json.loads(r["source_nodes"])
        except (json.JSONDecodeError, TypeError):
            node_ids = []
        nodes = []
        for nid in node_ids:
            node = engine.db.conn.execute(
                "SELECT id, title, content, type, tier, space, created FROM nodes WHERE id = ?",
                (nid,),
            ).fetchone()
            if node:
                nodes.append(dict(node))
        proposal["nodes"] = nodes

        # Extract merged preview from proposed_action if present
        action = proposal.get("proposed_action", "")
        if "\n---\n" in action:
            parts = action.split("\n---\n")
            proposal["action_summary"] = parts[0].strip()
            proposal["merged_preview"] = parts[1].strip() if len(parts) > 1 else None
        else:
            proposal["action_summary"] = action
            proposal["merged_preview"] = None

        result.append(proposal)
    return result


@router.post("/proposals/{proposal_id}")
async def resolve_proposal(proposal_id: str, body: ResolveProposalRequest, request: Request):
    """Approve or reject a proposal. Executes merge on approval."""
    engine = request.app.state.engine
    from datetime import datetime, timezone

    proposal = engine.db.conn.execute(
        "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
    ).fetchone()
    if proposal is None:
        raise HTTPException(status_code=404, detail="Proposal not found")

    action = body.action.value

    with engine.db.transaction() as conn:
        conn.execute(
            "UPDATE proposals SET status = ?, resolved = ? WHERE id = ?",
            (action, datetime.now(timezone.utc).isoformat(), proposal_id),
        )

    merge_result = None
    conflict_result = None

    if action == "approved" and proposal["type"] == "merge":
        try:
            node_ids = json.loads(proposal["source_nodes"])
            if len(node_ids) == 2:
                merge_result = engine.execute_merge(
                    node_ids[0], node_ids[1], proposal_id=proposal_id
                )
        except Exception:
            logger.exception("Merge failed for proposal %s", proposal_id)
            merge_result = "Merge failed"

    if action == "approved" and proposal["type"] == "conflict":
        try:
            node_ids = json.loads(proposal["source_nodes"])
            if len(node_ids) == 2:
                from ormah.models.node import ConnectRequest, EdgeType
                engine.connect(ConnectRequest(
                    source_id=node_ids[0],
                    target_id=node_ids[1],
                    edge_type=EdgeType.contradicts,
                ))
                conflict_result = f"Created contradicts edge between {node_ids[0][:8]} and {node_ids[1][:8]}"
        except Exception:
            logger.exception("Failed to create contradicts edge for proposal %s", proposal_id)
            conflict_result = "Failed to create contradicts edge"

    return {
        "status": action,
        "proposal_id": proposal_id,
        "merge_result": merge_result,
        "conflict_result": conflict_result,
    }


@router.post("/maintenance")
async def run_maintenance(request: Request):
    """Claude-in-the-loop maintenance: get pending work or apply Claude's decisions.

    Phase 1 — call with no body (or ``{}``):
        Returns four batches of candidates (link, conflict, merge, consolidation).

    Phase 2 — call with ``{"results": {...}}``:
        Claude submits its analysis; ormah applies edges/merges/consolidations.
    """
    body = await request.json()
    engine = request.app.state.engine
    if "results" in body:
        counts = engine.apply_maintenance_results(body["results"])
        return {"status": "applied", "summary": counts}
    batches = engine.get_maintenance_batches()
    return batches


@router.get("/merges")
async def list_merges(
    request: Request,
    limit: int = Query(20, description="Maximum number of merges to return"),
):
    """List recent merge history."""
    engine = request.app.state.engine
    merges = engine.list_merges(limit=limit)
    return merges


@router.get("/audit")
async def list_audit_log(
    request: Request,
    limit: int = Query(20, description="Maximum number of entries to return"),
    node_id: str | None = Query(None, description="Filter by node ID"),
    operation: str | None = Query(None, description="Filter by operation type"),
):
    """List recent audit log entries."""
    engine = request.app.state.engine
    return engine.list_audit_log(limit=limit, node_id=node_id, operation=operation)


@router.post("/merges/{merge_id}/undo", response_model=TextResponse)
async def undo_merge(merge_id: str, request: Request):
    """Undo a merge by ID."""
    engine = request.app.state.engine
    text = engine.undo_merge(merge_id)
    return TextResponse(text=text)
