"""MCP (Model Context Protocol) server adapter — thin HTTP client."""

from __future__ import annotations

import json
import logging
import asyncio
import time
import uuid

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ormah.adapters.space_detect import detect_space_from_cwd
from ormah.adapters.tool_schemas import TOOLS
from ormah.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = f"http://localhost:{settings.port}"
_DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0
_MAINTENANCE_TIMEOUT_SECONDS = 300.0
_MAINTENANCE_POLL_INTERVAL_SECONDS = 1.0
_MAINTENANCE_JOB_IDS: dict[str, str] = {}


def _coerce_list(value):
    """Coerce a value to a list — handles clients that serialize arrays as JSON strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return value


def _timeout_for_tool(name: str) -> float:
    """Return an HTTP timeout suitable for the given MCP tool."""
    if name == "run_maintenance":
        return _MAINTENANCE_TIMEOUT_SECONDS
    return _DEFAULT_TOOL_TIMEOUT_SECONDS


def _format_timeout_error(name: str) -> str:
    """Return a user-facing timeout message for a tool call."""
    timeout = _timeout_for_tool(name)
    return f"Error: request to Ormah server timed out after {timeout:.0f}s while running tool '{name}'"


def _maintenance_key(session_id: str | None) -> str:
    return session_id or "default"


def create_mcp_server(
    base_url: str,
    default_space: str | None = None,
    session_id: str | None = None,
) -> Server:
    """Create an MCP server that delegates to the HTTP API."""
    server = Server("ormah")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["parameters"],
            )
            for t in TOOLS
        ]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = await _dispatch(
                base_url,
                name,
                arguments,
                default_space=default_space,
                session_id=session_id,
            )
            return [TextContent(type="text", text=result)]
        except httpx.ConnectError:
            return [
                TextContent(
                    type="text",
                    text="Ormah server not running. Start it with: ormah server start -d",
                )
            ]
        except httpx.ReadTimeout:
            return [
                TextContent(
                    type="text",
                    text=_format_timeout_error(name),
                )
            ]
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return [TextContent(type="text", text=f"Error: {e}")]

    return server


def _handle_error(resp: httpx.Response) -> str:
    """Convert HTTP error responses to user-friendly messages."""
    if resp.status_code == 404:
        detail = resp.json().get("detail", "Not found.")
        return detail
    detail = resp.json().get("detail", resp.text)
    return f"Error: {resp.status_code} {detail}"


def _format_maintenance_batches(batches: dict) -> str:
    """Format Phase 1 maintenance batches as readable text for the agent."""
    lines: list[str] = []
    summary = batches.get("summary", "nothing to process")
    lines.append(f"Maintenance batches ready: {summary}")
    lines.append(
        "Submit ALL evaluated pairs via the edges list in Phase 2 — "
        "use edge_type 'none' for pairs with no relationship (including non-duplicate merge pairs). "
        "This records them so they won't reappear in future runs."
    )
    lines.append("")

    def _pair_block(prefix: str, i: int, c: dict) -> None:
        a, b = c["node_a"], c["node_b"]
        sim = c.get("similarity", 0)
        score = c.get("score")
        score_str = f" score={score:.2f}" if score else ""
        space_str = f"{a.get('space') or 'global'} ↔ {b.get('space') or 'global'}"
        lines.append(f"[{prefix}{i}] sim={sim:.2f}{score_str} | {space_str}")
        lines.append(f"  A [{a['type']}] {a['id']}  \"{a['title']}\"")
        if a.get("content"):
            lines.append(f"     {a['content'][:300]}")
        lines.append(f"  B [{b['type']}] {b['id']}  \"{b['title']}\"")
        if b.get("content"):
            lines.append(f"     {b['content'][:300]}")
        lines.append("")

    link_candidates = batches.get("link_candidates", [])
    if link_candidates:
        lines.append(
            f"## Link Candidates ({len(link_candidates)} pairs)\n"
            "Classify each. Edge types: supports, part_of, depends_on, contradicts, related_to, none\n"
        )
        for i, c in enumerate(link_candidates, 1):
            _pair_block("L", i, c)

    conflict_candidates = batches.get("conflict_candidates", [])
    if conflict_candidates:
        lines.append(
            f"## Conflict Candidates ({len(conflict_candidates)} pairs)\n"
            "Check for contradictions or evolved beliefs. Edge types: contradicts, evolved_from, none\n"
        )
        for i, c in enumerate(conflict_candidates, 1):
            _pair_block("C", i, c)

    merge_candidates = batches.get("merge_candidates", [])
    if merge_candidates:
        lines.append(
            f"## Merge Candidates ({len(merge_candidates)} pairs)\n"
            "Decide if each pair is a duplicate. If yes, provide merged_content + merged_title via merges list.\n"
            "If no, submit as edge_type 'none' via edges list.\n"
        )
        for i, c in enumerate(merge_candidates, 1):
            _pair_block("M", i, c)

    clusters = batches.get("consolidation_clusters", [])
    if clusters:
        lines.append(f"## Consolidation Clusters ({len(clusters)} clusters)\nSynthesize each into one crisp memory.\n")
        for i, cluster in enumerate(clusters, 1):
            lines.append(f"[Cluster {i}] {len(cluster)} nodes")
            for j, n in enumerate(cluster, 1):
                lines.append(f"  {j}. [{n['type']}] {n['id']}  \"{n['title']}\"")
                if n.get("content"):
                    lines.append(f"     {n['content'][:200]}")
            lines.append("")

    if not (link_candidates or conflict_candidates or merge_candidates or clusters):
        lines.append("Nothing to process.")

    return "\n".join(lines)


async def _dispatch(
    base_url: str,
    name: str,
    args: dict,
    default_space: str | None = None,
    session_id: str | None = None,
) -> str:
    async with httpx.AsyncClient(base_url=base_url, timeout=_timeout_for_tool(name)) as client:
        if name == "remember":
            body = {
                "content": args["content"],
                "type": args.get("type", "fact"),
                "tier": args.get("tier", "working"),
            }
            if args.get("title"):
                body["title"] = args["title"]
            if args.get("space"):
                body["space"] = args["space"]
            if args.get("tags"):
                body["tags"] = _coerce_list(args["tags"])
            if args.get("about_self"):
                body["about_self"] = True
            if args.get("space_locked"):
                body["space_locked"] = True
            if "confidence" in args:
                body["confidence"] = args["confidence"]
            if args.get("links"):
                body["connections"] = [{"target": node_id} for node_id in _coerce_list(args["links"])]
            params = {}
            if default_space:
                params["default_space"] = default_space
            resp = await client.post("/agent/remember", json=body, params=params)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "recall":
            body = {"query": args["query"]}
            if args.get("limit"):
                body["limit"] = args["limit"]
            if args.get("types"):
                body["types"] = _coerce_list(args["types"])
            if args.get("spaces"):
                body["spaces"] = _coerce_list(args["spaces"])
            if args.get("created_after"):
                body["created_after"] = args["created_after"]
            if args.get("created_before"):
                body["created_before"] = args["created_before"]
            if session_id:
                body["session_id"] = session_id
            params = {}
            if default_space:
                params["default_space"] = default_space
            resp = await client.post("/agent/recall", json=body, params=params)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "recall_node":
            params = {"session_id": session_id} if session_id else {}
            resp = await client.get(f"/agent/recall/{args['node_id']}", params=params)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "update_memory":
            body = {}
            for key in ("content", "type", "tier", "title", "space"):
                if key in args and args[key] is not None:
                    body[key] = args[key]
            if args.get("tags") is not None:
                body["tags"] = _coerce_list(args["tags"])
            resp = await client.post(f"/agent/update/{args['node_id']}", json=body)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "connect_memories":
            body = {
                "source_id": args["source_id"],
                "target_id": args["target_id"],
            }
            if "edge" in args:
                body["edge"] = args["edge"]
            if "weight" in args:
                body["weight"] = args["weight"]
            resp = await client.post("/agent/connect", json=body)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "mark_outdated":
            body = {}
            if args.get("reason"):
                body["reason"] = args["reason"]
            resp = await client.post(
                f"/agent/outdated/{args['node_id']}", json=body if body else None
            )
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "list_proposals":
            resp = await client.get("/agent/proposals")
            if not resp.is_success:
                return _handle_error(resp)
            rows = resp.json()
            if not rows:
                return "No pending proposals."
            lines = []
            for r in rows:
                lines.append(
                    f"[{r['type']}] {r['proposed_action']}\n"
                    f"  ID: {r['id']}\n"
                    f"  Reason: {r.get('reason') or 'N/A'}\n"
                    f"  Nodes: {r['source_nodes']}\n"
                    f"  Created: {r['created']}"
                )
            return "\n\n".join(lines)

        elif name == "resolve_proposal":
            body = {"action": args["action"]}
            resp = await client.post(
                f"/agent/proposals/{args['proposal_id']}", json=body
            )
            if not resp.is_success:
                return _handle_error(resp)
            data = resp.json()
            status = data["status"]
            pid = data["proposal_id"]
            merge_result = data.get("merge_result")
            if merge_result:
                return str(merge_result)
            return f"Proposal {pid[:8]} {status}."

        elif name == "list_merges":
            params = {}
            if args.get("limit"):
                params["limit"] = args["limit"]
            resp = await client.get("/agent/merges", params=params)
            if not resp.is_success:
                return _handle_error(resp)
            merges = resp.json()
            if not merges:
                return "No merge history."
            lines = []
            for m in merges:
                status = "UNDONE" if m["undone_at"] else "active"
                lines.append(
                    f"[{status}] Kept: {m['kept_node_id'][:8]}  Removed: {m['removed_node_id'][:8]}\n"
                    f"  Merge ID: {m['id']}\n"
                    f"  Merged at: {m['merged_at']}"
                )
            return "\n\n".join(lines)

        elif name == "list_audit_log":
            params = {}
            if args.get("limit"):
                params["limit"] = args["limit"]
            if args.get("node_id"):
                params["node_id"] = args["node_id"]
            if args.get("operation"):
                params["operation"] = args["operation"]
            resp = await client.get("/agent/audit", params=params)
            if not resp.is_success:
                return _handle_error(resp)
            entries = resp.json()
            if not entries:
                return "No audit log entries."
            lines = []
            for e in entries:
                lines.append(
                    f"[{e['operation']}] Node: {e['node_id'][:8]}...\n"
                    f"  ID: {e['id']}\n"
                    f"  Detail: {e.get('detail') or 'N/A'}\n"
                    f"  Performed at: {e['performed_at']}"
                )
            return "\n\n".join(lines)

        elif name == "undo_merge":
            resp = await client.post(f"/agent/merges/{args['merge_id']}/undo")
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "submit_feedback":
            body = {
                "node_id": args["node_id"],
                "signal": args["signal"],
                "source": args.get("source", "explicit"),
            }
            if args.get("whisper_log_id") is not None:
                body["whisper_log_id"] = args["whisper_log_id"]
            resp = await client.post("/agent/feedback", json=body)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "run_maintenance":
            body = {}
            key = _maintenance_key(session_id)
            if args.get("job_id"):
                body["job_id"] = args["job_id"]
            elif key in _MAINTENANCE_JOB_IDS:
                body["job_id"] = _MAINTENANCE_JOB_IDS[key]
            if args.get("results"):
                body["results"] = args["results"]
            resp = await client.post("/agent/maintenance", json=body)
            if not resp.is_success:
                return _handle_error(resp)
            data = resp.json()
            job_id = data.get("job_id")
            if job_id:
                _MAINTENANCE_JOB_IDS[key] = job_id
            data = await _poll_maintenance_until_ready(
                client,
                data,
                expect_apply_summary="results" in args,
            )
            if data.get("job_id"):
                _MAINTENANCE_JOB_IDS[key] = data["job_id"]
            if "results" in args:
                _MAINTENANCE_JOB_IDS.pop(key, None)
                return json.dumps({"status": "applied", "summary": data.get("apply_summary", {})})
            batches = data.get("batches")
            if isinstance(batches, dict):
                return _format_maintenance_batches(batches)
            return resp.text

        else:
            return f"Unknown tool: {name}"


async def _poll_maintenance_until_ready(
    client: httpx.AsyncClient,
    initial: dict,
    *,
    expect_apply_summary: bool,
) -> dict:
    """Poll maintenance status until the requested phase is ready."""
    deadline = time.monotonic() + _MAINTENANCE_TIMEOUT_SECONDS
    data = initial
    job_id = data.get("job_id")

    while True:
        status = data.get("status")
        if expect_apply_summary:
            if status == "completed" and isinstance(data.get("apply_summary"), dict):
                return data
        else:
            if status == "awaiting_results" and isinstance(data.get("batches"), dict):
                return data
        if status == "failed":
            error = data.get("last_error") or "maintenance job failed"
            raise RuntimeError(error)
        if time.monotonic() >= deadline:
            raise httpx.ReadTimeout(_format_timeout_error("run_maintenance"))
        await _sleep_for_poll_interval()
        params = {"job_id": job_id} if job_id else None
        resp = await client.get("/agent/maintenance", params=params)
        if not resp.is_success:
            raise httpx.HTTPStatusError(
                f"Unexpected maintenance polling status: {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()


async def _sleep_for_poll_interval() -> None:
    await asyncio.sleep(_MAINTENANCE_POLL_INTERVAL_SECONDS)


async def run_mcp_stdio():
    """Run the MCP server over stdio transport."""
    session_id = str(uuid.uuid4())
    default_space = detect_space_from_cwd()
    logger.info(
        "Detected project space: %s (mcp session %s)",
        default_space or "(global)",
        session_id[:8],
    )

    server = create_mcp_server(
        _BASE_URL,
        default_space=default_space,
        session_id=session_id,
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for MCP stdio server."""
    import asyncio
    import logging

    from ormah.logging_setup import setup_logging

    setup_logging(
        log_format=settings.log_format,
        level=getattr(logging, settings.log_level),
    )
    asyncio.run(run_mcp_stdio())


if __name__ == "__main__":
    main()
