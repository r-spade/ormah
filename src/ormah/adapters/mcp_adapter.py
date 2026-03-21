"""MCP (Model Context Protocol) server adapter — thin HTTP client."""

from __future__ import annotations

import logging

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ormah.adapters.space_detect import detect_space_from_cwd
from ormah.adapters.tool_schemas import TOOLS
from ormah.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = f"http://localhost:{settings.port}"


def create_mcp_server(base_url: str, default_space: str | None = None) -> Server:
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

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            result = await _dispatch(base_url, name, arguments, default_space=default_space)
            return [TextContent(type="text", text=result)]
        except httpx.ConnectError:
            return [
                TextContent(
                    type="text",
                    text="Ormah server not running. Start it with: ormah serve",
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


async def _dispatch(
    base_url: str, name: str, args: dict, default_space: str | None = None
) -> str:
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
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
                body["tags"] = args["tags"]
            if args.get("about_self"):
                body["about_self"] = True
            if "confidence" in args:
                body["confidence"] = args["confidence"]
            if args.get("links"):
                body["connections"] = [{"target": node_id} for node_id in args["links"]]
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
                body["types"] = args["types"]
            if args.get("spaces"):
                body["spaces"] = args["spaces"]
            if args.get("created_after"):
                body["created_after"] = args["created_after"]
            if args.get("created_before"):
                body["created_before"] = args["created_before"]
            params = {}
            if default_space:
                params["default_space"] = default_space
            resp = await client.post("/agent/recall", json=body, params=params)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "recall_node":
            resp = await client.get(f"/agent/recall/{args['node_id']}")
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "update_memory":
            body = {}
            for key in ("content", "type", "tier", "tags", "title", "space"):
                if key in args and args[key] is not None:
                    body[key] = args[key]
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

        elif name == "get_context":
            params = {}
            if default_space:
                params["space"] = default_space
            if args.get("task_hint"):
                params["task_hint"] = args["task_hint"]
            resp = await client.get("/agent/context", params=params)
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

        elif name == "get_self":
            resp = await client.get("/agent/self")
            if not resp.is_success:
                return _handle_error(resp)
            return resp.json()["text"]

        elif name == "run_maintenance":
            body = {}
            if args.get("results"):
                body["results"] = args["results"]
            resp = await client.post("/agent/maintenance", json=body)
            if not resp.is_success:
                return _handle_error(resp)
            return resp.text

        else:
            return f"Unknown tool: {name}"


async def run_mcp_stdio():
    """Run the MCP server over stdio transport."""
    default_space = detect_space_from_cwd()
    logger.info("Detected project space: %s", default_space or "(global)")

    server = create_mcp_server(_BASE_URL, default_space=default_space)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for MCP stdio server."""
    import asyncio

    from ormah.logging_setup import setup_logging

    setup_logging(log_format=settings.log_format)
    asyncio.run(run_mcp_stdio())


if __name__ == "__main__":
    main()
