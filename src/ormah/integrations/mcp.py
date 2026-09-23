"""Host-aware stdio entry point using the existing Ormah MCP tool contracts."""
from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from mcp.server.stdio import stdio_server

from ormah.adapters.mcp_adapter import create_mcp_server
from .common import instructions
from .runtime import base_url, headers, session_key, space_for


async def run(host: str, workspace: str | None) -> None:
    workspace = workspace or os.environ.get("ORMAH_WORKSPACE")
    # No guessed IDE process cwd: hosts must pass their actual workspace, or use
    # ORMAH_SPACE. An unbound user-level connection is intentionally global.
    server = create_mcp_server(
        base_url(), default_space=await space_for(workspace),
        session_id=session_key(host, f"mcp:{uuid.uuid4()}", workspace),
        headers=headers(), instructions=instructions(),
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--workspace")
    args = parser.parse_args()
    asyncio.run(run(args.host, args.workspace))


if __name__ == "__main__":
    main()
