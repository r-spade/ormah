"""Kiro UserPromptSubmit emits plain stdout context with exit zero."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .runtime import whisper


async def handle(payload: object, workspace: str | None = None) -> str:
    if not isinstance(payload, dict) or payload.get("hook_event_name") not in ("UserPromptSubmit", "userPromptSubmit"):
        return ""
    prompt, session, cwd = (payload.get(key) for key in ("prompt", "session_id", "cwd"))
    if not all(isinstance(value, str) and value.strip() for value in (prompt, session, cwd)):
        return ""
    if workspace and Path(cwd).resolve() != Path(workspace).resolve():
        return ""
    # Match the global MCP default; do not guess the IDE server's working dir.
    selected = workspace or os.environ.get("ORMAH_WORKSPACE")
    return await whisper("kiro", prompt, session, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    args = parser.parse_args()
    result = ""
    try:
        result = asyncio.run(handle(json.loads(sys.stdin.read(1024 * 1024)), args.workspace))
    except (OSError, ValueError, TypeError):
        pass
    # This host injects stdout itself, not a hookSpecificOutput JSON envelope.
    if result:
        print(result)


if __name__ == "__main__":
    main()
