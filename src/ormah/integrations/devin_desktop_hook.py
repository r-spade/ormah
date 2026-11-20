"""Devin Local UserPromptSubmit contract; deliberately not Cascade's pre_user_prompt."""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .runtime import whisper


async def handle(payload: object, workspace: str | None = None) -> dict:
    if not isinstance(payload, dict) or payload.get("hook_event_name") != "UserPromptSubmit":
        return {}
    prompt, session = payload.get("prompt"), payload.get("session_id")
    if not all(isinstance(value, str) and value.strip() for value in (prompt, session)):
        return {}
    if workspace:
        actual = os.environ.get("DEVIN_PROJECT_DIR")
        if not actual or Path(actual).resolve() != Path(workspace).resolve():
            return {}
    # User-level MCP has no documented workspace substitution. Keep both paths
    # global unless the operator explicitly supplies ORMAH_WORKSPACE/ORMAH_SPACE.
    selected = workspace or os.environ.get("ORMAH_WORKSPACE")
    text = await whisper("devin_desktop", prompt, session, selected)
    return {"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": text,
    }} if text else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    args = parser.parse_args()
    result = {}
    try:
        result = asyncio.run(handle(json.loads(sys.stdin.read(1024 * 1024)), args.workspace))
    except (OSError, ValueError, TypeError):
        pass
    print(json.dumps(result))


if __name__ == "__main__":
    main()
