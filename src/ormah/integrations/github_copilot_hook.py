"""VS Code Local UserPromptSubmit stdin/stdout contract (not Copilot CLI)."""
import argparse
import asyncio
import json
import sys

from .runtime import whisper


async def handle(payload: object, workspace: str | None = None) -> dict:
    if not isinstance(payload, dict) or payload.get("hook_event_name") != "UserPromptSubmit":
        return {}
    prompt, session = payload.get("prompt"), payload.get("session_id")
    cwd = workspace or payload.get("cwd")
    if not all(isinstance(value, str) and value for value in (prompt, session, cwd)):
        return {}
    text = await whisper("github_copilot", prompt, session, cwd)
    if not text:
        return {}
    return {"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": text,
    }}


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
