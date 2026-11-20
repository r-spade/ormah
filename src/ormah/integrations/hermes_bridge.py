"""Bounded Ormah interpreter bridge for a Hermes native plugin."""
import asyncio
import json
import os
import sys

from .runtime import whisper


async def handle(payload):
    if not isinstance(payload, dict):
        return ""
    prompt, session = payload.get("prompt"), payload.get("session")
    workspace = payload.get("workspace") or os.environ.get("ORMAH_WORKSPACE")
    if (not isinstance(prompt, str) or not isinstance(session, str)
            or workspace is not None and not isinstance(workspace, str)):
        return ""
    return await whisper("hermes", prompt, session, workspace)


def main():
    text = ""
    try:
        text = asyncio.run(handle(json.loads(sys.stdin.read(1024 * 1024))))
    except (OSError, ValueError, TypeError):
        pass
    print(json.dumps({"text": text}))


if __name__ == "__main__":
    main()
