"""Private OpenClaw whisper bridge; tools use the existing persistent MCP server."""
import asyncio
import json
import sys

from .runtime import whisper


def main():
    result = ""
    try:
        data = json.loads(sys.stdin.read(1024 * 1024))
        if isinstance(data, dict) and all(isinstance(data.get(k), str)
                                          for k in ("prompt", "session", "workspace")):
            result = asyncio.run(whisper("openclaw", data["prompt"], data["session"], data["workspace"]))
    except (OSError, ValueError, TypeError):
        pass
    print(json.dumps({"text": result}))


if __name__ == "__main__":
    main()
