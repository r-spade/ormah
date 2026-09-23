"""Private stdin bridge used by the packaged Kilo extension."""
import asyncio
import json
import sys

from .runtime import whisper


def main():
    text = ""
    try:
        payload = json.loads(sys.stdin.read(1024 * 1024))
        if (isinstance(payload, dict) and isinstance(payload.get("prompt"), str)
                and isinstance(payload.get("session"), str)
                and isinstance(payload.get("workspace"), str)):
            text = asyncio.run(whisper("kilo", payload["prompt"], payload["session"],
                                       payload["workspace"]))
    except (OSError, ValueError, TypeError):
        pass
    print(json.dumps({"text": text}))


if __name__ == "__main__":
    main()
