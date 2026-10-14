"""PreInvocation injectSteps contract; transcript fixture captured from agy 1.2.9."""
import argparse
import asyncio
import json
import os
import stat
import sys
from pathlib import Path

from .runtime import whisper

MAX_TRANSCRIPT_TAIL = 1024 * 1024


def latest_prompt(payload: dict) -> str:
    session = payload.get("conversationId")
    artifact, transcript = payload.get("artifactDirectoryPath"), payload.get("transcriptPath")
    if not all(isinstance(v, str) and v for v in (session, artifact, transcript)):
        return ""
    root, path = Path(artifact).expanduser().resolve(), Path(transcript).expanduser().resolve()
    if (root.name != session or path.parent != root / ".system_generated/logs"
            or path.name not in ("transcript.jsonl", "transcript_full.jsonl")):
        return ""
    # O_NONBLOCK and fstat avoid waiting on a FIFO/device masquerading as a log.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            return ""
        offset = max(0, info.st_size - MAX_TRANSCRIPT_TAIL)
        stream.seek(offset)
        data = stream.read(MAX_TRANSCRIPT_TAIL)
    if offset:
        data = data.partition(b"\n")[2]
    if not data.endswith(b"\n"):
        return ""  # Partial append: do not fall back to a stale prior prompt.
    for line in reversed(data.decode("utf-8").splitlines()):
        row = json.loads(line)
        if not isinstance(row, dict):
            return ""
        if row.get("type") != "USER_INPUT" or row.get("source") != "USER_EXPLICIT":
            continue
        if row.get("status") != "DONE" or not isinstance(row.get("content"), str):
            return ""
        text = row["content"]
        if not text.startswith("<USER_REQUEST>\n") or "\n</USER_REQUEST>" not in text:
            return ""
        return text[len("<USER_REQUEST>\n"):].split("\n</USER_REQUEST>", 1)[0].strip()
    return ""


async def handle(payload: object, workspace: str | None = None) -> dict:
    workspace = workspace or os.environ.get("ORMAH_WORKSPACE")
    if not isinstance(payload, dict):
        return {}
    if any(type(payload.get(k)) is not int or payload[k] < 0
           for k in ("invocationNum", "initialNumSteps")):
        return {}
    roots = payload.get("workspacePaths")
    if not isinstance(roots, list) or any(not isinstance(root, str) for root in roots):
        return {}
    if workspace and (len(roots) != 1 or Path(roots[0]).resolve() != Path(workspace).resolve()):
        return {}
    try:
        prompt = latest_prompt(payload)
    except (OSError, ValueError, TypeError):
        return {}
    if not prompt:
        return {}
    # User-scope tools and hooks intentionally both use global/default ORMAH_SPACE.
    text = await whisper("antigravity", prompt, payload["conversationId"], workspace)
    return {"injectSteps": [{"ephemeralMessage": text}]} if text else {}


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
