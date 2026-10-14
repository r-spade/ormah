"""Cline VS Code HookFactory contract, distinct from SDK CLI file hooks."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from .runtime import whisper


async def handle(payload: object, workspace: str) -> dict:
    output = {"cancel": False, "contextModification": "", "errorMessage": ""}
    if not isinstance(payload, dict) or payload.get("hookName") != "UserPromptSubmit":
        return output
    roots = payload.get("workspaceRoots")
    if (not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], str)
            or Path(roots[0]).resolve() != Path(workspace).resolve()):
        return output
    event = payload.get("userPromptSubmit")
    if not isinstance(event, dict):
        return output
    prompt, session = event.get("prompt"), payload.get("taskId")
    if not all(isinstance(value, str) and value for value in (prompt, session)):
        return output
    output["contextModification"] = (await whisper("cline", prompt, session, workspace))[:48000]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    result = {"cancel": False, "contextModification": "", "errorMessage": ""}
    try:
        result = asyncio.run(handle(json.loads(sys.stdin.read(1024 * 1024)), args.workspace))
    except (OSError, ValueError, TypeError):
        pass
    print(json.dumps(result))


if __name__ == "__main__":
    main()
