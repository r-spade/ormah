"""VS Code Local and shared Copilot runtime stdin/stdout hook contracts."""
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


async def handle_transformed(payload: object, workspace: str | None = None) -> dict:
    """Append memory without replacing any existing model-facing content."""
    if not isinstance(payload, dict):
        return {}
    prompt, session = payload.get("prompt"), payload.get("sessionId")
    original = payload.get("transformedPrompt")
    cwd = workspace or payload.get("cwd")
    if not all(isinstance(value, str) and value.strip()
               for value in (prompt, session, cwd, original)):
        return {}
    # The shared runtime can also fire the legacy event while discarding its
    # output. Its retrieval must not mark this event's session as already served.
    text = await whisper("github_copilot_runtime", prompt, session, cwd)
    if not text:
        return {}
    return {"modifiedTransformedPrompt": original + "\n\n" + text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    parser.add_argument("--event", choices=["UserPromptSubmit", "userPromptTransformed"],
                        default="UserPromptSubmit")
    args = parser.parse_args()
    result = {}
    try:
        handler = handle_transformed if args.event == "userPromptTransformed" else handle
        result = asyncio.run(handler(json.loads(sys.stdin.read(1024 * 1024)), args.workspace))
    except (OSError, ValueError, TypeError):
        pass
    print(json.dumps(result))


if __name__ == "__main__":
    main()
