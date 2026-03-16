"""CLI adapter — thin synchronous HTTP client for terminal access to ormah."""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

from pathlib import Path

from ormah.adapters.space_detect import detect_space_from_dir, resolve_space
from ormah.config import settings

BASE = f"http://localhost:{settings.port}"


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=30.0)


def _api(fn):
    """Run fn(), catching connection and HTTP errors."""
    try:
        return fn()
    except httpx.ConnectError:
        print("Ormah server not running. Start it with: ormah", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)


def cmd_context(args):
    space = resolve_space(args.space)
    params = {}
    if space:
        params["space"] = space
    if args.task:
        params["task_hint"] = args.task

    def call():
        with _client() as c:
            r = c.get("/agent/context", params=params)
            r.raise_for_status()
            print(r.json()["text"])

    _api(call)


def cmd_recall(args):
    space = resolve_space(args.space)
    body: dict = {"query": args.query}
    if args.limit:
        body["limit"] = args.limit
    if args.types:
        body["types"] = args.types.split(",")
    params = {"default_space": space} if space else {}

    def call():
        with _client() as c:
            r = c.post("/agent/recall", json=body, params=params)
            r.raise_for_status()
            if args.json:
                print(json.dumps(r.json(), indent=2))
            else:
                print(r.json()["text"])

    _api(call)


def cmd_remember(args):
    content = sys.stdin.read() if args.content == "-" else args.content
    space = resolve_space(args.space)
    body: dict = {
        "content": content,
        "type": args.type,
        "tier": args.tier,
    }
    if args.title:
        body["title"] = args.title
    if args.tags:
        body["tags"] = args.tags.split(",")
    if args.about_self:
        body["about_self"] = True
    if space:
        body["space"] = space
    params = {"default_space": space} if space else {}

    def call():
        with _client() as c:
            r = c.post("/agent/remember", json=body, params=params)
            r.raise_for_status()
            print(r.json()["text"])

    _api(call)


def cmd_ingest(args):
    if args.source == "-":
        content = sys.stdin.read()
    else:
        with open(args.source) as f:
            content = f.read()
    space = resolve_space(args.space)
    body: dict = {"content": content}
    params = {"default_space": space} if space else {}

    def call():
        with _client() as c:
            r = c.post("/ingest/conversation", json=body, params=params)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                print(data["result"], file=sys.stderr)
                sys.exit(1)
            count = data.get("extracted", 0)
            if count == 0:
                print("No new memories extracted from the conversation.")
            else:
                lines = [f"Extracted {count} memories:"]
                for mem in data.get("memories", []):
                    lines.append(f"  - {mem['title']} (ID: {mem['node_id'][:8]}...)")
                print("\n".join(lines))

    _api(call)


def cmd_ingest_session(args):
    from ormah.transcript.parser import parse_transcript

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    result = parse_transcript(path)

    if result.user_turn_count < args.min_turns:
        print(
            f"Skipped: only {result.user_turn_count} user turns "
            f"(minimum: {args.min_turns})"
        )
        return

    pct = (result.cleaned_chars / result.total_chars * 100) if result.total_chars else 0
    print(
        f"Parsed: {result.user_turn_count} turns, "
        f"{result.cleaned_chars} chars ({pct:.1f}% of {result.total_chars})"
    )

    if not result.conversation.strip():
        print("No conversation text extracted.")
        return

    space = resolve_space(args.space)
    body: dict = {"content": result.conversation}
    params: dict = {}
    if space:
        params["default_space"] = space
    if args.dry_run:
        params["dry_run"] = "true"

    def call():
        with _client() as c:
            r = c.post(
                "/ingest/conversation",
                json=body,
                params=params,
                timeout=120.0,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "error":
                print(data["result"], file=sys.stderr)
                sys.exit(1)
            count = data.get("extracted", 0)
            if count == 0:
                print("No memories extracted from session.")
            else:
                label = "Would extract" if args.dry_run else "Extracted"
                lines = [f"{label} {count} memories:"]
                for mem in data.get("memories", []):
                    title = mem.get("title", "untitled")
                    node_id = mem.get("node_id")
                    if node_id:
                        lines.append(f"  - {title} (ID: {node_id[:8]}...)")
                    else:
                        mem_type = mem.get("type", "fact")
                        lines.append(f"  - [{mem_type}] {title}")
                print("\n".join(lines))

    _api(call)


def cmd_node(args):
    def call():
        with _client() as c:
            r = c.get(f"/agent/recall/{args.id}")
            r.raise_for_status()
            if args.json:
                print(json.dumps(r.json(), indent=2))
            else:
                print(r.json()["text"])

    _api(call)


def cmd_status(args):
    def call():
        with _client() as c:
            r = c.get("/admin/health")
            r.raise_for_status()
            data = r.json()
            print(f"Status: {data['status']}")
            if "jobs" in data:
                for name, info in data["jobs"].items():
                    state = info.get("state", "unknown")
                    last = info.get("last_run")
                    line = f"  {name}: {state}"
                    if last:
                        line += f" (last: {last})"
                    print(line)

    _api(call)


def _whisper_client() -> httpx.Client:
    """Client with short timeout for whisper hook — fail fast, never block the user."""
    return httpx.Client(base_url=BASE, timeout=5.0)


def cmd_whisper_inject(args):
    """Read hook JSON from stdin, fetch context, output additionalContext."""
    try:
        raw = sys.stdin.read()
        hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Malformed input — exit silently
        sys.exit(0)

    prompt = hook_data.get("prompt", "")
    cwd = hook_data.get("cwd", "")

    if not prompt:
        sys.exit(0)

    # Resolve space from the hook-provided cwd, not our process cwd
    space = detect_space_from_dir(cwd) if cwd else None

    session_id = hook_data.get("session_id", "")

    body: dict = {"prompt": prompt}
    if space:
        body["space"] = space
    if session_id:
        body["session_id"] = session_id

    try:
        with _whisper_client() as c:
            r = c.post("/agent/whisper", json=body)
            r.raise_for_status()
            text = r.json().get("text", "")
    except Exception:
        # Server down, timeout, or any error — exit silently
        sys.exit(0)

    if not text.strip():
        text = ""

    # Track prompt count per session (used by nudge and periodic extraction)
    count = 0
    if session_id:
        cursors = _load_cursors()
        count_key = f"nudge:{session_id}"
        count = cursors.get(count_key, 0) + 1
        cursors[count_key] = count
        _save_cursors(cursors)

        # Append nudge at interval
        if (settings.whisper_nudge_interval > 0
                and count % settings.whisper_nudge_interval == 0):
            nudge = (
                "\n\n---\n"
                "Remember to use ormah's remember tool to store decisions, "
                "preferences or noteworthy facts from this conversation."
            )
            text += nudge

    if not text.strip():
        # Still trigger periodic extraction even when no inject text
        if (session_id
                and settings.whisper_out_enabled
                and settings.whisper_out_interval > 0
                and count % settings.whisper_out_interval == 0):
            transcript = _resolve_transcript_path(session_id)
            if transcript:
                _spawn_background_store(transcript, cwd, session_id)
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": text,
        }
    }
    print(json.dumps(output))

    # Periodic background extraction — after output is printed
    if (session_id
            and settings.whisper_out_enabled
            and settings.whisper_out_interval > 0
            and count % settings.whisper_out_interval == 0):
        transcript = _resolve_transcript_path(session_id)
        if transcript:
            _spawn_background_store(transcript, cwd, session_id)

    sys.exit(0)


def _resolve_transcript_path(session_id: str) -> Path | None:
    """Find Claude Code transcript JSONL for a session ID."""
    claude_projects = Path("~/.claude/projects").expanduser()
    if not claude_projects.is_dir():
        return None
    matches = list(claude_projects.glob(f"*/{session_id}.jsonl"))
    return matches[0] if matches else None


def _spawn_background_store(transcript_path: Path, cwd: str, session_id: str) -> None:
    """Fire-and-forget: spawn 'ormah whisper store' in background."""
    import shutil
    import subprocess

    ormah_bin = shutil.which("ormah") or "ormah"
    hook_json = json.dumps({
        "transcript_path": str(transcript_path),
        "cwd": cwd,
        "session_id": session_id,
    })
    try:
        subprocess.Popen(
            [ormah_bin, "whisper", "store"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        ).communicate(input=hook_json.encode(), timeout=1)
    except Exception:
        pass  # fire and forget


def _whisper_store_client() -> httpx.Client:
    """Client with longer timeout for whisper-out — extraction can take 30s+."""
    return httpx.Client(base_url=BASE, timeout=60.0)


_WHISPER_CURSOR_DIR = Path(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))) / "ormah"
_WHISPER_CURSOR_FILE = _WHISPER_CURSOR_DIR / "whisper-cursors.json"


def _load_cursors() -> dict:
    try:
        return json.loads(_WHISPER_CURSOR_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cursors(cursors: dict) -> None:
    try:
        _WHISPER_CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        _WHISPER_CURSOR_FILE.write_text(json.dumps(cursors))
    except OSError:
        pass


def cmd_whisper_store(args):
    """PreCompact/SessionEnd hook handler: extract and store memories from transcript."""
    try:
        raw = sys.stdin.read()
        hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    transcript_path = hook_data.get("transcript_path", "")
    cwd = hook_data.get("cwd", "")
    session_id = hook_data.get("session_id", "")

    if not transcript_path:
        sys.exit(0)

    path = Path(transcript_path)
    if not path.exists():
        sys.exit(0)

    # Load cursor — only process new content since last extraction
    cursors = _load_cursors()
    cursor_key = session_id or str(path)
    start_offset = cursors.get(cursor_key, 0)

    # Skip if file hasn't grown since last extraction
    if start_offset >= path.stat().st_size:
        sys.exit(0)

    from ormah.transcript.parser import parse_transcript

    try:
        result = parse_transcript(path, start_offset=start_offset)
    except Exception:
        sys.exit(0)

    min_turns = settings.whisper_out_min_turns
    if result.user_turn_count < min_turns:
        sys.exit(0)

    if not result.conversation.strip():
        sys.exit(0)

    space = detect_space_from_dir(cwd) if cwd else None

    body: dict = {"content": result.conversation}
    params: dict = {"extra_tags": "whisper-out"}
    if space:
        params["default_space"] = space

    try:
        with _whisper_store_client() as c:
            r = c.post("/ingest/conversation", json=body, params=params)
            r.raise_for_status()
    except Exception:
        # Server down, timeout, or any error — exit silently, never block compaction
        sys.exit(0)

    # Update cursor only after successful extraction
    cursors[cursor_key] = result.end_offset
    _save_cursors(cursors)

    sys.exit(0)


def cmd_whisper_setup(args):
    """Generate Claude Code hook config for the whisper hook."""
    import shutil

    ormah_bin = shutil.which("ormah") or "ormah"
    hooks: dict = {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{ormah_bin} whisper inject",
                        "timeout": 10,
                    }
                ]
            }
        ],
    }
    hooks["PreCompact"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{ormah_bin} whisper store",
                    "timeout": 300,
                    "async": True,
                }
            ]
        }
    ]
    hooks["SessionEnd"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{ormah_bin} whisper store",
                    "timeout": 300,
                }
            ]
        }
    ]
    hook_config = {"hooks": hooks}

    if args.glob:
        settings_path = os.path.expanduser("~/.claude/settings.json")
    else:
        settings_path = os.path.join(".claude", "settings.local.json")

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(settings_path)), exist_ok=True)

    # Merge with existing settings if file exists
    existing = {}
    if os.path.exists(settings_path):
        try:
            with open(settings_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    existing.update(hook_config)

    with open(settings_path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")

    print(f"Wrote hook config to {settings_path}")
    print(json.dumps(hook_config, indent=2))


def main():
    p = argparse.ArgumentParser(
        prog="ormah-cli",
        description="Terminal interface to the ormah memory system.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- context ---
    ctx = sub.add_parser("context", help="Get context for piping into agent prompts")
    ctx.add_argument("--task", help="Task hint to filter context")
    ctx.add_argument("--space", help="Override space detection")
    ctx.set_defaults(func=cmd_context)

    # --- recall ---
    rec = sub.add_parser("recall", help="Search memories")
    rec.add_argument("query", help="Search query")
    rec.add_argument("--limit", type=int, help="Max results")
    rec.add_argument("--types", help="Comma-separated memory types to filter")
    rec.add_argument("--json", action="store_true", help="Output raw JSON")
    rec.add_argument("--space", help="Override space detection")
    rec.set_defaults(func=cmd_recall)

    # --- remember ---
    rem = sub.add_parser("remember", help="Store a memory")
    rem.add_argument("content", help="Memory content (use - for stdin)")
    rem.add_argument("--title", help="Short title")
    rem.add_argument("--type", default="fact", help="Memory type (default: fact)")
    rem.add_argument("--tier", default="working", help="Tier: core/working/archival (default: working)")
    rem.add_argument("--tags", help="Comma-separated tags")
    rem.add_argument("--about-self", action="store_true", help="Link to user identity")
    rem.add_argument("--space", help="Override space detection")
    rem.set_defaults(func=cmd_remember)

    # --- ingest ---
    ing = sub.add_parser("ingest", help="Ingest a conversation log")
    ing.add_argument("source", help="File path or - for stdin")
    ing.add_argument("--space", help="Override space detection")
    ing.set_defaults(func=cmd_ingest)

    # --- ingest-session ---
    ings = sub.add_parser("ingest-session", help="Ingest a Claude Code JSONL session transcript")
    ings.add_argument("path", help="Path to Claude Code JSONL transcript")
    ings.add_argument("--dry-run", action="store_true", help="Extract but don't store — print what would be ingested")
    ings.add_argument("--space", help="Override project space (default: auto-detect from cwd)")
    ings.add_argument("--min-turns", type=int, default=5, help="Minimum user turns with text to consider worth ingesting (default: 5)")
    ings.set_defaults(func=cmd_ingest_session)

    # --- node ---
    nd = sub.add_parser("node", help="Get a specific memory by ID")
    nd.add_argument("id", help="Memory UUID")
    nd.add_argument("--json", action="store_true", help="Output raw JSON")
    nd.set_defaults(func=cmd_node)

    # --- status ---
    st = sub.add_parser("status", help="Check server health")
    st.set_defaults(func=cmd_status)

    # --- whisper ---
    wh = sub.add_parser("whisper", help="Claude Code hook integration")
    wh_sub = wh.add_subparsers(dest="whisper_cmd", required=True)

    wh_inject = wh_sub.add_parser("inject", help="Hook handler: inject memories into prompt")
    wh_inject.set_defaults(func=cmd_whisper_inject)

    wh_store = wh_sub.add_parser("store", help="Hook handler: extract memories before compaction")
    wh_store.set_defaults(func=cmd_whisper_store)

    wh_setup = wh_sub.add_parser("setup", help="Generate Claude Code hook config")
    wh_setup.add_argument("--global", dest="glob", action="store_true", help="Write to global ~/.claude/settings.json instead of local")
    wh_setup.set_defaults(func=cmd_whisper_setup)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
