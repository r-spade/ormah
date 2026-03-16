"""One-shot interactive setup: hooks, MCP registration, server auto-start, LLM config."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import httpx

from importlib import resources

from ormah.config import settings
from ormah.server_manager import (
    get_ormah_bin_path,
    install_autostart,
    is_server_running,
    wait_for_server,
)

ENV_DIR = Path.home() / ".config" / "ormah"
ENV_PATH = ENV_DIR / ".env"

CLAUDE_MD_SENTINEL_START = "<!-- ormah:start -->"
CLAUDE_MD_SENTINEL_END = "<!-- ormah:end -->"

# Provider definitions: (display name, provider, env var for API key, default model)
LLM_PROVIDERS = [
    ("Anthropic (Claude)", "litellm", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
    ("OpenAI", "litellm", "OPENAI_API_KEY", "gpt-4.1-mini"),
    ("Google Gemini", "litellm", "GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
    ("Ollama (local)", "ollama", None, "llama3.2"),
    ("None", "none", None, None),
]


def _merge_json_file(path: str, updates: dict) -> None:
    """Read a JSON file, deep-merge updates, and write back."""
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(existing.get(key), dict):
            existing[key].update(value)
        else:
            existing[key] = value

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def configure_claude_hooks(ormah_bin: str) -> None:
    """Write Claude Code hook config to global settings using absolute paths."""
    settings_path = os.path.expanduser("~/.claude/settings.json")

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

    _merge_json_file(settings_path, {"hooks": hooks})
    print("  Ormah now whispers memories to Claude before every message.")
    print("  Memories are auto-extracted on compaction and session end.")


def configure_claude_code_mcp(ormah_bin: str) -> None:
    """Register ormah MCP server in Claude Code user config.

    Uses ``claude mcp add`` when available (correct format, user scope).
    Falls back to direct JSON editing if the claude CLI is not on PATH.
    """
    # Prefer the official CLI — it writes the correct format
    claude_bin = shutil.which("claude")
    if claude_bin:
        try:
            result = subprocess.run(
                [claude_bin, "mcp", "add", "ormah", "--scope", "user", "--", ormah_bin, "mcp"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print("  Registered as an MCP tool \u2014 Claude can store and recall memories.")
                return
            # Already registered — remove and re-add to update the command path
            if "already exists" in result.stderr or "already exists" in result.stdout:
                subprocess.run(
                    [claude_bin, "mcp", "remove", "ormah", "--scope", "user"],
                    capture_output=True, timeout=10,
                )
                result2 = subprocess.run(
                    [claude_bin, "mcp", "add", "ormah", "--scope", "user", "--", ormah_bin, "mcp"],
                    capture_output=True, text=True, timeout=10,
                )
                if result2.returncode == 0:
                    print("  Registered as an MCP tool \u2014 Claude can store and recall memories.")
                    return
        except Exception:
            pass

    # Fallback: write directly (no "type" field — stdio is the default)
    config_path = os.path.expanduser("~/.claude.json")
    mcp_entry = {
        "ormah": {
            "command": ormah_bin,
            "args": ["mcp"],
        }
    }
    _merge_json_file(config_path, {"mcpServers": mcp_entry})
    print("  Registered as an MCP tool \u2014 Claude can store and recall memories.")


def configure_claude_desktop(ormah_bin: str) -> None:
    """Register ormah MCP server in Claude Desktop config (if installed)."""
    config_path = os.path.expanduser(
        "~/Library/Application Support/Claude/claude_desktop_config.json"
    )

    if not os.path.exists(os.path.dirname(config_path)):
        return

    mcp_entry = {
        "ormah": {
            "command": ormah_bin,
            "args": ["mcp"],
        }
    }

    _merge_json_file(config_path, {"mcpServers": mcp_entry})
    print("  Connected to Claude Desktop too.")


def install_claude_md() -> None:
    """Install ormah instructions into ~/.claude/CLAUDE.md."""
    target = Path.home() / ".claude" / "CLAUDE.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    # Load instructions from package data
    instructions = resources.files("ormah").joinpath("instructions.md").read_text()
    block = f"{CLAUDE_MD_SENTINEL_START}\n{instructions}{CLAUDE_MD_SENTINEL_END}\n"

    existing = target.read_text() if target.exists() else ""

    if CLAUDE_MD_SENTINEL_START in existing and CLAUDE_MD_SENTINEL_END in existing:
        # Replace existing block
        start = existing.index(CLAUDE_MD_SENTINEL_START)
        end = existing.index(CLAUDE_MD_SENTINEL_END) + len(CLAUDE_MD_SENTINEL_END)
        # Consume trailing newline if present
        if end < len(existing) and existing[end] == "\n":
            end += 1
        updated = existing[:start] + block + existing[end:]
    elif existing:
        # Append with blank line separator
        updated = existing.rstrip("\n") + "\n\n" + block
    else:
        # New file
        updated = block

    target.write_text(updated)
    print("  Added ormah instructions to ~/.claude/CLAUDE.md")


def _read_env_file() -> dict[str, str]:
    """Read existing .env file into a dict."""
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def _write_env_file(env: dict[str, str]) -> None:
    """Write env dict to the global config file with secure permissions."""
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in env.items():
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    os.chmod(ENV_PATH, 0o600)


def _prompt_choice(prompt: str, options: list[str], allow_skip: bool = False) -> int | None:
    """Show a numbered menu and return the selected index (0-based), or None for skip."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"    {i}. {option}")
    if allow_skip:
        print(f"    {len(options) + 1}. Skip for now")

    while True:
        try:
            raw = input("\n  Choice: ").strip()
            if not raw:
                continue
            choice = int(raw)
            if allow_skip and choice == len(options) + 1:
                return None
            if 1 <= choice <= len(options):
                return choice - 1
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def configure_identity() -> str | None:
    """Ask the user for their name. Returns the name or None if skipped."""
    try:
        name = input("  What should ormah call you? ").strip()
    except EOFError:
        name = ""

    if not name:
        print("  Skipped \u2014 ormah will learn your name naturally.")
        return None
    return name


def seed_identity(name: str) -> None:
    """Store the user's name in ormah via the running server."""
    print("\n  Seeding your first memory...")
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"http://localhost:{settings.port}/agent/remember",
                json={
                    "content": f"User's name is {name}",
                    "type": "person",
                    "tier": "core",
                    "about_self": True,
                    "title": "User's name",
                },
            )
        print("  Ormah now knows your name.")
    except Exception:
        print("  Could not seed identity \u2014 server may not be ready yet.")
        print("  No worries, ormah will learn your name naturally.")


_MONTHLY_COST_HINT: dict[str, str] = {
    "claude-haiku": "~$1-3/month with typical use",
    "gpt-4.1-mini": "~$1-3/month with typical use",
    "gpt-4o-mini": "~$0.50-1/month with typical use",
    "gemini": "~$0.25-1/month with typical use",
}


def _cost_hint(model: str) -> str:
    """Return a human-friendly monthly cost estimate for a model."""
    for prefix, hint in _MONTHLY_COST_HINT.items():
        if prefix in model.lower():
            return hint
    return "varies by usage"


def _detect_api_key() -> tuple[str, str, str, str, str] | None:
    """Scan environment for known API keys. Returns (display, provider, var, model, key) or None."""
    for display_name, provider, key_var, default_model in LLM_PROVIDERS:
        if key_var is None:
            continue
        key = os.environ.get(key_var, "")
        if key:
            return (display_name, provider, key_var, default_model, key)
    return None


def configure_llm() -> None:
    """Interactive LLM provider setup for background analysis."""

    # --- Auto-detect: check environment for existing API keys ---
    detected = _detect_api_key()
    if detected is not None:
        display_name, provider, api_key_var, default_model, key = detected
        hint = _cost_hint(default_model)
        print(f"\n  Found {api_key_var} in your environment.")
        print(f"  ormah uses a small, cheap model for background analysis")
        print(f"  (linking memories, detecting contradictions, deduplication).")
        print(f"\n  Default: {default_model}")
        print(f"  Estimated cost: {hint}")
        try:
            answer = input("\n  Use this key? (Y/n) ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("n", "no"):
            env = _read_env_file()
            env["ORMAH_LLM_PROVIDER"] = provider
            env["ORMAH_LLM_MODEL"] = default_model
            env[api_key_var] = key
            _write_env_file(env)
            print(f"  Saved to {ENV_PATH}")
            return

    # --- Manual selection: no key detected or user declined ---
    print("\n  Which LLM should ormah use for background analysis?")
    print("  (Links related memories, detects contradictions, cleans up duplicates)\n")

    display_names = [p[0] for p in LLM_PROVIDERS]
    choice = _prompt_choice("", display_names, allow_skip=False)

    if choice is None:
        choice = len(LLM_PROVIDERS) - 1  # "None" option

    display_name, provider, api_key_var, default_model = LLM_PROVIDERS[choice]

    # Handle "None" selection
    if provider == "none":
        env = _read_env_file()
        env["ORMAH_LLM_PROVIDER"] = "none"
        _write_env_file(env)
        print()
        print("  No LLM configured. Core memory works fine without one:")
        print("  store, recall, and whisper all work. Background features")
        print("  (auto-linking, conflict detection, deduplication) are disabled.")
        print("  Run `ormah setup` again to add an LLM later.")
        return

    env = _read_env_file()
    env["ORMAH_LLM_PROVIDER"] = provider
    env["ORMAH_LLM_MODEL"] = default_model

    if api_key_var:
        # Check if already set in environment
        existing_key = os.environ.get(api_key_var, "")
        if existing_key:
            print(f"\n  Found {api_key_var} in environment, using that.")
            env[api_key_var] = existing_key
        else:
            print()
            try:
                key = getpass.getpass(f"  Enter your {display_name} API key: ")
            except EOFError:
                key = ""
            if key:
                env[api_key_var] = key
            else:
                print("  No key provided. Background analysis won't work until you add one.")
                print(f"  You can add it later to {ENV_PATH}")
    else:
        # Ollama — no key needed
        print(f"\n  Using {display_name} with model '{default_model}'.")
        print("  Make sure Ollama is running: https://ollama.com")

    _write_env_file(env)
    print(f"  Saved to {ENV_PATH}")


_COST_PER_MTOK: dict[str, tuple[float, float]] = {
    # (input_cost, output_cost) per million tokens
    # More specific prefixes first — first match wins
    "claude-haiku": (1.0, 5.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-opus": (15.0, 75.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-4.1-mini": (0.80, 3.20),
    "gpt-4.1-nano": (0.20, 0.80),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini": (0.075, 0.30),
}


def _estimate_cost(total_input_tokens: int, model: str) -> tuple[float, float] | None:
    """Estimate (input_cost, output_cost) in dollars. Returns None for unknown models."""
    # Match model prefix to cost table
    for prefix, (inp, out) in _COST_PER_MTOK.items():
        if prefix in model.lower():
            input_cost = total_input_tokens / 1_000_000 * inp
            output_tokens = total_input_tokens * 0.15
            output_cost = output_tokens / 1_000_000 * out
            return (input_cost, output_cost)
    return None


def _discover_transcripts() -> list[tuple[Path, str | None]]:
    """Find JSONL transcripts in ~/.claude/projects/, sorted by mtime descending.

    Returns list of (path, space_name) tuples.
    """
    from ormah.background.session_watcher import _space_from_encoded_dir

    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return []

    transcripts: list[tuple[Path, str | None, float]] = []
    for jsonl_file in projects_dir.rglob("*.jsonl"):
        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            continue
        space = _space_from_encoded_dir(jsonl_file.parent.name)
        transcripts.append((jsonl_file, space, mtime))

    # Sort by mtime descending (most recent first)
    transcripts.sort(key=lambda x: x[2], reverse=True)
    return [(path, space) for path, space, _ in transcripts]


def backfill_transcripts() -> None:
    """Ingest existing Claude Code transcripts to bootstrap the memory graph."""
    from ormah.transcript.parser import parse_transcript

    # Gate: check LLM provider
    env = _read_env_file()
    llm_provider = env.get("ORMAH_LLM_PROVIDER", "none")
    if llm_provider == "none":
        return

    llm_model = env.get("ORMAH_LLM_MODEL", "")

    print("\n  Checking for existing Claude Code transcripts...")

    # Discover transcripts
    all_transcripts = _discover_transcripts()
    if not all_transcripts:
        print("  No transcripts found in ~/.claude/projects/ — skipping backfill.")
        return

    # Pre-filter: parse each and keep those with >= 5 user turns
    eligible: list[tuple[Path, str | None, int, int]] = []  # (path, space, turns, cleaned_chars)
    for path, space in all_transcripts:
        try:
            result = parse_transcript(path)
        except Exception:
            continue
        if result.user_turn_count >= 5:
            eligible.append((path, space, result.user_turn_count, result.cleaned_chars))

    if not eligible:
        print("  No transcripts with enough content found — skipping backfill.")
        return

    # Scope selection
    selected: list[tuple[Path, str | None, int, int]] | None = None
    total = len(eligible)

    if total > 20:
        pct_count = max(1, int(total * 0.15))
        options = [
            f"Last 20 sessions",
            f"Last 15% ({pct_count} sessions)",
            f"All {total} sessions",
            "Skip backfill",
        ]
        print(f"\n  Found {total} eligible transcripts.")
        choice = _prompt_choice("  How many to ingest?", options)
        if choice == 0:
            selected = eligible[:20]
        elif choice == 1:
            selected = eligible[:pct_count]
        elif choice == 2:
            selected = eligible
        else:
            print("  Skipping backfill.")
            return
    else:
        try:
            answer = input(f"\n  Found {total} transcripts. Ingest them? (y/N) ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("  Skipping backfill.")
            return
        selected = eligible

    if not selected:
        return

    # Estimate cost
    total_chars = sum(chars for _, _, _, chars in selected)
    total_tokens = total_chars // 4

    # Check if ollama (free)
    is_free = llm_provider == "ollama"
    if is_free:
        print(f"\n  Will ingest {len(selected)} transcripts (~{total_tokens:,} tokens).")
        print("  Using local Ollama — no API cost.")
    else:
        costs = _estimate_cost(total_tokens, llm_model)
        if costs is not None:
            input_cost, output_cost = costs
            total_cost = input_cost + output_cost
            print(f"\n  Will ingest {len(selected)} transcripts (~{total_tokens:,} tokens).")
            print(f"  Estimated cost: ${total_cost:.2f} ({llm_model})")
        else:
            print(f"\n  Will ingest {len(selected)} transcripts (~{total_tokens:,} tokens).")
            print(f"  Unknown cost for model '{llm_model}'.")

    # Confirm
    try:
        confirm = input("  Proceed? (y/N) ").strip().lower()
    except EOFError:
        confirm = ""
    if confirm not in ("y", "yes"):
        print("  Skipping backfill.")
        return

    # Ingest
    base_url = f"http://localhost:{settings.port}"
    total_memories = 0
    print()

    for i, (path, space, turns, _) in enumerate(selected, 1):
        space_label = space or "unknown"
        try:
            result = parse_transcript(path)
            if not result.conversation.strip():
                print(f"  [{i}/{len(selected)}] {space_label} — {turns} turns — skipped (empty)")
                continue

            params: dict = {}
            if space:
                params["default_space"] = space

            with httpx.Client(base_url=base_url, timeout=120.0) as client:
                r = client.post(
                    "/ingest/conversation",
                    json={"content": result.conversation},
                    params=params,
                )
                r.raise_for_status()
                data = r.json()

            if data.get("status") == "error":
                print(f"  [{i}/{len(selected)}] {space_label} — {turns} turns — error: {data.get('result', 'unknown')}")
                continue

            count = data.get("extracted", 0)
            total_memories += count
            print(f"  [{i}/{len(selected)}] {space_label} — {turns} turns — {count} memories")

        except Exception as e:
            print(f"  [{i}/{len(selected)}] {space_label} — {turns} turns — failed: {e}")

    print(f"\n  Backfill complete: {total_memories} memories extracted from {len(selected)} transcripts.")


def _diagnose_server_failure() -> None:
    """Print a helpful error when the server fails to start."""
    port = settings.port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(("localhost", port))
        if result == 0:
            print(f"  Error: port {port} is already in use.")
            print(f"  Set ORMAH_PORT in {ENV_PATH} to use a different port.")
        else:
            print("  Server did not start. Check ~/.local/share/ormah/logs/ormah.err.log")
    finally:
        sock.close()


def run_setup() -> None:
    """Interactive first-time setup."""
    print("Setting up ormah...\n")

    # 1. Find absolute path to ormah binary
    ormah_bin = get_ormah_bin_path()

    # 2. Ask for name (store in variable, seed after server is up)
    user_name = configure_identity()

    # 3. Configure LLM (writes to .env with 600 permissions)
    configure_llm()

    # 4. Start server + install auto-start
    if is_server_running():
        print("\n  Server already running.")
        server_ok = True
    else:
        print(f"\n  Starting ormah server on port {settings.port}...")
        install_autostart(ormah_bin)
        print("  Installed auto-start so it launches on login.")
        print("  (First run may take a few minutes to download the embedding model ~420 MB)")
        if wait_for_server(timeout=300.0):
            print("  Server is running.")
            server_ok = True
        else:
            _diagnose_server_failure()
            server_ok = False

    # 5. Seed identity memory via API (needs server running)
    if user_name and server_ok:
        seed_identity(user_name)

    # 6. Hook up Claude Code (hooks + MCP registration)
    print("\n  Hooking up Claude Code...")
    configure_claude_hooks(ormah_bin)
    configure_claude_code_mcp(ormah_bin)
    configure_claude_desktop(ormah_bin)
    install_claude_md()

    # 7. Cold start backfill (needs server + LLM)
    if server_ok:
        backfill_transcripts()

    # 8. Final verification message
    print(
        '\nOrmah is ready! Try asking your AI: "What do you know about me?"\n'
        '\nThe more you use it, the more it remembers.'
    )
