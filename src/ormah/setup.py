"""One-shot interactive setup: hooks, MCP registration, server auto-start, LLM config."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import httpx

from importlib import resources

from ormah.config import settings
from ormah.console import info, ok, play_finale, step, warn
from ormah.server_manager import (
    get_ormah_bin_path,
    install_autostart,
    is_server_running,
    wait_for_server,
)

ENV_DIR = Path.home() / ".config" / "ormah"
ENV_PATH = ENV_DIR / ".env"
WRAPPER_PATH = ENV_DIR / "start-server.sh"

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
    ok("Whisper hooks installed \u2014 memories flow before every message")


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
                ok("MCP tool registered \u2014 Claude can store and recall memories")
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
                    ok("MCP tool registered \u2014 Claude can store and recall memories")
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
    ok("MCP tool registered \u2014 Claude can store and recall memories")


def configure_claude_desktop(ormah_bin: str) -> bool:
    """Register ormah MCP server in Claude Desktop config (if installed).

    Returns True if Claude Desktop was detected and configured.
    """
    import platform as _platform

    if _platform.system() != "Darwin":
        # Claude Desktop config path is macOS-specific for now
        return False

    config_dir = os.path.expanduser("~/Library/Application Support/Claude")
    config_path = os.path.join(config_dir, "claude_desktop_config.json")

    if not os.path.exists(config_dir):
        return False

    mcp_entry = {
        "ormah": {
            "command": ormah_bin,
            "args": ["mcp"],
        }
    }

    _merge_json_file(config_path, {"mcpServers": mcp_entry})
    ok("Connected to Claude Desktop \u2014 MCP tools available")
    info("Whisper hooks require Claude Code; Desktop uses MCP tools directly")
    return True


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
    ok("Instructions added to ~/.claude/CLAUDE.md")


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


def generate_server_wrapper(ormah_bin: str) -> Path:
    """Generate daemon wrapper that inherits API keys from user's shell."""
    ENV_DIR.mkdir(parents=True, exist_ok=True)

    script = f"""\
#!/usr/bin/env bash
set -euo pipefail

# Capture API keys from user's login shell
while IFS= read -r line; do
    key="${{line%%=*}}"
    value="${{line#*=}}"
    export "$key=$value"
done < <("${{SHELL:-/bin/bash}}" -lic 'env' 2>/dev/null \\
    | grep -E '^(ANTHROPIC_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|GROQ_API_KEY|MISTRAL_API_KEY|COHERE_API_KEY|AZURE_API_KEY|AZURE_API_BASE|AZURE_API_VERSION|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_REGION_NAME)=' \\
    || true)

# Load ormah-specific config
_env_file="$HOME/.config/ormah/.env"
if [ -f "$_env_file" ]; then
    set -a; . "$_env_file"; set +a
fi

exec {ormah_bin} server start
"""

    WRAPPER_PATH.write_text(script)
    os.chmod(WRAPPER_PATH, 0o700)
    return WRAPPER_PATH


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
        except EOFError:
            return None
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def configure_identity() -> str | None:
    """Ask the user for their name. Returns the name or None if skipped."""
    try:
        name = input("  What should ormah call you? ").strip()
    except EOFError:
        name = ""

    if not name:
        info("Skipped \u2014 ormah will learn your name naturally")
        return None
    return name


def seed_identity(name: str) -> None:
    """Store the user's name in ormah via the running server."""
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
        ok("Ormah now knows your name")
    except Exception:
        warn("Could not seed identity \u2014 server may not be ready yet")
        info("No worries, ormah will learn your name naturally")


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
        print("  ormah uses a small, cheap model for background analysis")
        print("  (linking memories, detecting contradictions, deduplication).")
        print(f"  Default: {default_model} ({hint})")
        try:
            answer = input("\n  Use this key? (Y/n) ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("n", "no"):
            env = _read_env_file()
            env["ORMAH_LLM_PROVIDER"] = provider
            env["ORMAH_LLM_MODEL"] = default_model
            _write_env_file(env)
            ok(f"Using {api_key_var} from environment with {default_model}")
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
        info("No LLM configured \u2014 core memory works without one")
        info("Run 'ormah setup' again to add an LLM later")
        return

    env = _read_env_file()
    env["ORMAH_LLM_PROVIDER"] = provider
    env["ORMAH_LLM_MODEL"] = default_model

    if api_key_var:
        # Check if already set in environment
        existing_key = os.environ.get(api_key_var, "")
        if existing_key:
            ok(f"Using {api_key_var} from environment with {default_model}")
        else:
            shell_profile = "~/.zshrc" if os.environ.get("SHELL", "").endswith("zsh") else "~/.bashrc"
            warn(f"No {api_key_var} found in your environment")
            print(f"  Add it to your shell profile ({shell_profile}):")
            print(f"    export {api_key_var}=your-key-here")
            print("  Then restart your shell or run: source " + shell_profile)
    else:
        # Ollama — no key needed
        ok(f"Using {display_name} with model '{default_model}'")
        info("Make sure Ollama is running: https://ollama.com")

    _write_env_file(env)


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

    step("Backfilling transcripts")

    # Discover transcripts
    all_transcripts = _discover_transcripts()
    if not all_transcripts:
        info("No transcripts found \u2014 skipping backfill")
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
        info("No transcripts with enough content \u2014 skipping backfill")
        return

    # Scope selection
    selected: list[tuple[Path, str | None, int, int]] | None = None
    total = len(eligible)

    if total > 20:
        pct_count = max(1, int(total * 0.15))
        options = [
            "Last 20 sessions",
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
            info("Skipping backfill")
            return
    else:
        try:
            answer = input(f"\n  Found {total} transcripts. Ingest them? (y/N) ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            info("Skipping backfill")
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
        info("Skipping backfill")
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
                info(f"[{i}/{len(selected)}] {space_label} \u2014 {turns} turns \u2014 skipped (empty)")
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
                warn(f"[{i}/{len(selected)}] {space_label} \u2014 {turns} turns \u2014 error: {data.get('result', 'unknown')}")
                continue

            count = data.get("extracted", 0)
            total_memories += count
            info(f"[{i}/{len(selected)}] {space_label} \u2014 {turns} turns \u2014 {count} memories")

        except Exception as e:
            warn(f"[{i}/{len(selected)}] {space_label} \u2014 {turns} turns \u2014 failed: {e}")

    ok(f"Backfill complete: {total_memories} memories from {len(selected)} transcripts")


def _diagnose_server_failure() -> None:
    """Print a helpful error when the server fails to start."""
    port = settings.port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(("localhost", port))
        if result == 0:
            warn(f"Port {port} is already in use")
            info(f"Set ORMAH_PORT in {ENV_PATH} to use a different port")
        else:
            warn("Server did not start")
            info("Check ~/.local/share/ormah/logs/ormah.err.log")
    finally:
        sock.close()


def _remove_claude_hooks() -> None:
    """Remove ormah whisper hooks from ~/.claude/settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        info("No ~/.claude/settings.json found — skipping")
        return
    try:
        data = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, ValueError):
        warn("Could not parse ~/.claude/settings.json — skipping")
        return

    hooks_top = data.get("hooks")
    if not isinstance(hooks_top, dict):
        info("No hooks section — nothing to remove")
        return

    def _is_ormah_hook(entry: dict) -> bool:
        cmd = entry.get("command", "")
        return "whisper inject" in cmd or "whisper store" in cmd

    changed = False
    to_delete = []
    for event, matchers in hooks_top.items():
        if not isinstance(matchers, list):
            continue
        new_matchers = []
        for matcher in matchers:
            if not isinstance(matcher, dict):
                new_matchers.append(matcher)
                continue
            inner = matcher.get("hooks", [])
            filtered = [h for h in inner if not _is_ormah_hook(h)]
            if len(filtered) != len(inner):
                changed = True
            if filtered:
                new_matchers.append({**matcher, "hooks": filtered})
            else:
                changed = True
        if new_matchers:
            hooks_top[event] = new_matchers
        else:
            to_delete.append(event)
            changed = True

    for k in to_delete:
        del hooks_top[k]
    if not hooks_top:
        del data["hooks"]
        changed = True

    if changed:
        settings_path.write_text(json.dumps(data, indent=2) + "\n")
        ok("Removed whisper hooks from ~/.claude/settings.json")
    else:
        info("No ormah hooks found in settings.json")


def _remove_mcp_registration() -> None:
    """Unregister ormah MCP server from Claude Code (and Claude Desktop on macOS)."""
    import platform as _platform

    # Try official CLI first
    claude_bin = shutil.which("claude")
    if claude_bin:
        try:
            result = subprocess.run(
                [claude_bin, "mcp", "remove", "ormah", "--scope", "user"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                ok("Removed ormah MCP registration from Claude Code")
            else:
                # Fallback: edit ~/.claude.json directly
                _remove_mcp_from_json(Path.home() / ".claude.json")
        except Exception:
            _remove_mcp_from_json(Path.home() / ".claude.json")
    else:
        _remove_mcp_from_json(Path.home() / ".claude.json")

    # macOS: also check Claude Desktop
    if _platform.system() == "Darwin":
        desktop_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if desktop_config.exists():
            _remove_mcp_from_json(desktop_config)


def _remove_mcp_from_json(config_path: Path) -> None:
    """Remove ormah entry from mcpServers in a JSON config file."""
    if not config_path.exists():
        return
    try:
        data = json.loads(config_path.read_text())
    except (json.JSONDecodeError, ValueError):
        warn(f"Could not parse {config_path} — skipping")
        return

    mcp_servers = data.get("mcpServers", {})
    if "ormah" not in mcp_servers:
        return

    del mcp_servers["ormah"]
    if not mcp_servers:
        del data["mcpServers"]
    else:
        data["mcpServers"] = mcp_servers

    config_path.write_text(json.dumps(data, indent=2) + "\n")
    ok(f"Removed ormah from {config_path}")


def _remove_claude_md_block() -> None:
    """Remove the ormah instructions block from ~/.claude/CLAUDE.md."""
    target = Path.home() / ".claude" / "CLAUDE.md"
    if not target.exists():
        info("No ~/.claude/CLAUDE.md found — skipping")
        return

    existing = target.read_text()
    if CLAUDE_MD_SENTINEL_START not in existing or CLAUDE_MD_SENTINEL_END not in existing:
        info("No ormah block found in CLAUDE.md — skipping")
        return

    start = existing.index(CLAUDE_MD_SENTINEL_START)
    end = existing.index(CLAUDE_MD_SENTINEL_END) + len(CLAUDE_MD_SENTINEL_END)
    # Consume trailing newline if present
    if end < len(existing) and existing[end] == "\n":
        end += 1

    updated = existing[:start] + existing[end:]
    # Collapse triple (or more) newlines to double
    import re
    updated = re.sub(r"\n{3,}", "\n\n", updated)

    target.write_text(updated)
    ok("Removed ormah block from ~/.claude/CLAUDE.md")


def _get_running_server_data_dir() -> Path | None:
    """Return the data directory of the running ormah server by inspecting its open files.

    This works regardless of which version of ormah installed the server, because it reads
    the actual open file descriptors of the live process rather than the current config.
    Must be called BEFORE the server is stopped.
    """
    # Step 1: find the server PID via systemd, then pgrep as fallback
    pid: str | None = None
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", "ormah.service",
             "--property=MainPID", "--value"],
            capture_output=True, text=True, timeout=5,
        )
        candidate = r.stdout.strip()
        if candidate and candidate != "0":
            pid = candidate
    except Exception:
        pass

    if pid is None:
        try:
            r = subprocess.run(
                ["pgrep", "-f", "ormah server start"],
                capture_output=True, text=True, timeout=5,
            )
            lines = r.stdout.strip().splitlines()
            if lines:
                pid = lines[0].strip()
        except Exception:
            pass

    if not pid:
        return None

    # Step 2: find an open index.db file in /proc (Linux) or via lsof (cross-platform)
    # Linux: /proc/{pid}/fd symlinks are fast and require no extra tools
    try:
        fd_dir = Path(f"/proc/{pid}/fd")
        if fd_dir.exists():
            for fd_link in fd_dir.iterdir():
                try:
                    target = Path(os.readlink(fd_link))
                    if target.name == "index.db" and target.exists():
                        return target.parent
                except OSError:
                    continue
    except Exception:
        pass

    # macOS / fallback: lsof
    try:
        r = subprocess.run(
            ["lsof", "-p", pid, "-Fn"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if line.startswith("n") and line.endswith("index.db"):
                db_path = Path(line[1:])
                if db_path.exists():
                    return db_path.parent
    except Exception:
        pass

    return None


def _remove_fastembed_cache() -> None:
    """Delete the fastembed model cache entries that ormah downloaded."""
    import tempfile
    from ormah.config import settings as _settings

    cache_dir = Path(
        os.environ.get("FASTEMBED_CACHE_PATH",
                       os.path.join(tempfile.gettempdir(), "fastembed_cache"))
    )
    if not cache_dir.exists():
        info("No fastembed model cache found — skipping")
        return

    # Build the set of cache subdirectory names to delete.
    # fastembed stores models as  models--{hf_source_repo.replace('/', '--')}
    # Use fastembed's own model registry to resolve model name → HF source repo.
    model_dirs: set[str] = set()

    try:
        from fastembed import TextEmbedding
        for m in TextEmbedding.list_supported_models():
            if m.get("model") == _settings.embedding_model:
                hf_repo = (m.get("sources") or {}).get("hf", "")
                if hf_repo:
                    model_dirs.add(f"models--{hf_repo.replace('/', '--')}")
                break
    except Exception:
        pass

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        for m in TextCrossEncoder.list_supported_models():
            if m.get("model") == _settings.whisper_reranker_model:
                hf_repo = (m.get("sources") or {}).get("hf", "")
                if hf_repo:
                    model_dirs.add(f"models--{hf_repo.replace('/', '--')}")
                break
    except Exception:
        pass

    if not model_dirs:
        warn(f"Could not identify model cache dirs — delete manually: {cache_dir}")
        return

    for dir_name in sorted(model_dirs):
        model_path = cache_dir / dir_name
        if model_path.exists():
            shutil.rmtree(model_path)
            ok(f"Deleted model cache: {model_path}")
        else:
            info(f"Model cache not found: {model_path} — skipping")

    # Remove the cache dir itself if now empty
    try:
        if cache_dir.exists() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    except OSError:
        pass


def run_uninstall(yes: bool = False) -> None:
    """Remove all ormah integrations, data, and optionally the package itself."""
    print("This will remove all ormah integrations and data.\n")

    if not yes:
        try:
            answer = input("Are you sure? (y/N) ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            info("Uninstall cancelled")
            return

        try:
            confirm = input('Type "yes" to confirm: ').strip()
        except EOFError:
            confirm = ""
        if confirm != "yes":
            info("Uninstall cancelled")
            return

    print()

    # Snapshot the running server's data directory BEFORE stopping it.
    # This is the only reliable way to find where data lives regardless of which
    # version of ormah is installed (older releases used a relative Path("memory")
    # that resolves differently depending on the invoking binary's config).
    live_data_dir = _get_running_server_data_dir()

    # a. Stop daemon
    step("Stopping server")
    from ormah.server_manager import uninstall_autostart
    uninstall_autostart()

    # b. Remove Claude Code hooks
    step("Removing Claude Code hooks")
    _remove_claude_hooks()

    # c. Remove MCP registration
    step("Removing MCP registration")
    _remove_mcp_registration()

    # d. Remove CLAUDE.md block
    step("Removing CLAUDE.md instructions")
    _remove_claude_md_block()

    # e. Delete data directories
    step("Deleting data directories")

    xdg_dirs = [
        Path.home() / ".local" / "share" / "ormah",
        Path.home() / ".cache" / "ormah",
        Path.home() / ".config" / "ormah",
    ]
    data_dirs: list[Path] = list(xdg_dirs)

    # Add the live server's actual data dir if it falls outside the XDG tree.
    # Also add the config-derived path as a safety net (handles custom ORMAH_MEMORY_DIR).
    from ormah.config import settings as _settings
    config_mem_dir = _settings.memory_dir
    if not config_mem_dir.is_absolute():
        config_mem_dir = Path.home() / config_mem_dir
    config_mem_dir = config_mem_dir.resolve()

    for candidate in filter(None, [live_data_dir, config_mem_dir]):
        if not any(candidate == d or str(candidate).startswith(str(d) + "/")
                   for d in xdg_dirs):
            if candidate not in data_dirs:
                data_dirs.append(candidate)

    for d in data_dirs:
        if d.exists():
            shutil.rmtree(d)
            ok(f"Deleted {d}")
        else:
            info(f"{d} not found — skipping")

    # f. Delete fastembed model cache
    step("Removing embedding model cache")
    _remove_fastembed_cache()

    # h. Uninstall the package
    step("Uninstalling ormah package")
    try:
        result = subprocess.run(
            ["uv", "tool", "uninstall", "ormah"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            ok("Package uninstalled via uv")
        else:
            warn("Could not uninstall via uv — remove manually with: uv tool uninstall ormah")
    except Exception:
        warn("Could not uninstall via uv — remove manually with: uv tool uninstall ormah")

    print()
    ok("Ormah has been uninstalled")


def run_setup(ci: bool = False) -> None:
    """First-time setup. Pass ci=True (or set ORMAH_CI=1) for non-interactive mode."""
    ci = ci or os.environ.get("ORMAH_CI") == "1"

    print("Setting up ormah...\n")

    # 1. Find absolute path to ormah binary
    ormah_bin = get_ormah_bin_path()

    # 2. Ask for name (store in variable, seed after server is up)
    if ci:
        user_name = None
    else:
        user_name = configure_identity()

    # 3. Configure LLM (writes to .env with 600 permissions)
    if ci:
        env = _read_env_file()
        env["ORMAH_LLM_PROVIDER"] = "none"
        _write_env_file(env)
        info("CI mode — LLM set to none")
    else:
        configure_llm()

    # 3.5 Generate server wrapper
    wrapper_path = generate_server_wrapper(ormah_bin)

    # 4. Start server + install auto-start
    if is_server_running():
        ok("Server already running")
        server_ok = True
    else:
        step("Starting server")
        install_autostart(ormah_bin, wrapper_path=str(wrapper_path))
        ok("Installed auto-start (launches on login)")

        if wait_for_server(show_progress=True):
            server_ok = True
        else:
            _diagnose_server_failure()
            server_ok = False

    # 5. Seed identity memory via API (needs server running)
    if user_name and server_ok:
        seed_identity(user_name)

    # 6. Hook up Claude integrations
    has_claude_code = shutil.which("claude") is not None
    has_claude_desktop = os.path.exists(
        os.path.expanduser("~/Library/Application Support/Claude")
    )

    if has_claude_code:
        step("Hooking up Claude Code")
        configure_claude_hooks(ormah_bin)
        configure_claude_code_mcp(ormah_bin)
        install_claude_md()

    desktop_configured = configure_claude_desktop(ormah_bin)

    if not has_claude_code and not desktop_configured:
        warn("No Claude client detected")
        info("You can manually configure MCP in your AI client:")
        print(f"    Command: {ormah_bin} mcp")
        info("Or run 'ormah setup' again after installing Claude Code or Claude Desktop")

    # 7. Cold start backfill (needs server + LLM)
    if server_ok and not ci:
        backfill_transcripts()

    # 8. Finale animation + completion message
    step("Setup complete")
    if not ci:
        play_finale()
    ok('Ormah is ready! Try asking your AI: "What do you know about me?"')
    info("The more you use it, the more it remembers.")
