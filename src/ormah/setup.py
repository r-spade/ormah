"""One-shot interactive setup: hooks, MCP registration, server auto-start, LLM config."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import webbrowser
from pathlib import Path
import re

import httpx

from importlib import resources

from ormah.config import settings
from ormah.console import info, ok, play_finale, step, warn
from ormah.embeddings.cache import get_fastembed_cache_dir, get_model_cache_dirname
from ormah.server_manager import (
    get_ormah_bin_path,
    install_autostart,
    is_server_running,
    wait_for_server,
)

ENV_DIR = Path.home() / ".config" / "ormah"
ENV_PATH = ENV_DIR / ".env"
WRAPPER_PATH = ENV_DIR / "ormah-server"

CLAUDE_MD_SENTINEL_START = "<!-- ormah:start -->"
CLAUDE_MD_SENTINEL_END = "<!-- ormah:end -->"
CODEX_AGENTS_SENTINEL_START = "<!-- ormah:start -->"
CODEX_AGENTS_SENTINEL_END = "<!-- ormah:end -->"

# Provider definitions: (display name, provider, env var for API key, default model)
LLM_PROVIDERS = [
    ("Anthropic (Claude)", "litellm", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001"),
    ("OpenAI", "litellm", "OPENAI_API_KEY", "gpt-4.1-mini"),
    ("Google Gemini", "litellm", "GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
    ("Ollama (local)", "ollama", None, "llama3.2"),
    ("None", "none", None, None),
]


def _preload_local_models() -> None:
    """Preload embedding and whisper reranker models into Ormah's shared cache."""
    cache_dir = get_fastembed_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    step("Preloading local models")
    info(f"Model cache: {cache_dir}")

    try:
        from fastembed import TextEmbedding

        TextEmbedding(settings.embedding_model, cache_dir=str(cache_dir))
        ok(f"Embedding model ready: {settings.embedding_model}")
    except Exception as e:
        warn(f"Could not preload embedding model {settings.embedding_model}: {e}")

    if not settings.whisper_reranker_enabled:
        info("Whisper reranker disabled — skipping reranker preload")
        return

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        TextCrossEncoder(settings.whisper_reranker_model, cache_dir=str(cache_dir))
        ok(f"Whisper reranker ready: {settings.whisper_reranker_model}")
    except Exception as e:
        warn(f"Could not preload whisper reranker {settings.whisper_reranker_model}: {e}")


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
    # PostToolUse checkpoint — fires after high-impact operations and prompts
    # the agent to call remember() before continuing, enabling mid-session
    # write-back rather than waiting until session end.
    checkpoint_script = Path(settings_path).parent / "hooks" / "ormah-checkpoint.sh"
    checkpoint_script.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_script.write_text(
        "#!/usr/bin/env bash\n"
        "# Ormah mid-session checkpoint hook (installed by ormah setup).\n"
        "# Reads PostToolUse JSON from stdin; outputs a checkpoint reminder\n"
        "# after high-impact operations so agents save state immediately.\n"
        "set -euo pipefail\n"
        "input=$(cat)\n"
        'tool_name=$(echo "$input" | python3 -c '
        '"import sys,json; d=json.load(sys.stdin); print(d.get(\'tool_name\',\'\'))" 2>/dev/null || echo "")\n'
        'tool_input=$(echo "$input" | python3 -c '
        '"import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get(\'tool_input\',\'\')))" 2>/dev/null || echo "")\n'
        "is_high_impact=0\n"
        'case "$tool_name" in\n'
        "  Bash)\n"
        '    cmd=$(echo "$tool_input" | python3 -c '
        '"import sys,json; d=json.load(sys.stdin); print(d.get(\'command\',\'\'))" 2>/dev/null || echo "")\n'
        '    if echo "$cmd" | grep -qiE '
        '"git (commit|push|merge|rebase)|systemctl|deploy|docker|kubectl|npm publish|pip install|uv (sync|add|remove)|rm -rf|DROP|ALTER TABLE"; then\n'
        "      is_high_impact=1\n"
        "    fi\n"
        "    ;;\n"
        "  Write)\n"
        "    is_high_impact=1\n"
        "    ;;\n"
        "  Edit)\n"
        '    file=$(echo "$tool_input" | python3 -c '
        '"import sys,json; d=json.load(sys.stdin); print(d.get(\'file_path\',\'\'))" 2>/dev/null || echo "")\n'
        '    if echo "$file" | grep -qiE '
        r'"\.(env|json|toml|yaml|yml|sh|service|conf|ini)$|CLAUDE\.md|settings"; then'
        "\n"
        "      is_high_impact=1\n"
        "    fi\n"
        "    ;;\n"
        "esac\n"
        'if [ "$is_high_impact" -eq 1 ]; then\n'
        '  echo "CHECKPOINT: A high-impact operation just completed. '
        "If a decision was made, a config changed, or a milestone was reached — "
        'call remember() now to save it before continuing."\n'
        "fi\n"
    )
    checkpoint_script.chmod(0o755)

    hooks["PostToolUse"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": str(checkpoint_script),
                    "timeout": 5,
                }
            ]
        }
    ]

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


def configure_codex_hooks(ormah_bin: str) -> None:
    """Write Codex hook config to ~/.codex/hooks.json and enable the feature flag."""
    hooks_path = Path.home() / ".codex" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)

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
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{ormah_bin} whisper store",
                        "timeout": 300,
                    }
                ]
            }
        ],
    }

    _merge_json_file(str(hooks_path), {"hooks": hooks})
    _enable_codex_feature("codex_hooks")
    ok("Codex hooks installed — memories flow before every message")


def _remove_toml_table_block(text: str, table_name: str) -> str:
    """Remove a top-level TOML table block while preserving surrounding content."""
    lines = text.splitlines(keepends=True)
    start = None
    end = None
    header = f"[{table_name}]"

    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            end = len(lines)
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    end = j
                    break
            break

    if start is None or end is None:
        return text

    updated = "".join(lines[:start] + lines[end:])
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated.lstrip("\n")


def _upsert_toml_table_key(text: str, table_name: str, key: str, rendered_value: str) -> str:
    """Insert or update a key within a top-level TOML table."""
    lines = text.splitlines(keepends=True)
    header = f"[{table_name}]"
    start = None
    end = None

    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            end = len(lines)
            for j in range(i + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    end = j
                    break
            break

    entry = f"{key} = {rendered_value}\n"

    if start is None or end is None:
        block = f"{header}\n{entry}"
        if text.rstrip():
            return text.rstrip() + "\n\n" + block
        return block

    block_lines = lines[start:end]
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")

    replaced = False
    for idx in range(1, len(block_lines)):
        if key_pattern.match(block_lines[idx]):
            block_lines[idx] = entry
            replaced = True
            break

    if not replaced:
        block_lines.append(entry)

    updated = "".join(lines[:start] + block_lines + lines[end:])
    return updated


def _enable_codex_feature(feature_name: str) -> None:
    """Enable a Codex feature flag in ~/.codex/config.toml."""
    config_path = Path.home() / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = config_path.read_text() if config_path.exists() else ""
    updated = _upsert_toml_table_key(existing, "features", feature_name, "true")
    config_path.write_text(updated)


def _upsert_codex_mcp_config(ormah_bin: str) -> None:
    """Write or update the Ormah MCP entry in ~/.codex/config.toml."""
    config_path = Path.home() / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = config_path.read_text() if config_path.exists() else ""
    existing = _remove_toml_table_block(existing, "mcp_servers.ormah").rstrip()

    block = (
        "[mcp_servers.ormah]\n"
        f"command = {json.dumps(ormah_bin)}\n"
        'args = ["mcp"]\n'
    )

    if existing:
        updated = f"{existing}\n\n{block}"
    else:
        updated = block

    config_path.write_text(updated)


def configure_codex_mcp(ormah_bin: str) -> None:
    """Register Ormah MCP server in Codex config."""
    codex_bin = shutil.which("codex")
    if codex_bin:
        try:
            result = subprocess.run(
                [codex_bin, "mcp", "add", "ormah", "--", ormah_bin, "mcp"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                ok("Connected to Codex — MCP tools available")
                return
            if "already exists" in result.stderr or "already exists" in result.stdout:
                subprocess.run(
                    [codex_bin, "mcp", "remove", "ormah"],
                    capture_output=True, timeout=10,
                )
                result2 = subprocess.run(
                    [codex_bin, "mcp", "add", "ormah", "--", ormah_bin, "mcp"],
                    capture_output=True, text=True, timeout=10,
                )
                if result2.returncode == 0:
                    ok("Connected to Codex — MCP tools available")
                    return
        except Exception:
            pass

    _upsert_codex_mcp_config(ormah_bin)
    ok("Connected to Codex — MCP tools available")


def _install_markdown_block(
    target: Path,
    content_path: str,
    sentinel_start: str,
    sentinel_end: str,
) -> None:
    """Install or replace a sentinel-wrapped markdown block in a target file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    instructions = resources.files("ormah").joinpath(content_path).read_text()
    block = f"{sentinel_start}\n{instructions}{sentinel_end}\n"

    existing = target.read_text() if target.exists() else ""

    if sentinel_start in existing and sentinel_end in existing:
        start = existing.index(sentinel_start)
        end = existing.index(sentinel_end) + len(sentinel_end)
        if end < len(existing) and existing[end] == "\n":
            end += 1
        updated = existing[:start] + block + existing[end:]
    elif existing:
        updated = existing.rstrip("\n") + "\n\n" + block
    else:
        updated = block

    target.write_text(updated)


def _plugin_enabled_in_settings(settings_path: Path, plugin_name: str) -> bool:
    """Return True when the plugin is enabled in a Claude settings file."""
    if not settings_path.exists():
        return False
    try:
        data = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, ValueError):
        return False

    enabled_plugins = data.get("enabledPlugins", {})
    if not isinstance(enabled_plugins, dict):
        return False

    for key, enabled in enabled_plugins.items():
        if enabled is not True:
            continue
        if key == plugin_name or key.startswith(f"{plugin_name}@"):
            return True
    return False


def _candidate_project_roots(cwd: Path | None = None) -> list[Path]:
    """Return ancestor directories that may hold repo-scoped Claude settings."""
    base = (cwd or Path.cwd()).resolve()
    home = Path.home().resolve()

    try:
        home.relative_to(base)
        home_is_below_base = True
    except ValueError:
        home_is_below_base = False

    roots: list[Path] = []
    for root in [base, *base.parents]:
        roots.append(root)
        if root == home and not home_is_below_base:
            break

    # User scope lives under ~/.claude/settings.json, not under a home-level
    # ".claude/settings.json" discovered by walking ancestors from a repo cwd.
    if roots and roots[-1] == home and not (home / ".git").exists():
        roots.pop()

    return roots


def _detect_claude_plugin_scope(
    plugin_name: str = "ormah",
    cwd: Path | None = None,
) -> tuple[str, Path]:
    """Infer the active Claude plugin scope for the current working tree."""
    base = cwd or Path.cwd()
    project_roots = _candidate_project_roots(base)

    for root in project_roots:
        if _plugin_enabled_in_settings(root / ".claude" / "settings.local.json", plugin_name):
            return "local", root

    for root in project_roots:
        if _plugin_enabled_in_settings(root / ".claude" / "settings.json", plugin_name):
            return "project", root

    if _plugin_enabled_in_settings(Path.home() / ".claude" / "settings.json", plugin_name):
        return "user", base

    info("Could not detect Ormah plugin install scope — defaulting to project CLAUDE.md")
    return "project", base


def _get_claude_md_target(scope: str = "user", cwd: Path | None = None) -> tuple[Path, str]:
    """Resolve the target CLAUDE.md path for a given scope."""
    if scope == "user":
        return Path.home() / ".claude" / "CLAUDE.md", "~/.claude/CLAUDE.md"
    if scope == "project":
        base = cwd or Path.cwd()
        return base / "CLAUDE.md", "./CLAUDE.md"
    if scope == "local":
        base = cwd or Path.cwd()
        return base / "CLAUDE.local.md", "./CLAUDE.local.md"
    if scope == "auto":
        detected_scope, base = _detect_claude_plugin_scope(cwd=cwd)
        return _get_claude_md_target(scope=detected_scope, cwd=base)
    raise ValueError(f"Unsupported CLAUDE.md scope: {scope}")


def install_claude_md(scope: str = "user", cwd: Path | None = None) -> None:
    """Install ormah instructions into a Claude Code CLAUDE.md file."""
    target, label = _get_claude_md_target(scope=scope, cwd=cwd)
    _install_markdown_block(
        target,
        "instructions.md",
        CLAUDE_MD_SENTINEL_START,
        CLAUDE_MD_SENTINEL_END,
    )
    ok(f"Instructions added to {label}")


def _codex_agents_target() -> Path:
    """Return the effective global Codex instructions file."""
    codex_home = Path.home() / ".codex"
    override = codex_home / "AGENTS.override.md"
    if override.exists():
        return override
    return codex_home / "AGENTS.md"


def install_codex_md() -> None:
    """Install ormah instructions into Codex global AGENTS.md."""
    target = _codex_agents_target()
    _install_markdown_block(
        target,
        "codex_instructions.md",
        CODEX_AGENTS_SENTINEL_START,
        CODEX_AGENTS_SENTINEL_END,
    )
    ok(f"Instructions added to {target}")


def install_codex_agents() -> None:
    """Install Ormah custom agent definitions into ~/.codex/agents/."""
    target = Path.home() / ".codex" / "agents"
    target.mkdir(parents=True, exist_ok=True)
    content = resources.files("ormah").joinpath("agents/ormah-maintenance.toml").read_text()
    (target / "ormah-maintenance.toml").write_text(content)
    ok("Agent definition installed — ormah-maintenance subagent available in Codex")


def install_claude_agents() -> None:
    """Install ormah custom agent definitions into ~/.claude/agents/."""
    target = Path.home() / ".claude" / "agents"
    target.mkdir(parents=True, exist_ok=True)
    content = resources.files("ormah").joinpath("agents/ormah-maintenance.md").read_text()
    (target / "ormah-maintenance.md").write_text(content)
    ok("Agent definition installed — ormah-maintenance subagent available")


def install_claude_commands() -> None:
    """Install ormah slash command definitions into ~/.claude/commands/."""
    target = Path.home() / ".claude" / "commands"
    target.mkdir(parents=True, exist_ok=True)
    content = resources.files("ormah").joinpath("commands/ormah-maintenance.md").read_text()
    (target / "ormah-maintenance.md").write_text(content)
    ok("Slash command installed — /ormah-maintenance available")


def _remove_claude_agents() -> None:
    """Remove ormah agent definitions from ~/.claude/agents/."""
    agent_file = Path.home() / ".claude" / "agents" / "ormah-maintenance.md"
    if agent_file.exists():
        agent_file.unlink()
        ok("Removed ormah-maintenance agent definition")


def _remove_claude_commands() -> None:
    """Remove ormah slash command definitions from ~/.claude/commands/."""
    command_file = Path.home() / ".claude" / "commands" / "ormah-maintenance.md"
    if command_file.exists():
        command_file.unlink()
        ok("Removed ormah-maintenance slash command")


def _remove_codex_agents() -> None:
    """Remove Ormah agent definitions from ~/.codex/agents/."""
    agent_file = Path.home() / ".codex" / "agents" / "ormah-maintenance.toml"
    if agent_file.exists():
        agent_file.unlink()
        ok("Removed ormah-maintenance Codex agent definition")


def _remove_codex_hooks() -> None:
    """Remove ormah whisper hooks from ~/.codex/hooks.json."""
    hooks_path = Path.home() / ".codex" / "hooks.json"
    if not hooks_path.exists():
        info("No ~/.codex/hooks.json found — skipping")
        return
    try:
        data = json.loads(hooks_path.read_text())
    except (json.JSONDecodeError, ValueError):
        warn("Could not parse ~/.codex/hooks.json — skipping")
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

    for key in to_delete:
        del hooks_top[key]
    if not hooks_top:
        del data["hooks"]
        changed = True

    if changed:
        hooks_path.write_text(json.dumps(data, indent=2) + "\n")
        ok("Removed whisper hooks from ~/.codex/hooks.json")
    else:
        info("No ormah hooks found in hooks.json")


def _remove_markdown_block(target: Path, sentinel_start: str, sentinel_end: str, label: str) -> bool:
    """Remove a sentinel-wrapped markdown block from a target file."""
    if not target.exists():
        info(f"No {label} found — skipping")
        return False

    existing = target.read_text()
    if sentinel_start not in existing or sentinel_end not in existing:
        info(f"No ormah block found in {label} — skipping")
        return False

    start = existing.index(sentinel_start)
    end = existing.index(sentinel_end) + len(sentinel_end)
    if end < len(existing) and existing[end] == "\n":
        end += 1

    updated = existing[:start] + existing[end:]
    updated = re.sub(r"\n{3,}", "\n\n", updated)

    target.write_text(updated)
    return True


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


def configure_agent_maintenance(has_claude_code: bool, has_codex: bool) -> bool:
    """Ask whether to enable automatic agent-backed maintenance.

    Returns True if maintenance was enabled, False if skipped.
    """
    if has_claude_code and has_codex:
        agent_label = "Claude Code or Codex"
    elif has_claude_code:
        agent_label = "Claude Code"
    elif has_codex:
        agent_label = "Codex"
    else:
        return False

    print(f"\n  Use {agent_label} for automatic memory maintenance?")
    print("  (Runs judgment-heavy graph maintenance in the background when due,")
    print("   at most once every 24 hours by default. No separate API key needed.)")
    print("   This covers maintenance decisions, not transcript extraction.")
    try:
        answer = input("\n  Enable? (Y/n) ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("n", "no"):
        env = _read_env_file()
        env["ORMAH_CLAUDE_MAINTENANCE_ENABLED"] = "true"
        _write_env_file(env)
        if has_codex:
            _enable_codex_feature("multi_agent")
        ok(f"Automatic maintenance enabled — {agent_label} can run run_maintenance when signalled")
        return True
    else:
        info("Skipped automatic maintenance — run 'ormah setup' again to enable later")
        return False


def _print_setup_summary(ormah_bin: str) -> None:
    ok("Ormah is ready.")
    info("Installed locations:")
    print(f"    CLI: {ormah_bin}")
    print(f"    Config: {ENV_PATH}")
    print(f"    Memory: {settings.memory_dir}")
    print(f"    Graph UI: http://localhost:{settings.port}")


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
            info("Check ~/.local/share/ormah/logs/ormah.log")
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
    """Unregister ormah MCP server from supported AI clients."""
    import platform as _platform

    # Claude Code
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

    # Codex
    codex_bin = shutil.which("codex")
    if codex_bin:
        try:
            result = subprocess.run(
                [codex_bin, "mcp", "remove", "ormah"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                ok("Removed ormah MCP registration from Codex")
            else:
                _remove_codex_mcp_config()
        except Exception:
            _remove_codex_mcp_config()
    else:
        _remove_codex_mcp_config()


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


def _remove_codex_mcp_config() -> None:
    """Remove ormah entry from ~/.codex/config.toml."""
    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return

    existing = config_path.read_text()
    updated = _remove_toml_table_block(existing, "mcp_servers.ormah")
    if updated == existing:
        return

    config_path.write_text(updated.rstrip() + ("\n" if updated.strip() else ""))
    ok(f"Removed ormah from {config_path}")


def _remove_claude_md_block() -> None:
    """Remove the ormah instructions block from ~/.claude/CLAUDE.md."""
    target = Path.home() / ".claude" / "CLAUDE.md"
    if _remove_markdown_block(
        target, CLAUDE_MD_SENTINEL_START, CLAUDE_MD_SENTINEL_END, "~/.claude/CLAUDE.md"
    ):
        ok("Removed ormah block from ~/.claude/CLAUDE.md")


def _remove_codex_md_block() -> None:
    """Remove the ormah instructions block from the active Codex AGENTS file."""
    target = _codex_agents_target()
    if _remove_markdown_block(
        target, CODEX_AGENTS_SENTINEL_START, CODEX_AGENTS_SENTINEL_END, str(target)
    ):
        ok(f"Removed ormah block from {target}")


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
    from ormah.config import settings as _settings

    cache_dir = get_fastembed_cache_dir()
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
                dir_name = get_model_cache_dirname(_settings.embedding_model, [m])
                if dir_name:
                    model_dirs.add(dir_name)
                break
    except Exception:
        pass

    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        for m in TextCrossEncoder.list_supported_models():
            if m.get("model") == _settings.whisper_reranker_model:
                dir_name = get_model_cache_dirname(_settings.whisper_reranker_model, [m])
                if dir_name:
                    model_dirs.add(dir_name)
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
    _remove_codex_hooks()

    # c. Remove MCP registration
    step("Removing MCP registration")
    _remove_mcp_registration()

    # d. Remove CLAUDE.md block
    step("Removing CLAUDE.md instructions")
    _remove_claude_md_block()
    _remove_codex_md_block()
    _remove_codex_agents()
    _remove_claude_agents()
    _remove_claude_commands()

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


def run_setup(
    ci: bool = False,
    update: bool = False,
    skip_client_setup: bool = False,
) -> None:
    """First-time setup. Pass ci=True (or set ORMAH_CI=1) for non-interactive mode.
    Pass update=True to skip interactive questions and only reapply hooks/MCP config.
    Pass skip_client_setup=True when integrations are managed externally (for example by a plugin)."""
    ci = ci or os.environ.get("ORMAH_CI") == "1"

    if update:
        print("Updating ormah integrations...\n")
    else:
        print("Setting up ormah...\n")

    # 1. Find absolute path to ormah binary
    ormah_bin = get_ormah_bin_path()

    # 2. Detect supported coding agents and offer maintenance upfront — no API key needed
    has_claude_code = shutil.which("claude") is not None
    has_codex = shutil.which("codex") is not None or (Path.home() / ".codex").exists()
    agent_maintenance = False
    if (has_claude_code or has_codex) and not ci and not update and not skip_client_setup:
        if has_claude_code and has_codex:
            step("Claude Code and Codex detected")
        elif has_claude_code:
            step("Claude Code detected")
        else:
            step("Codex detected")
        agent_maintenance = configure_agent_maintenance(has_claude_code, has_codex)

    # 3. Configure LLM — skip if agent-backed maintenance is handling background jobs
    if ci:
        env = _read_env_file()
        env["ORMAH_LLM_PROVIDER"] = "none"
        _write_env_file(env)
        info("CI mode — LLM set to none")
    elif update:
        pass  # keep existing LLM config
    elif agent_maintenance:
        env = _read_env_file()
        env["ORMAH_LLM_PROVIDER"] = "none"
        _write_env_file(env)
    else:
        configure_llm()

    # 4. Generate server wrapper
    wrapper_path = generate_server_wrapper(ormah_bin)

    # 4.5 Preload local models into Ormah's shared model cache
    _preload_local_models()

    # 5. Start server + install auto-start
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

    desktop_configured = False
    if not skip_client_setup:
        if has_claude_code:
            step("Hooking up Claude Code")
            configure_claude_hooks(ormah_bin)
            configure_claude_code_mcp(ormah_bin)
            install_claude_md()
            install_claude_agents()
            install_claude_commands()

        if has_codex:
            step("Hooking up Codex")
            configure_codex_hooks(ormah_bin)
            configure_codex_mcp(ormah_bin)
            install_codex_md()
            install_codex_agents()

        desktop_configured = configure_claude_desktop(ormah_bin)

        if not has_claude_code and not has_codex and not desktop_configured:
            warn("No Claude Code, Claude Desktop, or Codex detected")
            info("You can manually configure MCP in your AI client:")
            print(f"    Command: {ormah_bin} mcp")
            info("Or run 'ormah setup' again after installing Claude Code, Claude Desktop, or Codex")

    # 7. Cold start backfill (needs server + LLM)
    if server_ok and not ci:
        backfill_transcripts()

    # 8. Finale animation + completion message
    step("Setup complete")
    if not ci:
        play_finale()
    _print_setup_summary(ormah_bin)
    if server_ok and not ci:
        webbrowser.open(f"http://localhost:{settings.port}")
