"""One-shot interactive setup: hooks, MCP registration, server auto-start, LLM config."""

from __future__ import annotations

import contextlib
import glob
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import webbrowser
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable

import httpx

from ormah.config import settings
from ormah.console import info, ok, play_finale, step, warn
from ormah.embeddings.cache import get_fastembed_cache_dir, get_model_cache_dirname
from ormah.server_manager import (
    get_ormah_bin_path,
    is_server_running,
    restart_with_autostart,
)

ENV_DIR = Path.home() / ".config" / "ormah"


def _find_binary(name: str) -> str | None:
    """Find a binary by name, checking PATH and common install locations.

    GUI apps launched from the system tray don't inherit the user's full shell
    PATH (e.g. nvm, mise, homebrew shims), so shutil.which alone misses
    binaries the user can run fine from a terminal. This checks the extra spots.
    """
    found = shutil.which(name)
    if found:
        return found
    # High-priority: version-manager shims (newest nvm version first, then mise)
    nvm_paths = [
        Path(p)
        for p in sorted(
            glob.glob(str(Path.home() / ".nvm" / "versions" / "node" / "*" / "bin" / name)),
            reverse=True,  # lexicographic desc → newest node version first
        )
    ]
    candidates: list[Path] = [
        Path.home() / ".local" / "share" / "mise" / "shims" / name,
        *nvm_paths,
        Path.home() / ".local" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,   # macOS Homebrew (Apple Silicon)
        Path("/usr/local/homebrew/bin") / name,  # macOS Homebrew (Intel)
        Path("/usr/bin") / name,
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
ENV_PATH = ENV_DIR / ".env"
WRAPPER_PATH = ENV_DIR / "ormah-server"
CLOUD_RECOVERY_FILENAMES = frozenset({"cloud.key", "ormah-recovery-kit.md"})
DESKTOP_BUNDLE_IDENTIFIER = "dev.ormah.desktop"
DESKTOP_PRODUCT_NAME = "Ormah"
MACOS_SYSTEM_APPLICATIONS_DIR = Path("/Applications")

CLAUDE_MD_SENTINEL_START = "<!-- ormah:start -->"
CLAUDE_MD_SENTINEL_END = "<!-- ormah:end -->"
CODEX_AGENTS_SENTINEL_START = "<!-- ormah:start -->"
CODEX_AGENTS_SENTINEL_END = "<!-- ormah:end -->"
PI_AGENTS_MD_SENTINEL_START = "<!-- ormah:start -->"
PI_AGENTS_MD_SENTINEL_END = "<!-- ormah:end -->"
PI_PACKAGE_SOURCE = "npm:ormah-pi"

# Provider definitions: (display name, provider, env var for API key, default model)
LLM_PROVIDERS = [
    (
        "Anthropic Claude Haiku 4.5 (recommended)",
        "litellm",
        "ANTHROPIC_API_KEY",
        "claude-haiku-4-5-20251001",
    ),
    (
        "Anthropic Claude Sonnet (higher cost)",
        "litellm",
        "ANTHROPIC_API_KEY",
        "claude-sonnet-4-5-20250929",
    ),
    ("OpenAI", "litellm", "OPENAI_API_KEY", "gpt-4.1-mini"),
    ("Google Gemini", "litellm", "GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
    ("Ollama (local)", "ollama", None, "llama3.2"),
    ("None", "none", None, None),
]

_LLM_KEY_POLICY_KEYS = ("ORMAH_LLM_API_KEY_ENV_VAR", "ORMAH_LLM_INHERIT_API_KEY")


def _preload_local_models() -> None:
    """Preload embedding and whisper reranker models into Ormah's shared cache."""
    cache_dir = get_fastembed_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    step("Preloading local models")
    info(f"Model cache: {cache_dir}")

    if settings.embedding_provider == "local":
        try:
            from fastembed import TextEmbedding

            TextEmbedding(settings.embedding_model, cache_dir=str(cache_dir))
            ok(f"Embedding model ready: {settings.embedding_model}")
        except Exception as e:
            warn(f"Could not preload embedding model {settings.embedding_model}: {e}")
    else:
        info(
            "Skipping FastEmbed embedding preload for "
            f"{settings.embedding_provider} model: {settings.embedding_model}"
        )

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


def _is_ormah_hook(entry: dict) -> bool:
    """True when a hook entry is one Ormah installs (argv-aware, not substring).

    Recognizes BOTH install forms:
      - CLI (ormah setup): `<...>/ormah whisper inject|store`
      - Plugin wrapper: `<...>/ormah-whisper-inject` | `<...>/ormah-whisper-store`
        (see integrations/claude-plugin/hooks/hooks.json)
    A third-party command that merely contains the substring "whisper inject"/
    "whisper store" is never misclassified. Works for install dedup and uninstall
    alike, and is resilient to the ormah binary path changing between runs.
    """
    if not isinstance(entry, dict):
        return False
    cmd = entry.get("command", "")
    if not isinstance(cmd, str):
        return False
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return False
    if not parts:
        return False
    name = Path(parts[0]).name
    if name in ("ormah-whisper-inject", "ormah-whisper-store"):
        return True  # plugin wrapper form
    return (
        len(parts) >= 3
        and name == "ormah"
        and parts[1] == "whisper"
        and parts[2] in ("inject", "store")
    )  # CLI form


def _merge_hooks(existing: dict, ormah_hooks: dict) -> dict:
    """Merge Ormah hook groups into an existing hooks dict, preserving co-tenants.

    For each event Ormah claims: strip prior Ormah entries (via _is_ormah_hook),
    keep every third-party hook, then append Ormah's matchers. Events Ormah does
    not claim are left untouched. Idempotent. Pure (no I/O).

    Matcher drop rule: a matcher is dropped only when removing Ormah hooks left it
    completely empty (i.e. it held *only* Ormah hooks). Matchers with no "hooks" key,
    an empty hooks list, or hooks that are all third-party are preserved verbatim.
    """
    merged = dict(existing)
    for event, ormah_matchers in ormah_hooks.items():
        current = merged.get(event)
        if not isinstance(current, list):
            current = []
        cleaned = []
        for matcher in current:
            if not isinstance(matcher, dict):
                cleaned.append(matcher)
                continue
            inner = matcher.get("hooks", [])
            kept = [h for h in inner if not _is_ormah_hook(h)]
            if len(kept) == len(inner):
                cleaned.append(matcher)  # nothing removed -> preserve verbatim
            elif kept:
                cleaned.append({**matcher, "hooks": kept})  # removed some, others remain
            # else: held ONLY Ormah hooks -> drop the now-empty matcher (intentional cleanup)
        merged[event] = cleaned + list(ormah_matchers)
    return merged


def _strip_ormah_hooks(existing: dict) -> tuple[dict, bool]:
    """Remove Ormah hook entries while preserving every untouched matcher.

    Returns the cleaned hooks mapping and whether any Ormah hook was removed.
    Missing, empty, or malformed inner ``hooks`` values are preserved verbatim:
    only a matcher actually changed by removing an Ormah hook may be rewritten
    or dropped.
    """
    cleaned = dict(existing)
    changed = False
    for event, matchers in existing.items():
        if not isinstance(matchers, list):
            continue

        cleaned_matchers = []
        event_changed = False
        for matcher in matchers:
            if not isinstance(matcher, dict):
                cleaned_matchers.append(matcher)
                continue

            inner = matcher.get("hooks")
            if not isinstance(inner, list):
                cleaned_matchers.append(matcher)
                continue

            kept = [hook for hook in inner if not _is_ormah_hook(hook)]
            if len(kept) == len(inner):
                cleaned_matchers.append(matcher)
                continue

            event_changed = True
            if kept:
                cleaned_matchers.append({**matcher, "hooks": kept})

        if event_changed:
            changed = True
            if cleaned_matchers:
                cleaned[event] = cleaned_matchers
            else:
                cleaned.pop(event, None)

    return cleaned, changed


def _atomic_write(path: str, text: str, mode: int | None = None) -> None:
    """Write text to `path` atomically (temp file in the same dir + os.replace).

    Prevents a crash mid-write from leaving a truncated/corrupt config — the
    target is either the old bytes or the full new bytes, never a partial file.
    If ``path`` is a symlink, atomically replace its resolved target so the link
    itself remains intact.
    """
    destination = os.path.realpath(path) if os.path.islink(path) else os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, destination)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _install_hooks(path: str, ormah_hooks: dict) -> bool:
    """Read a JSON hooks config, merge Ormah hooks preserving co-tenants, write back.

    Returns True if the merged config was written, False if it aborted without
    writing. Fail-closed: if the file exists but does not parse OR does not hold a
    JSON object, warn and abort (mirrors the uninstall no-op) so a hand-edited
    config with a transient syntax error is never replaced by a hooks-only file,
    losing theme/permissions. The write is atomic (no partial-write corruption).
    """
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            warn(f"Could not parse {path} — leaving it unchanged; hooks not configured")
            return False
        if not isinstance(existing, dict):
            warn(f"{path} is not a JSON object — leaving it unchanged; hooks not configured")
            return False
    current = existing.get("hooks")
    if current is None:
        current = {}
    elif not isinstance(current, dict):
        warn(f"{path} has a non-object 'hooks' section — leaving it unchanged; hooks not configured")
        return False
    for event in ormah_hooks:
        value = current.get(event)
        if value is not None and not isinstance(value, list):
            warn(f"{path} has a non-list '{event}' hooks entry — leaving it unchanged; hooks not configured")
            return False
    try:
        existing["hooks"] = _merge_hooks(current, ormah_hooks)
    except Exception:
        warn(f"{path} has a malformed hooks structure — leaving it unchanged; hooks not configured")
        return False
    _atomic_write(path, json.dumps(existing, indent=2) + "\n")
    return True


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

    if _install_hooks(settings_path, hooks):
        ok("Whisper hooks installed \u2014 memories flow before every message")


def configure_claude_code_mcp(ormah_bin: str) -> None:
    """Register ormah MCP server in Claude Code user config.

    Uses ``claude mcp add`` when available (correct format, user scope).
    Falls back to direct JSON editing if the claude CLI is not on PATH.
    """
    # Prefer the official CLI — it writes the correct format
    claude_bin = _find_binary("claude")
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

    if _install_hooks(str(hooks_path), hooks):
        _enable_codex_feature("hooks", deprecated_feature_names=("codex_hooks",))
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


def _remove_toml_table_key(text: str, table_name: str, key: str) -> str:
    """Remove a key from a top-level TOML table."""
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

    if start is None or end is None:
        return text

    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    block_lines = [
        line
        for idx, line in enumerate(lines[start:end])
        if idx == 0 or not key_pattern.match(line)
    ]
    return "".join(lines[:start] + block_lines + lines[end:])


def _enable_codex_feature(
    feature_name: str,
    *,
    deprecated_feature_names: tuple[str, ...] = (),
) -> None:
    """Enable a Codex feature flag in ~/.codex/config.toml."""
    config_path = Path.home() / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = config_path.read_text() if config_path.exists() else ""
    for deprecated_feature_name in deprecated_feature_names:
        existing = _remove_toml_table_key(existing, "features", deprecated_feature_name)
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
    codex_bin = _find_binary("codex")
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


def _enabled_plugin_keys(settings_path: Path, plugin_name: str) -> list[str]:
    """Return the fully-qualified enabledPlugins keys for a plugin, in file order."""
    if not settings_path.exists():
        return []
    try:
        data = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, ValueError):
        return []

    enabled_plugins = data.get("enabledPlugins", {})
    if not isinstance(enabled_plugins, dict):
        return []

    return [
        key
        for key, enabled in enabled_plugins.items()
        if enabled is True and (key == plugin_name or key.startswith(f"{plugin_name}@"))
    ]


def _plugin_enabled_in_settings(settings_path: Path, plugin_name: str) -> bool:
    """Return True when the plugin is enabled in a Claude settings file."""
    return bool(_enabled_plugin_keys(settings_path, plugin_name))


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


def _pi_agent_dir() -> Path:
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".pi" / "agent"


def _pi_agents_md_target() -> Path:
    """Return Pi's global instructions file."""
    return _pi_agent_dir() / "AGENTS.md"


def install_pi_md(scope: str = "user", cwd: Path | None = None) -> None:
    """Install ormah instructions into Pi's AGENTS.md (global or project)."""
    if scope == "project":
        base = cwd or Path.cwd()
        target, label = base / "AGENTS.md", "./AGENTS.md"
    else:  # user — Pi's global instructions file
        target, label = _pi_agents_md_target(), "~/.pi/agent/AGENTS.md"
    _install_markdown_block(
        target,
        "pi_instructions.md",
        PI_AGENTS_MD_SENTINEL_START,
        PI_AGENTS_MD_SENTINEL_END,
    )
    ok(f"Instructions added to {label}")


def install_pi_agents() -> None:
    """Install the Ormah maintenance subagent prompt into Pi's agent directory."""
    target = _pi_agent_dir() / "agents"
    target.mkdir(parents=True, exist_ok=True)
    content = resources.files("ormah").joinpath("agents/ormah-pi-maintenance.md").read_text()
    (target / "ormah-maintenance.md").write_text(content)
    ok("Agent definition installed — ormah-maintenance subagent available in Pi")


def _pi_settings_path() -> Path:
    return _pi_agent_dir() / "settings.json"


def _pi_source_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []

    sources: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            sources.append(entry)
        elif isinstance(entry, dict):
            for key in ("source", "path", "package"):
                candidate = entry.get(key)
                if isinstance(candidate, str):
                    sources.append(candidate)
                    break
    return sources


def _is_ormah_pi_source(source: str) -> bool:
    normalized = source.lower()
    return (
        "ormah-pi" in normalized
        or "r-spade/ormah" in normalized
        or "integrations/pi-plugin/ormah-pi.ts" in normalized
    )


def _pi_registered_sources() -> list[str]:
    settings_path = _pi_settings_path()
    if not settings_path.exists():
        return []
    try:
        data = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []

    sources = [
        *_pi_source_values(data.get("packages")),
        *_pi_source_values(data.get("extensions")),
    ]
    return [source for source in sources if _is_ormah_pi_source(source)]


def _remove_pi_settings_entries() -> None:
    settings_path = _pi_settings_path()
    if not settings_path.exists():
        return
    try:
        data = json.loads(settings_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return
    if not isinstance(data, dict):
        return

    changed = False
    for key in ("packages", "extensions"):
        value = data.get(key)
        if isinstance(value, str):
            if _is_ormah_pi_source(value):
                data.pop(key)
                changed = True
            continue
        if not isinstance(value, list):
            continue

        filtered = [
            entry
            for entry in value
            if not any(_is_ormah_pi_source(source) for source in _pi_source_values([entry]))
        ]
        if len(filtered) != len(value):
            changed = True
            if filtered:
                data[key] = filtered
            else:
                data.pop(key)

    if changed:
        settings_path.write_text(json.dumps(data, indent=2) + "\n")


def _pi_extension_registered() -> bool:
    return bool(_pi_registered_sources())


def configure_pi_extension(ormah_bin: str) -> None:
    """Ensure the official Ormah Pi package is installed.

    Pi has no external hooks.json or MCP config to write — the ormah-pi extension
    registers its own before_agent_start whisper hook and HTTP-proxied memory tools,
    so wiring Pi means installing the package plus Ormah's instructions and agent.
    """
    del ormah_bin  # Kept for compatibility with other configure_* helpers.
    if _pi_extension_registered():
        ok("ormah-pi extension detected — whisper + memory tools active after /reload")
        return

    pi_bin = _find_binary("pi")
    if pi_bin is None:
        raise RuntimeError("Pi is not installed")

    result = subprocess.run(
        [pi_bin, "install", PI_PACKAGE_SOURCE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"Could not install {PI_PACKAGE_SOURCE}: {detail}")
    if not _pi_extension_registered():
        raise RuntimeError(f"Pi did not register {PI_PACKAGE_SOURCE} in settings.json")
    ok("ormah-pi extension installed — whisper + memory tools active after /reload")


def _remove_pi_extension() -> None:
    sources = _pi_registered_sources()
    if not sources:
        return

    pi_bin = _find_binary("pi")
    if pi_bin is not None:
        for source in dict.fromkeys(sources):
            result = subprocess.run(
                [pi_bin, "remove", source],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "unknown error").strip()
                raise RuntimeError(f"Could not remove {source}: {detail}")
    _remove_pi_settings_entries()
    ok("Removed ormah-pi extension from Pi")


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

    cleaned_hooks, changed = _strip_ormah_hooks(hooks_top)

    if changed:
        if cleaned_hooks:
            data["hooks"] = cleaned_hooks
        else:
            data.pop("hooks", None)
        _atomic_write(str(hooks_path), json.dumps(data, indent=2) + "\n")
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
    """Write env dict to the global config file, preserving comments and ordering.

    Existing KEY= lines are updated in place; keys absent from `env` are dropped;
    full-line comments, blank lines, and ordering are kept verbatim; new keys
    append at end.

    CONTRACT: callers MUST pass the FULL env (from `_read_env_file()`) unless they
    intentionally want absent keys removed — a partial dict deletes the missing
    user keys. Non-goal: an inline trailing comment on a key whose VALUE this call
    rewrites is dropped (full-line comments and untouched keys keep theirs).
    """
    original = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in original:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in env and key not in seen:
                original_value = stripped.partition("=")[2].strip()
                if env[key] == original_value:
                    out.append(line)
                else:
                    out.append(f"{key}={env[key]}")
                seen.add(key)
            # key removed by caller -> drop the line
        else:
            out.append(line)  # comment / blank / other -> verbatim
    for key, value in env.items():
        if key not in seen:
            out.append(f"{key}={value}")
    _atomic_write(str(ENV_PATH), "\n".join(out) + "\n", mode=0o600)


def generate_server_wrapper(ormah_bin: str) -> Path:
    """Generate daemon wrapper with explicit, scoped API-key inheritance."""
    ENV_DIR.mkdir(parents=True, exist_ok=True)

    script = f"""\
#!/usr/bin/env bash
set -euo pipefail

# Load Ormah config without executing arbitrary shell from the env file.
_env_file="$HOME/.config/ormah/.env"
if [ -f "$_env_file" ]; then
    while IFS= read -r line; do
        case "$line" in
            ORMAH_*=*) export "$line" ;;
        esac
    done < "$_env_file"
fi

# Import exactly one approved API key into the daemon, only after explicit opt-in.
if [ "${{ORMAH_LLM_PROVIDER:-none}}" != "none" ] \\
   && [ "${{ORMAH_LLM_INHERIT_API_KEY:-false}}" = "true" ] \\
   && [ -n "${{ORMAH_LLM_API_KEY_ENV_VAR:-}}" ]; then
    _allowed_llm_keys=" ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY "
    _allowed_llm_keys="${{_allowed_llm_keys}}GROQ_API_KEY MISTRAL_API_KEY "
    _allowed_llm_keys="${{_allowed_llm_keys}}COHERE_API_KEY AZURE_API_KEY "
    case "$_allowed_llm_keys" in
        *" $ORMAH_LLM_API_KEY_ENV_VAR "*)
            while IFS= read -r line; do
                key="${{line%%=*}}"
                if [ "$key" = "$ORMAH_LLM_API_KEY_ENV_VAR" ]; then
                    export "$line"
                    break
                fi
            done < <("${{SHELL:-/bin/bash}}" -lic 'env' 2>/dev/null || true)
            ;;
    esac
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


def _disable_llm(env: dict[str, str]) -> None:
    """Disable server-side LLM use and clear any daemon key inheritance policy."""
    env["ORMAH_LLM_PROVIDER"] = "none"
    env.pop("ORMAH_LLM_MODEL", None)
    for key in _LLM_KEY_POLICY_KEYS:
        env.pop(key, None)


def _persist_env_delta(before: dict[str, str], after: dict[str, str]) -> None:
    """Apply this setup action's changes without overwriting concurrent writers."""

    from ormah.cloud.settings import persist_settings_delta

    persist_settings_delta(before, after)


def _enable_llm(
    env: dict[str, str],
    provider: str,
    model: str,
    *,
    api_key_var: str | None = None,
    inherit_api_key: bool = False,
) -> None:
    """Persist LLM policy. Never persist the actual API key value."""
    env["ORMAH_LLM_PROVIDER"] = provider
    env["ORMAH_LLM_MODEL"] = model
    for key in _LLM_KEY_POLICY_KEYS:
        env.pop(key, None)
    if api_key_var and inherit_api_key:
        env["ORMAH_LLM_API_KEY_ENV_VAR"] = api_key_var
        env["ORMAH_LLM_INHERIT_API_KEY"] = "true"


def configure_llm() -> None:
    """Interactive LLM provider setup for background analysis."""

    print("\n  Optional: enable intelligent Ormah services?")
    print("  Ormah works without an API key. With one, Ormah can maintain itself,")
    print("  backfill transcripts, and keep your memory graph cleaner over time.")
    print("  Cloud AI providers are called through their APIs, and API costs will apply.")

    try:
        answer = input("\n  Enable intelligent services? (y/N) ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        env = _read_env_file()
        before = dict(env)
        _disable_llm(env)
        _persist_env_delta(before, env)
        print()
        info("Server-side LLM disabled — core memory works without one")
        info("Run 'ormah setup' again to enable later")
        return

    print("\n  Which provider should Ormah use for background maintenance?\n")

    display_names = [p[0] for p in LLM_PROVIDERS]
    choice = _prompt_choice("", display_names, allow_skip=False)

    if choice is None:
        choice = len(LLM_PROVIDERS) - 1  # "None" option

    display_name, provider, api_key_var, default_model = LLM_PROVIDERS[choice]

    # Handle "None" selection
    if provider == "none":
        env = _read_env_file()
        before = dict(env)
        _disable_llm(env)
        _persist_env_delta(before, env)
        print()
        info("No LLM configured \u2014 core memory works without one")
        info("Run 'ormah setup' again to add an LLM later")
        return

    env = _read_env_file()
    before = dict(env)

    if api_key_var:
        hint = _cost_hint(default_model)
        existing_key = os.environ.get(api_key_var, "")
        if existing_key:
            print(f"\n  Found {api_key_var} in your environment.")
            print(f"  Model: {default_model} ({hint})")
            print("  Ormah will not copy the key value into its config.")
            print("  If allowed, the daemon inherits only this selected key.")
            try:
                prompt = f"\n  Allow Ormah to use {api_key_var}? (y/N) "
                key_answer = input(prompt).strip().lower()
            except EOFError:
                key_answer = ""
            if key_answer in ("y", "yes"):
                _enable_llm(
                    env,
                    provider,
                    default_model,
                    api_key_var=api_key_var,
                    inherit_api_key=True,
                )
                ok(f"Enabled {display_name} with {default_model}")
            else:
                _disable_llm(env)
                info("Server-side LLM disabled — no API key will be inherited")
        else:
            if os.environ.get("SHELL", "").endswith("zsh"):
                shell_profile = "~/.zshrc"
            else:
                shell_profile = "~/.bashrc"
            warn(f"No {api_key_var} found in your environment")
            print(f"  Add it to your shell profile ({shell_profile}):")
            print(f"    export {api_key_var}=your-key-here")
            print("  Then restart your shell and run 'ormah setup' again.")
            _disable_llm(env)
    else:
        # Ollama — no key needed
        _enable_llm(env, provider, default_model)
        ok(f"Using {display_name} with model '{default_model}'")
        info("Make sure Ollama is running: https://ollama.com")

    _persist_env_delta(before, env)


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
    from ormah.background.session_watcher import (
        _is_subagent_transcript,
        _space_from_encoded_dir,
    )

    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return []

    transcripts: list[tuple[Path, str | None, float]] = []
    for jsonl_file in projects_dir.rglob("*.jsonl"):
        if _is_subagent_transcript(jsonl_file):
            continue  # internal agent scratch — never backfill (see session_watcher)
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


def configure_agent_maintenance(agents: list[AgentDescriptor]) -> bool:
    """Ask whether to enable automatic agent-backed maintenance.

    Returns True if maintenance was enabled, False if skipped.
    """
    if not agents:
        return False
    agent_label = " or ".join(agent.name for agent in agents)

    print(f"\n  Use {agent_label} for automatic memory maintenance?")
    print("  (Runs judgment-heavy graph maintenance in the background when due,")
    print("   at most once every 24 hours by default. No separate API key needed.)")
    try:
        answer = input("\n  Enable? (Y/n) ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("n", "no"):
        env = _read_env_file()
        before = dict(env)
        env["ORMAH_CLAUDE_MAINTENANCE_ENABLED"] = "true"
        _persist_env_delta(before, env)
        if any(agent.id == "codex" for agent in agents):
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

    cleaned_hooks, changed = _strip_ormah_hooks(hooks_top)

    if changed:
        if cleaned_hooks:
            data["hooks"] = cleaned_hooks
        else:
            data.pop("hooks", None)
        _atomic_write(str(settings_path), json.dumps(data, indent=2) + "\n")
        ok("Removed whisper hooks from ~/.claude/settings.json")
    else:
        info("No ormah hooks found in settings.json")


def _remove_mcp_registration() -> None:
    """Unregister ormah MCP server from supported AI clients."""
    import platform as _platform

    # Claude Code
    claude_bin = _find_binary("claude")
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
    codex_bin = _find_binary("codex")
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

    _atomic_write(str(config_path), json.dumps(data, indent=2) + "\n")
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


def _remove_pi_md_block() -> None:
    """Remove the ormah instructions block from ~/.pi/agent/AGENTS.md."""
    target = _pi_agents_md_target()
    if _remove_markdown_block(
        target, PI_AGENTS_MD_SENTINEL_START, PI_AGENTS_MD_SENTINEL_END, "~/.pi/agent/AGENTS.md"
    ):
        ok("Removed ormah block from ~/.pi/agent/AGENTS.md")


def _remove_pi_agents() -> None:
    """Remove the Ormah maintenance subagent from Pi's agent directory."""
    agent_file = _pi_agent_dir() / "agents" / "ormah-maintenance.md"
    if agent_file.exists():
        agent_file.unlink()
        ok("Removed ormah-maintenance Pi agent definition")


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


def _uv_tool_install_candidates() -> tuple[list[Path], list[Path]]:
    """Return uv-tool paths owned by the Ormah install.

    The desktop app installs the Python package with a bundled uv sidecar.
    That creates the normal uv tool layout but does not guarantee a `uv`
    executable is available later on the user's PATH during uninstall.
    """
    home = Path.home()

    shims: list[Path] = [home / ".local" / "bin" / "ormah"]
    tool_dirs: list[Path] = [home / ".local" / "share" / "uv" / "tools" / "ormah"]

    uv_tool_bin_dir = os.environ.get("UV_TOOL_BIN_DIR")
    if uv_tool_bin_dir:
        shims.append(Path(uv_tool_bin_dir) / ("ormah.exe" if os.name == "nt" else "ormah"))

    uv_tool_dir = os.environ.get("UV_TOOL_DIR")
    if uv_tool_dir:
        tool_dirs.append(Path(uv_tool_dir) / "ormah")

    current_prefix = Path(sys.prefix).resolve(strict=False)
    if current_prefix.name == "ormah" and current_prefix.parent.name == "tools":
        tool_dirs.append(current_prefix)

    # Preserve order while de-duplicating.
    return list(dict.fromkeys(shims)), list(dict.fromkeys(tool_dirs))


def _remove_uv_tool_install_files() -> bool:
    """Best-effort cleanup for uv-installed Ormah command files.

    This only removes the known Ormah shim and uv tool environment paths. It
    intentionally does not delete arbitrary `ormah` executables discovered via
    PATH such as Homebrew or system-managed installs. A command at the standard
    desktop uv-tool shim path is treated as part of Ormah's managed install.
    """
    removed = False
    shims, tool_dirs = _uv_tool_install_candidates()

    for shim in shims:
        try:
            if not shim.exists() and not shim.is_symlink():
                continue
            if shim.name not in {"ormah", "ormah.exe"}:
                warn(f"Refusing to delete unexpected command path: {shim}")
                continue
            if shim.is_dir() and not shim.is_symlink():
                warn(f"Refusing to delete directory where command shim was expected: {shim}")
                continue
            shim.unlink()
            ok(f"Deleted command shim: {shim}")
            removed = True
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not delete command shim {shim}: {exc}")

    for tool_dir in tool_dirs:
        try:
            if not tool_dir.exists():
                continue
            if tool_dir.name != "ormah":
                warn(f"Refusing to delete unexpected uv tool directory: {tool_dir}")
                continue
            if tool_dir.is_symlink() or not tool_dir.is_dir():
                warn(f"Refusing to delete unexpected uv tool path: {tool_dir}")
                continue
            shutil.rmtree(tool_dir)
            ok(f"Deleted uv tool environment: {tool_dir}")
            removed = True
        except Exception as exc:  # noqa: BLE001
            warn(f"Could not delete uv tool environment {tool_dir}: {exc}")

    return removed


def _cloud_recovery_paths(config_dir: Path) -> tuple[Path, ...]:
    """Return recovery artifacts that uninstall must never delete."""
    return tuple(
        path
        for name in sorted(CLOUD_RECOVERY_FILENAMES)
        if (path := config_dir / name).exists() or path.is_symlink()
    )


class CloudRecoveryPreflightError(RuntimeError):
    """Raised when uninstall cannot prove cloud backups remain recoverable."""


@dataclass(frozen=True)
class CloudRecoveryPreflight:
    """Recovery artifacts verified before destructive uninstall work begins."""

    paths: tuple[Path, ...]
    kit_regenerated: bool = False


def _store_id_for_uninstall(memory_dirs: list[Path]) -> str | None:
    """Return the single store id uninstall is about to remove, if present."""
    from ormah.cloud.keys import CloudKeyError, extract_store_id

    store_ids: dict[str, list[Path]] = {}
    for memory_dir in dict.fromkeys(memory_dirs):
        store_path = memory_dir / ".store_id"
        if not store_path.is_file():
            continue
        try:
            value = store_path.read_text(encoding="utf-8").strip()
            store_id = extract_store_id(f"store_id: {value}")
        except (CloudKeyError, OSError) as exc:
            raise CloudRecoveryPreflightError(
                f"Cannot validate cloud store identity at {store_path}: {exc}"
            ) from exc
        if store_id is None:  # Defensive: an explicit store_id line must parse or raise.
            raise CloudRecoveryPreflightError(
                f"Cannot validate cloud store identity at {store_path}."
            )
        store_ids.setdefault(store_id, []).append(store_path)

    if len(store_ids) > 1:
        paths = ", ".join(str(path) for found in store_ids.values() for path in found)
        raise CloudRecoveryPreflightError(
            "Multiple cloud store IDs would be deleted, but one recovery kit can only "
            f"represent one store: {paths}"
        )
    return next(iter(store_ids), None)


def _validated_identity_strings(path: Path) -> list[str]:
    """Read and cryptographically validate every age identity in a file."""
    from ormah.cloud.keys import load_identities, load_identity_strings

    strings = load_identity_strings(path)
    load_identities(path)
    return strings


def _prepare_cloud_recovery(
    config_dir: Path,
    memory_dirs: list[Path],
) -> CloudRecoveryPreflight:
    """Ensure uninstall leaves a complete key + store-id recovery path.

    A valid kit is left byte-for-byte untouched. A missing or stale kit is
    regenerated only when the current key and one authoritative store id are
    both available. Valid mismatched store ids fail closed because overwriting
    that kit could orphan a different store.
    """
    from ormah.cloud.crypto import CloudCryptoError
    from ormah.cloud.keys import CloudKeyError, extract_store_id, write_recovery_kit

    paths = _cloud_recovery_paths(config_dir)
    if not paths:
        return CloudRecoveryPreflight(())

    key_path = config_dir / "cloud.key"
    kit_path = config_dir / "ormah-recovery-kit.md"
    expected_store_id = _store_id_for_uninstall(memory_dirs)

    key_identities: list[str] | None = None
    if key_path.exists() or key_path.is_symlink():
        if not key_path.is_file():
            raise CloudRecoveryPreflightError(
                f"Cloud key is not a readable file: {key_path}"
            )
        try:
            key_identities = _validated_identity_strings(key_path)
        except (CloudKeyError, CloudCryptoError, OSError) as exc:
            raise CloudRecoveryPreflightError(
                f"Cloud key validation failed at {key_path}: {exc}"
            ) from exc

    kit_identities: list[str] | None = None
    kit_store_id: str | None = None
    kit_error: Exception | None = None
    if kit_path.exists() or kit_path.is_symlink():
        if not kit_path.is_file():
            kit_error = CloudRecoveryPreflightError(
                f"Recovery kit is not a readable file: {kit_path}"
            )
        else:
            try:
                kit_identities = _validated_identity_strings(kit_path)
            except (CloudKeyError, CloudCryptoError, OSError) as exc:
                kit_error = exc
            try:
                kit_store_id = extract_store_id(str(kit_path))
            except (CloudKeyError, OSError) as exc:
                kit_error = exc

    if (
        kit_identities
        and kit_store_id
        and (key_identities is None or kit_identities == key_identities)
        and (expected_store_id is None or kit_store_id == expected_store_id)
    ):
        return CloudRecoveryPreflight(_cloud_recovery_paths(config_dir))

    if key_identities is None:
        detail = f": {kit_error}" if kit_error else ""
        raise CloudRecoveryPreflightError(
            "No valid cloud key is available to repair the incomplete recovery "
            f"kit at {kit_path}{detail}"
        )
    if (
        expected_store_id is not None
        and kit_store_id is not None
        and kit_store_id != expected_store_id
    ):
        raise CloudRecoveryPreflightError(
            f"Recovery kit store ID {kit_store_id} does not match the store being "
            f"removed ({expected_store_id}); refusing to overwrite either store's kit."
        )
    target_store_id = expected_store_id or kit_store_id
    if target_store_id is None:
        raise CloudRecoveryPreflightError(
            "The cloud key exists, but no store ID is available in either the "
            "recovery kit or the memory store. Uninstall cannot guarantee recovery."
        )

    try:
        write_recovery_kit(
            target_store_id,
            key_path=key_path,
            kit_path=kit_path,
        )
        regenerated_identities = _validated_identity_strings(kit_path)
        regenerated_store_id = extract_store_id(str(kit_path))
    except (CloudKeyError, CloudCryptoError, OSError) as exc:
        raise CloudRecoveryPreflightError(
            f"Could not create a complete recovery kit at {kit_path}: {exc}"
        ) from exc
    if regenerated_identities != key_identities or regenerated_store_id != target_store_id:
        raise CloudRecoveryPreflightError(
            f"Recovery-kit verification failed after writing {kit_path}."
        )
    return CloudRecoveryPreflight(_cloud_recovery_paths(config_dir), kit_regenerated=True)


def _remove_config_preserving_cloud_recovery(config_dir: Path) -> tuple[Path, ...]:
    """Delete Ormah config while retaining zero-knowledge recovery material."""
    preserved = _cloud_recovery_paths(config_dir)
    if not preserved:
        shutil.rmtree(config_dir)
        return ()

    preserved_names = {path.name for path in preserved}
    for child in config_dir.iterdir():
        if child.name in preserved_names:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()

    ok(f"Deleted Ormah config from {config_dir}")
    for path in preserved:
        warn(f"Preserved cloud recovery material: {path}")
    return preserved


@dataclass
class DesktopUninstallState:
    """Desktop presence and the one artifact the Python CLI may remove."""

    system: str
    autostart_path: Path | None = None
    unrecognized_autostart: Path | None = None
    artifacts: list[Path] = field(default_factory=list)
    package_removal_command: str | None = None
    autostart_disabled: bool = False

    @property
    def detected(self) -> bool:
        return bool(
            self.autostart_path
            or self.unrecognized_autostart
            or self.artifacts
            or self.package_removal_command
        )


def _macos_autostart_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{DESKTOP_PRODUCT_NAME}.plist"


def _linux_autostart_path(home: Path) -> Path:
    return home / ".config" / "autostart" / f"{DESKTOP_PRODUCT_NAME}.desktop"


def _macos_autostart_bundle(path: Path) -> Path | None:
    """Return the app bundle named by a verified Tauri LaunchAgent."""

    if path.is_symlink() or not path.is_file() or path.name != "Ormah.plist":
        return None
    try:
        data = plistlib.loads(path.read_bytes())
    except Exception:  # A damaged plist is not safe to remove automatically.
        return None
    if not isinstance(data, dict) or data.get("Label") != DESKTOP_PRODUCT_NAME:
        return None
    arguments = data.get("ProgramArguments")
    if not isinstance(arguments, list) or not arguments or not isinstance(arguments[0], str):
        return None
    executable = Path(arguments[0])
    valid = (
        executable.is_absolute()
        and executable.name in {"ormah-desktop", DESKTOP_PRODUCT_NAME}
        and executable.parent.name == "MacOS"
        and executable.parent.parent.name == "Contents"
        and executable.parent.parent.parent.name == "Ormah.app"
    )
    return executable.parent.parent.parent if valid else None


def _is_ormah_macos_autostart(path: Path) -> bool:
    return _macos_autostart_bundle(path) is not None


def _linux_autostart_executable(path: Path) -> Path | None:
    """Return the executable named by a verified auto-launch desktop entry."""

    if path.is_symlink() or not path.is_file() or path.name != "Ormah.desktop":
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    required = {
        "[Desktop Entry]",
        "Type=Application",
        "Name=Ormah",
        "Terminal=false",
    }
    lines = {line.strip() for line in text.splitlines()}
    executable = next(
        (line.split("=", 1)[1].strip() for line in lines if line.startswith("Exec=")),
        "",
    )
    if not required.issubset(lines):
        return None
    if executable.endswith("/ormah-desktop") or executable.endswith(".AppImage"):
        return Path(executable)
    return None


def _is_ormah_linux_autostart(path: Path) -> bool:
    return _linux_autostart_executable(path) is not None


def _linux_debian_removal_command() -> str | None:
    """Return the normal command only when dpkg proves package ownership."""

    try:
        result = subprocess.run(
            ["dpkg-query", "-S", "/usr/bin/ormah-desktop"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or ":" not in result.stdout:
        return None
    package = result.stdout.split(":", 1)[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9.+-]+", package):
        return None
    return f"sudo apt remove {package}"


def _appimage_executable(entry: Path) -> Path | None:
    """Read an AppImage path for reporting only; never use it as a delete target."""

    if entry.is_symlink() or not entry.is_file():
        return None
    try:
        text = entry.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        if not line.startswith("Exec="):
            continue
        try:
            command = shlex.split(line.split("=", 1)[1])
        except ValueError:
            return None
        if not command:
            return None
        executable = Path(command[0]).expanduser()
        if executable.is_absolute() and executable.name.endswith(".AppImage"):
            return executable
    return None


def _inspect_desktop_installation(system: str | None = None) -> DesktopUninstallState:
    """Find desktop autostart and report-only artifacts without deleting anything."""

    system = system or platform.system()
    home = Path.home()
    state = DesktopUninstallState(system=system)
    if system == "Darwin":
        autostart = _macos_autostart_path(home)
        autostart_bundle = _macos_autostart_bundle(autostart)
        if autostart_bundle is not None:
            state.autostart_path = autostart
        elif autostart.exists() or autostart.is_symlink():
            state.unrecognized_autostart = autostart
        candidates = [
            MACOS_SYSTEM_APPLICATIONS_DIR / "Ormah.app",
            home / "Applications" / "Ormah.app",
            home / "Library" / "Application Support" / DESKTOP_BUNDLE_IDENTIFIER,
            home / "Library" / "WebKit" / DESKTOP_BUNDLE_IDENTIFIER,
        ]
        state.artifacts = [path for path in candidates if path.exists() or path.is_symlink()]
        if (
            autostart_bundle is not None
            and (autostart_bundle.exists() or autostart_bundle.is_symlink())
            and autostart_bundle not in state.artifacts
        ):
            state.artifacts.append(autostart_bundle)
    elif system == "Linux":
        autostart = _linux_autostart_path(home)
        autostart_executable = _linux_autostart_executable(autostart)
        if autostart_executable is not None:
            state.autostart_path = autostart
        elif autostart.exists() or autostart.is_symlink():
            state.unrecognized_autostart = autostart
        menu_entry = home / ".local" / "share" / "applications" / "ormah.desktop"
        if menu_entry.exists() or menu_entry.is_symlink():
            state.artifacts.append(menu_entry)
            appimage = _appimage_executable(menu_entry)
            if appimage is not None:
                state.artifacts.append(appimage)
            for size in ("128x128", "256x256"):
                icon = (
                    home
                    / ".local"
                    / "share"
                    / "icons"
                    / "hicolor"
                    / size
                    / "apps"
                    / "ormah-desktop.png"
                )
                if icon.exists() or icon.is_symlink():
                    state.artifacts.append(icon)
        if (
            autostart_executable is not None
            and autostart_executable.name.endswith(".AppImage")
            and (autostart_executable.exists() or autostart_executable.is_symlink())
            and autostart_executable not in state.artifacts
        ):
            state.artifacts.append(autostart_executable)
        state.package_removal_command = _linux_debian_removal_command()
    return state


def _disable_desktop_autostart(state: DesktopUninstallState) -> None:
    """Remove only the exact, revalidated Tauri autostart file."""

    path = state.autostart_path
    if path is None:
        return
    validator = (
        _is_ormah_macos_autostart
        if state.system == "Darwin"
        else _is_ormah_linux_autostart
    )
    if not validator(path):
        state.unrecognized_autostart = path
        warn(f"Refusing to remove changed desktop autostart entry: {path}")
        return
    try:
        path.unlink()
    except OSError as exc:
        warn(f"Could not disable Ormah Desktop autostart at {path}: {exc}")
        state.unrecognized_autostart = path
        return
    state.autostart_disabled = True
    ok(f"Disabled Ormah Desktop autostart: {path}")


def _print_desktop_uninstall_summary(state: DesktopUninstallState) -> bool:
    """Report exactly what remains; the hybrid CLI never deletes the desktop app."""

    complete = True
    if state.unrecognized_autostart is not None:
        warn("Desktop autostart could not be safely disabled; remove this entry manually:")
        warn(f"  {state.unrecognized_autostart}")
        complete = False
    elif state.autostart_path is not None and not state.autostart_disabled:
        warn(f"Desktop autostart remains enabled: {state.autostart_path}")
        complete = False

    if state.artifacts or state.package_removal_command:
        complete = False
        warn("Ormah Desktop remains installed; opening it can reinstall the backend:")
        for path in dict.fromkeys(state.artifacts):
            warn(f"  {path}")
        if state.system == "Darwin":
            info("Quit Ormah, then move Ormah.app to Trash.")
        if state.package_removal_command:
            info(f"Remove the desktop package with: {state.package_removal_command}")
        elif state.system == "Linux":
            info("Delete the Ormah AppImage after quitting the app.")
    return complete


def run_uninstall(yes: bool = False) -> None:
    """Remove Ormah while preserving zero-knowledge cloud recovery material."""
    print(
        "This will remove Ormah integrations, local memory, caches, and account "
        "configuration. Cloud recovery files are preserved.\n"
    )

    config_dir = Path.home() / ".config" / "ormah"
    recovery_paths = _cloud_recovery_paths(config_dir)
    if recovery_paths:
        warn("Cloud recovery material detected; uninstall will not delete it:")
        for path in recovery_paths:
            warn(f"  {path}")
        warn(
            "Keep these files safe. Deleting them can make encrypted cloud backups "
            "permanently unreadable."
        )
        print()

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

    from ormah.config import settings as _settings

    config_mem_dir = _settings.memory_dir
    if not config_mem_dir.is_absolute():
        config_mem_dir = Path.home() / config_mem_dir
    config_mem_dir = config_mem_dir.resolve()

    if recovery_paths:
        step("Verifying cloud recovery")
        try:
            recovery = _prepare_cloud_recovery(
                config_dir,
                list(filter(None, [live_data_dir, config_mem_dir])),
            )
        except CloudRecoveryPreflightError as exc:
            warn(str(exc))
            warn("Uninstall cancelled before removing any data or integrations.")
            return
        if recovery.kit_regenerated:
            ok(f"Recovery kit refreshed and verified: {config_dir / 'ormah-recovery-kit.md'}")
        else:
            ok("Cloud recovery material is complete and verified")

    # a. Disable the desktop's own login item before removing its Python
    # runtime. The hybrid CLI leaves the app/package for normal OS removal.
    desktop = _inspect_desktop_installation()
    if desktop.autostart_path is not None or desktop.unrecognized_autostart is not None:
        step("Disabling Ormah Desktop autostart")
        if desktop.unrecognized_autostart is not None:
            warn(
                "Uninstall cannot safely identify this desktop autostart entry: "
                f"{desktop.unrecognized_autostart}"
            )
            warn("Uninstall cancelled before removing the backend or integrations.")
            return
        _disable_desktop_autostart(desktop)
        if desktop.autostart_path is not None and not desktop.autostart_disabled:
            warn("Uninstall cancelled before removing the backend or integrations.")
            return

    # b. Stop daemon
    step("Stopping server")
    from ormah.server_manager import uninstall_autostart
    uninstall_autostart()

    # c. Remove Claude Code hooks
    step("Removing Claude Code hooks")
    _remove_claude_hooks()
    _remove_codex_hooks()

    # d. Remove MCP registration
    step("Removing MCP registration")
    _remove_mcp_registration()

    # d.5 Remove the Pi package registration before deleting Ormah's own files.
    step("Removing Pi extension")
    try:
        _remove_pi_extension()
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not remove ormah-pi automatically: {exc}")

    # e. Remove CLAUDE.md block
    step("Removing CLAUDE.md instructions")
    _remove_claude_md_block()
    _remove_codex_md_block()
    _remove_codex_agents()
    _remove_claude_agents()
    _remove_claude_commands()
    _remove_pi_md_block()
    _remove_pi_agents()

    # f. Delete data directories
    step("Deleting data directories")

    xdg_dirs = [
        Path.home() / ".local" / "share" / "ormah",
        Path.home() / ".cache" / "ormah",
        config_dir,
    ]
    data_dirs: list[Path] = list(xdg_dirs)

    # Add the live server's actual data dir if it falls outside the XDG tree.
    # Also add the config-derived path as a safety net (handles custom ORMAH_MEMORY_DIR).
    for candidate in filter(None, [live_data_dir, config_mem_dir]):
        if not any(candidate == d or str(candidate).startswith(str(d) + "/")
                   for d in xdg_dirs):
            if candidate not in data_dirs:
                data_dirs.append(candidate)

    for d in data_dirs:
        if d.exists():
            if d == config_dir:
                preserved = _remove_config_preserving_cloud_recovery(d)
                if not preserved:
                    ok(f"Deleted {d}")
            else:
                shutil.rmtree(d)
                ok(f"Deleted {d}")
        else:
            info(f"{d} not found — skipping")

    # g. Delete fastembed model cache
    step("Removing embedding model cache")
    _remove_fastembed_cache()

    # h. Uninstall the package
    step("Uninstalling ormah package")
    uv_uninstalled = False
    try:
        result = subprocess.run(
            ["uv", "tool", "uninstall", "ormah"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            ok("Package uninstalled via uv")
            uv_uninstalled = True
        else:
            warn("Could not uninstall via uv; checking for desktop uv tool files")
    except Exception:
        warn("Could not uninstall via uv; checking for desktop uv tool files")

    removed_tool_files = _remove_uv_tool_install_files()
    if not uv_uninstalled and not removed_tool_files:
        warn("Could not remove package files — remove manually with: uv tool uninstall ormah")

    desktop_complete = _print_desktop_uninstall_summary(desktop)
    package_complete = uv_uninstalled or removed_tool_files
    print()
    if desktop_complete and package_complete:
        ok("Ormah has been uninstalled")
    elif package_complete:
        warn("Ormah backend cleanup completed, but Ormah Desktop remains installed.")
    else:
        warn("Ormah cleanup is incomplete; follow the removal instructions above.")


# ---------------------------------------------------------------------------
# Agent registry — extensible, data-driven detection + wired-check per agent.
# Adding a new integration = one entry here, zero changes to callers or UI.
# ---------------------------------------------------------------------------


@dataclass
class AgentDescriptor:
    id: str
    name: str
    detect_fn: Callable[[], bool]
    is_wired_fn: Callable[[], bool]
    wire_fn: Callable[[], None]
    unwire_fn: Callable[[], None]
    supports_maintenance: bool = False
    # None = available on all platforms; ["darwin"] = macOS only, etc.
    platform: list[str] | None = field(default=None)


def _claude_code_plugin_provides_hooks() -> bool:
    """True when a user-scoped ormah plugin is enabled AND actually installed.

    Claude Code keeps the two states in two different files, and an enabled flag
    is not proof that a working plugin exists:
      - enabled:   ``enabledPlugins`` in ~/.claude/settings.json
      - installed: ``plugins[]`` in ~/.claude/plugins/installed_plugins.json,
                   carrying the scope and the resolved installPath.

    The two files must agree on the SAME fully-qualified key (e.g.
    ``ormah@some-market``), not merely both mention an "ormah"-prefixed key:
    a stale, still-installed marketplace entry must never stand in for the
    marketplace that is actually enabled, or a broken active install could
    hide behind a healthy but abandoned one.

    This predicate licenses deleting the user's own wiring, so it requires both,
    plus hooks/hooks.json AND .mcp.json under that installPath — the wire guard
    strips both the CLI hooks and the CLI's MCP entry, so proving only the
    hooks manifest is not enough: a half-finished update could ship hooks.json
    without yet shipping .mcp.json, and stripping the MCP entry with no
    plugin-provided replacement would cost the user remember/recall with the
    whisper still silently intact. A stale flag pointing at a missing cache
    dir or a half-finished update would otherwise leave the user with no
    whisper at all — silently.

    Only a user-scoped plugin counts: configure_claude_hooks writes to the global
    ~/.claude/settings.json, which serves every project, so those hooks are
    redundant only when the plugin is global too. A project-scoped plugin keeps
    its duplication rather than break the whisper everywhere else.

    Both manifests must also actually declare content, not merely exist:
    hooks.json must parse and contain at least one hook entry that
    ``_is_ormah_hook`` recognizes as Ormah's, and .mcp.json must parse and
    declare an ``ormah`` entry under ``mcpServers``. An interrupted update
    can leave empty placeholder files (``{"hooks": {}}`` / ``{"mcpServers":
    {}}``) that pass an ``is_file()`` check while providing nothing — that
    must not license the strip either.

    Fails open: any unreadable or unparseable config returns False, so setup
    wires exactly as it does today.
    """
    enabled_keys = _enabled_plugin_keys(Path.home() / ".claude" / "settings.json", "ormah")
    if not enabled_keys:
        return False

    registry_path = Path.home() / ".claude" / "plugins" / "installed_plugins.json"
    try:
        data = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False

    plugins = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(plugins, dict):
        return False

    for key in enabled_keys:
        entries = plugins.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("scope") != "user":
                continue
            install_path = entry.get("installPath")
            if not isinstance(install_path, str) or not install_path:
                continue
            install_dir = Path(install_path)
            if _hooks_manifest_wires_ormah(
                install_dir / "hooks" / "hooks.json"
            ) and _mcp_manifest_wires_ormah(install_dir / ".mcp.json"):
                return True
    return False


def _hooks_manifest_wires_ormah(hooks_json_path: Path) -> bool:
    """True when a plugin's hooks.json is a real manifest declaring an Ormah hook.

    An interrupted plugin update can leave hooks.json present but empty
    (``{"hooks": {}}``) or non-existent as JSON; either must not count as
    "the plugin provides hooks". Reuses ``_is_ormah_hook`` so a renamed event
    still counts — only the hook entry's command shape matters, not the
    event name.
    """
    try:
        data = json.loads(hooks_json_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return False
    for matchers in hooks.values():
        if not isinstance(matchers, list):
            continue
        for matcher in matchers:
            if not isinstance(matcher, dict):
                continue
            inner = matcher.get("hooks")
            if not isinstance(inner, list):
                continue
            if any(_is_ormah_hook(entry) for entry in inner):
                return True
    return False


def _mcp_manifest_wires_ormah(mcp_json_path: Path) -> bool:
    """True when a plugin's .mcp.json declares the ormah-mcp wrapper command.

    An interrupted plugin update can leave .mcp.json present but empty
    (``{"mcpServers": {}}``); that must not count as "the plugin provides
    the MCP server".
    """
    try:
        data = json.loads(mcp_json_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return False

    server = servers.get("ormah")
    if not isinstance(server, dict):
        return False

    command = server.get("command")
    if not isinstance(command, str) or not command.strip():
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return bool(parts) and Path(parts[0]).name == "ormah-mcp"


def _claude_code_detected() -> bool:
    return _find_binary("claude") is not None


def _claude_code_is_wired() -> bool:
    # The plugin ships the hooks and the MCP server — an install with a working
    # plugin is wired even when settings.json holds nothing of ours.
    if _claude_code_plugin_provides_hooks():
        return True
    # Check for ormah whisper hooks in settings.json and ormah MCP in .claude.json
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        data = json.loads(settings_path.read_text())
        hooks = data.get("hooks") or {}
        for matchers in hooks.values():
            if not isinstance(matchers, list):
                continue
            for matcher in matchers:
                if not isinstance(matcher, dict):
                    continue
                inner = matcher.get("hooks")
                if not isinstance(inner, list):
                    continue
                if any(_is_ormah_hook(entry) for entry in inner):
                    return True
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    claude_json = Path.home() / ".claude.json"
    try:
        data = json.loads(claude_json.read_text())
        return "ormah" in (data.get("mcpServers") or {})
    except (OSError, json.JSONDecodeError):
        return False


def _codex_detected() -> bool:
    return _find_binary("codex") is not None or (Path.home() / ".codex").exists()


def _codex_is_wired() -> bool:
    # Check for ormah whisper hooks in hooks.json or ormah MCP in config.toml
    hooks_path = Path.home() / ".codex" / "hooks.json"
    try:
        data = json.loads(hooks_path.read_text())
        hooks = data.get("hooks") or {}
        for matchers in hooks.values():
            if isinstance(matchers, list):
                for entry in matchers:
                    cmd = entry.get("command", "") if isinstance(entry, dict) else str(entry)
                    if "ormah whisper" in cmd:
                        return True
    except (OSError, json.JSONDecodeError):
        pass
    config_path = Path.home() / ".codex" / "config.toml"
    try:
        text = config_path.read_text()
        return "[mcp_servers.ormah]" in text
    except OSError:
        return False


def _claude_desktop_detected() -> bool:
    import platform as _platform
    if _platform.system() != "Darwin":
        return False
    return os.path.exists(
        os.path.expanduser("~/Library/Application Support/Claude")
    )


def _claude_desktop_is_wired() -> bool:
    config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    try:
        data = json.loads(config_path.read_text())
        return "ormah" in (data.get("mcpServers") or {})
    except (OSError, json.JSONDecodeError):
        return False


def _pi_detected() -> bool:
    return _find_binary("pi") is not None


def _pi_guidance_installed() -> bool:
    target = _pi_agents_md_target()
    try:
        text = target.read_text()
    except OSError:
        return False
    return PI_AGENTS_MD_SENTINEL_START in text and PI_AGENTS_MD_SENTINEL_END in text


def _pi_agent_installed() -> bool:
    agent_file = _pi_agent_dir() / "agents" / "ormah-maintenance.md"
    try:
        return "ormah_run_maintenance" in agent_file.read_text()
    except OSError:
        return False


def _pi_is_wired() -> bool:
    return _pi_extension_registered() and _pi_guidance_installed() and _pi_agent_installed()


def _claude_code_wire() -> None:
    # The plugin registers the same UserPromptSubmit/PreCompact/SessionEnd hooks
    # and the same MCP server. Wiring them again in ~/.claude/settings.json runs
    # both copies: the whisper fires twice per human turn, and no merge can dedupe
    # across the two files. The agent and slash command are namespaced by the
    # plugin (ormah:maintenance vs ormah-maintenance), so they are not duplicate
    # registrations — they stay installed, as does CLAUDE.md, which no plugin can
    # write.
    if _claude_code_plugin_provides_hooks():
        _remove_claude_hooks()
        _remove_mcp_from_json(Path.home() / ".claude.json")
        info(
            "Claude Code plugin already provides the hooks and MCP server "
            "— removed redundant CLI wiring"
        )
    else:
        ormah_bin = get_ormah_bin_path()
        configure_claude_hooks(ormah_bin)
        configure_claude_code_mcp(ormah_bin)

    install_claude_md()
    install_claude_agents()
    install_claude_commands()


def _claude_code_unwire() -> None:
    _remove_claude_hooks()
    _remove_mcp_from_json(Path.home() / ".claude.json")
    _remove_claude_md_block()
    _remove_claude_agents()
    _remove_claude_commands()


def _codex_wire() -> None:
    ormah_bin = get_ormah_bin_path()
    configure_codex_hooks(ormah_bin)
    configure_codex_mcp(ormah_bin)
    install_codex_md()
    install_codex_agents()


def _codex_unwire() -> None:
    _remove_codex_hooks()
    _remove_codex_mcp_config()
    _remove_codex_md_block()
    _remove_codex_agents()


def _claude_desktop_wire() -> None:
    ormah_bin = get_ormah_bin_path()
    configure_claude_desktop(ormah_bin)


def _claude_desktop_unwire() -> None:
    desktop_config = (
        Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    )
    _remove_mcp_from_json(desktop_config)


def _pi_wire() -> None:
    configure_pi_extension(get_ormah_bin_path())
    install_pi_md()
    install_pi_agents()


def _pi_unwire() -> None:
    _remove_pi_extension()
    _remove_pi_md_block()
    _remove_pi_agents()


AGENT_REGISTRY: list[AgentDescriptor] = [
    AgentDescriptor(
        id="claude_code",
        name="Claude Code",
        detect_fn=_claude_code_detected,
        is_wired_fn=_claude_code_is_wired,
        wire_fn=_claude_code_wire,
        unwire_fn=_claude_code_unwire,
        supports_maintenance=True,
    ),
    AgentDescriptor(
        id="codex",
        name="Codex CLI",
        detect_fn=_codex_detected,
        is_wired_fn=_codex_is_wired,
        wire_fn=_codex_wire,
        unwire_fn=_codex_unwire,
        supports_maintenance=True,
    ),
    AgentDescriptor(
        id="claude_desktop",
        name="Claude Desktop",
        detect_fn=_claude_desktop_detected,
        is_wired_fn=_claude_desktop_is_wired,
        wire_fn=_claude_desktop_wire,
        unwire_fn=_claude_desktop_unwire,
        platform=["darwin"],
    ),
    AgentDescriptor(
        id="pi",
        name="Pi",
        detect_fn=_pi_detected,
        is_wired_fn=_pi_is_wired,
        wire_fn=_pi_wire,
        unwire_fn=_pi_unwire,
        supports_maintenance=True,
    ),
]


def _get_agent(agent_id: str) -> AgentDescriptor:
    for agent in AGENT_REGISTRY:
        if agent.id == agent_id:
            return agent
    raise ValueError(f"Unknown agent: {agent_id!r}")


def _detected_agents() -> list[AgentDescriptor]:
    import platform as _platform

    current_os = _platform.system().lower()
    return [
        agent
        for agent in AGENT_REGISTRY
        if (agent.platform is None or current_os in agent.platform) and agent.detect_fn()
    ]


def wire_agent(agent_id: str) -> dict:
    """Wire ormah into a single agent by id. Returns {wired, errors}."""
    import platform as _platform
    agent = _get_agent(agent_id)
    current_os = _platform.system().lower()
    if agent.platform is not None and current_os not in agent.platform:
        return {"wired": [], "errors": {agent_id: f"Not available on {_platform.system()}"}}
    errors: dict[str, str] = {}
    with contextlib.redirect_stdout(sys.stderr):
        try:
            agent.wire_fn()
        except Exception as exc:
            errors[agent_id] = f"{type(exc).__name__}: {exc}"
    return {"wired": [agent_id] if not errors else [], "errors": errors}


def unwire_agent(agent_id: str) -> dict:
    """Remove ormah hooks/MCP/instructions for a single agent. Returns {unwired, errors}."""
    agent = _get_agent(agent_id)
    errors: dict[str, str] = {}
    with contextlib.redirect_stdout(sys.stderr):
        try:
            agent.unwire_fn()
        except Exception as exc:
            errors[agent_id] = f"{type(exc).__name__}: {exc}"
    return {"unwired": [agent_id] if not errors else [], "errors": errors}


def list_agents() -> list[dict]:
    """Return all agents with detection and wired status — for the UI agent panel.

    Each entry: {id, name, detected, wired, platform}.
    Safe to call from any context: pure filesystem checks, no side effects.
    """
    import platform as _platform
    current_os = _platform.system().lower()
    result = []
    for agent in AGENT_REGISTRY:
        # Include platform-specific agents but mark them so the UI can annotate.
        detected = agent.detect_fn()
        wired = agent.is_wired_fn() if detected else False
        result.append({
            "id": agent.id,
            "name": agent.name,
            "detected": detected,
            "wired": wired,
            "platform": agent.platform,
            "available_on_current_os": agent.platform is None or current_os in agent.platform,
        })
    return result


def detect_clients() -> dict[str, bool]:
    """Legacy flat detection dict — kept for backwards compatibility."""
    agents = list_agents()
    return {a["id"]: a["detected"] for a in agents}


def run_setup_json() -> dict:
    """Non-interactive agent wiring for the Mac app's one-click setup button.

    Wires hooks/MCP/guidance for every detected client and returns a structured
    result the app can render. Also preloads local retrieval models so a fresh
    desktop install has both the embedding model and whisper reranker cached.
    Deliberately narrow vs. run_setup(): no LLM prompts, no server start (the
    app owns the bundled server sidecar), no browser launch, no animations.

    Human-readable progress from model preload and the underlying configure_*
    helpers is sent to stderr so stdout stays clean JSON for the caller to
    parse.
    """
    ormah_bin = get_ormah_bin_path()
    detected_ids: list[str] = []
    wired: list[str] = []
    errors: dict[str, str] = {}
    warnings: dict[str, str] = {}

    with contextlib.redirect_stdout(sys.stderr):
        try:
            _preload_local_models()
        except Exception as exc:  # noqa: BLE001
            warnings["models"] = f"{type(exc).__name__}: {exc}"

        for agent in _detected_agents():
            detected_ids.append(agent.id)
            try:
                agent.wire_fn()
                wired.append(agent.id)
            except Exception as exc:  # noqa: BLE001
                errors[agent.id] = f"{type(exc).__name__}: {exc}"

    return {
        "ormah_bin": ormah_bin,
        "detected": detected_ids,
        "wired": wired,
        "errors": errors,
        "warnings": warnings,
    }


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

    # 2. Detect supported agents and offer maintenance upfront — no API key needed
    detected_agents = _detected_agents()
    maintenance_agents = [agent for agent in detected_agents if agent.supports_maintenance]
    agent_maintenance = False
    if maintenance_agents and not ci and not update and not skip_client_setup:
        step(f"{' and '.join(agent.name for agent in maintenance_agents)} detected")
        agent_maintenance = configure_agent_maintenance(maintenance_agents)

    # 3. Configure LLM — skip if agent-backed maintenance is handling background jobs
    if ci:
        env = _read_env_file()
        before = dict(env)
        _disable_llm(env)
        _persist_env_delta(before, env)
        info("CI mode — LLM set to none")
    elif update:
        env = _read_env_file()
        if (
            env.get("ORMAH_LLM_PROVIDER") == "litellm"
            and env.get("ORMAH_LLM_INHERIT_API_KEY") != "true"
        ):
            info("Cloud LLM key inheritance is disabled; run 'ormah setup' to opt in")
    elif agent_maintenance:
        env = _read_env_file()
        before = dict(env)
        _disable_llm(env)
        _persist_env_delta(before, env)
    else:
        configure_llm()

    # 4. Generate server wrapper
    wrapper_path = generate_server_wrapper(ormah_bin)

    # 4.5 Preload local models into Ormah's shared model cache
    _preload_local_models()

    # 5. A healthy port is not enough: setup guarantees the running backend is
    # owned by launchd/systemd, replacing a manual process when necessary.
    server_was_running = is_server_running()
    step("Restarting server" if server_was_running else "Starting server")
    server_ok = restart_with_autostart(
        ormah_bin,
        wrapper_path=str(wrapper_path),
        show_progress=True,
    )
    if server_ok:
        action = "Updated" if update else "Installed"
        ok(f"{action} auto-start (launches on login)")
    else:
        _diagnose_server_failure()

    if not skip_client_setup:
        for agent in detected_agents:
            step(f"Hooking up {agent.name}")
            agent.wire_fn()

        if not detected_agents:
            supported = ", ".join(agent.name for agent in AGENT_REGISTRY)
            warn(f"No supported agents detected ({supported})")
            info("You can manually configure MCP in your AI client:")
            print(f"    Command: {ormah_bin} mcp")
            info("Or run 'ormah setup' again after installing a supported agent")

    # 7. Cold start backfill (needs server + LLM)
    if server_ok and not ci:
        backfill_transcripts()

    # 8. Finale animation + completion message
    if not server_ok:
        step("Setup incomplete")
        warn("Ormah server did not start, so setup could not complete.")
        info("Fix the server startup error, then run 'ormah setup --update' again.")
        info("Check logs: ~/.local/share/ormah/logs/ormah.log")
        raise SystemExit(1)

    step("Setup complete")
    if not ci:
        play_finale()
    _print_setup_summary(ormah_bin)
    if not ci:
        webbrowser.open(f"http://localhost:{settings.port}")
