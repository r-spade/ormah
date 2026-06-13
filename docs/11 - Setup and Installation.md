# Setup and Installation

Verified against the current repository state on 2026-06-09.

Ormah ships with an interactive setup flow that configures the server, supported client integrations, and optional transcript backfill.

The core server is agent-agnostic. Setup currently installs the concrete integrations that exist today: Claude Code, Codex, and Claude Desktop MCP where applicable.

## Installation

### Terminal

```bash
bash <(curl -fsSL https://ormah.me/install.sh)
```

### Claude Code Plugin

Install entirely from within Claude Code — no terminal required:

1. Add the marketplace and install the plugin:
   ```
   /plugin marketplace add r-spade/ormah
   /plugin install ormah@ormah
   ```
2. Reload: `/reload-plugins`
3. Run `/ormah:setup`
4. Check that the Ormah MCP server is enabled via `/mcp` — if not, enable it there

`/ormah:setup` checks whether the `ormah` runtime is installed. If it is missing, it asks permission to run the shell installer with `--no-setup`, then runs `ormah setup --skip-client-setup` to start the server without overwriting any global Claude wiring. The plugin owns hooks, MCP, commands, and the maintenance agent — `ormah setup` only handles the server and models.

## Setup Wizard

**Code**: `src/ormah/setup.py`

`ormah setup` does several things:

1. finds the `ormah` binary
2. detects supported clients such as Claude Code / Codex
3. optionally enables agent-backed maintenance
4. optionally configures server-side LLM settings
5. generates `~/.config/ormah/ormah-server`
6. preloads embedding / reranker models
7. installs auto-start
8. waits for server health
9. installs supported client integrations
10. optionally offers transcript backfill

## Important Correction: Agent-Backed Maintenance

If the user chooses agent-backed maintenance during setup, the wizard sets:

```text
ORMAH_LLM_PROVIDER=none
```

That means setup does **not** keep a separate background LLM configured in parallel for those maintenance tasks. Older docs that imply both paths are configured together are misleading.

## Default LLM Settings vs Setup Choices

Repository defaults in `config.py` are:

- `llm_provider = none`
- `llm_model = claude-haiku-4-5-20251001`
- `llm_base_url = http://localhost:11434`
- `llm_num_predict = 4096`
- `llm_inherit_api_key = false`

`ormah setup` can rewrite the persisted `.env` to:

- an explicitly selected remote provider
- `ollama`
- `none`

Remote provider setup stores policy only. It may store `ORMAH_LLM_API_KEY_ENV_VAR=ANTHROPIC_API_KEY` and `ORMAH_LLM_INHERIT_API_KEY=true`, but it must not store the actual API key value.

## Hooks

The shared hook commands are `ormah whisper inject` and `ormah whisper store`; setup writes the client-specific configuration around those commands.

### Claude Code

Setup installs:

- `UserPromptSubmit -> ormah whisper inject`
- `PreCompact -> ormah whisper store`
- `SessionEnd -> ormah whisper store`

### Codex

Setup also has Codex integration:

- writes `~/.codex/hooks.json`
- enables the `codex_hooks` feature flag
- installs MCP/instruction support when available in the rest of setup

## Logs and Auto-Start

Auto-start uses:

- `launchd` on macOS
- `systemd --user` on Linux when available

Important correction:

- operational log path referenced by the CLI is `~/.local/share/ormah/logs/ormah.log`

Older docs that point to `~/Library/Logs/ormah/` or only to `journalctl` are not aligned with the current server-manager code and CLI messaging.

## Data Locations

| What | Path |
|---|---|
| memory files | `~/.local/share/ormah/memory/nodes/*.md` |
| SQLite db | `~/.local/share/ormah/memory/index.db` |
| config | `~/.config/ormah/.env` |
| wrapper | `~/.config/ormah/ormah-server` |
| whisper cursors | `~/.cache/ormah/whisper-cursors.json` |
| logs | `~/.local/share/ormah/logs/ormah.log` |

## Server Management

Supported commands:

```bash
ormah server start
ormah server start -d
ormah server stop
ormah server status
```

Health checks:

```bash
curl http://localhost:8787/admin/health
ormah server status
```

Note: `ormah status` is not the main top-level CLI entry in `src/ormah/cli.py`; the supported server-status path is `ormah server status`.

## Transcript Backfill

The setup flow can optionally discover recent transcript files, estimate cost, and ingest them.

This is separate from the always-on session watcher.

Important nuance: some automation paths are still client-specific even though the core memory engine is not. For example, the always-on session watcher currently watches Claude's project transcript location by default.

## Walkthrough Example

Typical first-time setup:

1. run `ormah setup`
2. choose whether maintenance should be agent-backed
3. if agent-backed maintenance is enabled, setup persists `ORMAH_LLM_PROVIDER=none`
4. generate wrapper and preload local models
5. install auto-start
6. register MCP and hooks
7. optionally backfill recent transcripts

## Code Anchors

- `src/ormah/setup.py`
- `src/ormah/server_manager.py`
- `src/ormah/cli.py`
