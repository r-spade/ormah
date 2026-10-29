# Setup and Installation

Installation paths, client wiring, and capture requirements checked against
`b56eda99` on 2026-10-01.

Ormah ships with an interactive setup flow that configures the server, supported client integrations, and optional transcript backfill.

Setup supports Claude Code, Codex CLI, Pi, and Claude Desktop on macOS.
Automatic Whisper depends on the client integration; sharing MCP tools alone
does not install a prompt hook.

## Installation

### Desktop app

Download Ormah for [macOS Apple Silicon](https://www.ormah.me/download/mac)
or [Linux x86_64](https://www.ormah.me/download/linux).
[All releases](https://github.com/r-spade/ormah/releases) lists the available
packages. Open the app and follow setup to connect your agents. The app
bootstraps its Python runtime; you do not need to install Python beforehand.

Desktop shows signed updates with **Update now** and **Later**. Installation
starts after your click; the app does not silently install or restart itself.

### Terminal

```bash
bash <(curl -fsSL https://ormah.me/install.sh)
```

The installer supports macOS and Linux, installs the runtime with uv, and runs
setup automatically. After a manual package installation or an installation
with `--no-setup`, run `ormah setup` yourself.

### Manual package install

If you already manage your Python tooling with uv:

```bash
uv tool install 'ormah[litellm]'
ormah setup
```

Installing the package alone does not configure the server or client integrations.

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

Plugin-safe setup skips wiring other clients. If you also use Codex, Pi, or
Claude Desktop, run the normal `ormah setup` to connect those clients.

### Pi

Normal setup detects Pi and installs its extension and guidance. If client
wiring was skipped, follow the [Pi setup guide](../integrations/pi-plugin/SETUP.md).
The manual extension install is `pi install npm:ormah-pi`.

## Try your first memory

In a connected agent, ask: “Remember that checkout retries must reuse the same
idempotency key.” Start a new session in the same project and ask: “What key
should checkout retries use?” To try a second agent, connect it to the same
Ormah store and project first.

Where the client exposes hook/context output, inspect it for the saved decision
before the reply. A correct answer alone does not prove automatic Whisper:
the agent might have explicitly called recall.

You can check storage and recall separately from your project directory:

```bash
ormah server status
ormah remember "Checkout retries reuse the same idempotency key." --type decision
ormah recall "checkout retries"
```

The server should report that it is running and recall should return the
decision. Open the [local graph](http://localhost:8787/ui) to inspect it.

If the server is unavailable, run `ormah server start -d`. If a client was
installed after Ormah, rerun setup and start a new client session. If recall
works but automatic context does not, check that the client supports the
integration below and that its hooks or extension are enabled. Whisper can
also stay silent when no relevant memory is selected.

## Setup Wizard

**Code**: `src/ormah/setup.py`

`ormah setup` does several things:

1. finds the `ormah` binary
2. detects supported clients such as Claude Code, Codex CLI, Pi, and Claude Desktop
3. optionally enables agent-backed maintenance
4. optionally configures server-side LLM settings
5. generates `~/.config/ormah/start-server.sh`
6. preloads embedding / reranker models
7. installs auto-start
8. waits for server health
9. installs supported client integrations
10. optionally offers transcript backfill

## Automatic recall, capture, and maintenance

Local storage, search, and Whisper retrieval do not need a separate model API
key. Agents can save memories directly through the memory tools. Extracting
memories from transcripts or watched notes needs an Ormah LLM provider.

If the user chooses agent-backed maintenance during setup, the wizard sets:

```text
ORMAH_LLM_PROVIDER=none
```

This leaves server-side LLM extraction off as well. Enable a provider separately
if you want transcript or note extraction. Agent-backed maintenance handles
graph cleanup; it does not itself enable the transcript watcher or replace its
extraction provider. See [capture and watchers](<10 - Hippocampus and Session Watcher.md>).

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

| Client | Memory tools | Automatic Whisper |
| --- | --- | --- |
| Claude Code | MCP | `UserPromptSubmit` hook |
| Codex CLI | MCP | `UserPromptSubmit` hook |
| Pi | Native extension | Before each agent turn |
| Claude Desktop on macOS | MCP | Tool access only; no prompt hook |

Use a client version that supports the listed hooks and reload or restart it
after configuration changes. These integrations assume a running local Ormah
server; a shared project name alone does not connect separate memory stores.

### Claude Code

Setup installs:

- `UserPromptSubmit -> ormah whisper inject`
- `PreCompact -> ormah whisper store`
- `SessionEnd -> ormah whisper store`

### Codex

Setup also has Codex integration:

- writes `~/.codex/hooks.json`
- enables the `hooks` feature flag (replacing the old `codex_hooks` name)
- installs `UserPromptSubmit -> ormah whisper inject`
- installs `Stop -> ormah whisper store`
- installs MCP/instruction support when available in the rest of setup

### Pi

The [Pi extension](../integrations/pi-plugin/README.md) supplies memory tools
over HTTP and injects Whisper before each agent turn. Its lifecycle capture
also requires a configured extraction provider. Setup installs the extension,
guidance, and maintenance prompt.

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
| wrapper | `~/.config/ormah/start-server.sh` |
| whisper cursors | `~/.cache/ormah/whisper-cursors.json` |
| logs | `~/.local/share/ormah/logs/ormah.log` |

Storage and retrieval are local by default. A connected agent's model processes
the memories supplied to it, and optional remote extraction providers process
the content sent to them. See [storage](<02 - Storage Layer.md>) and
[encrypted cloud backup and recovery](<15 - Cloud Backup, Verification, and Restore.md>)
for ownership and recovery details.

## Server Management

Supported commands:

```bash
ormah server start       # foreground; stops when this process exits
ormah server start -d    # supervised background service (recommended)
ormah server stop
ormah server status
```

`ormah server stop` also removes the supervised service. Ormah remains stopped
until `ormah setup` or `ormah server start -d` explicitly enables it again.

Health checks:

```bash
curl http://localhost:8787/admin/health
ormah server status
```

Note: `ormah status` is not the main top-level CLI entry in `src/ormah/cli.py`; the supported server-status path is `ormah server status`.

## Transcript Backfill

The setup flow can optionally discover recent transcript files, estimate cost, and ingest them.

This is separate from the optional session watcher, which is off by default.
Both backfill extraction and watcher extraction need an Ormah LLM provider.

When enabled, the session watcher uses Claude's project transcript location by
default and also includes the default Codex sessions directory if it exists.
Pi supplies its own lifecycle capture through the extension.

## Uninstall

`ormah uninstall` removes local data and integrations, but preserves the cloud
key and recovery kit. It stops if safe recovery material cannot be established.
It also disables a verified desktop login item while leaving the application
package itself for the platform's normal removal process. Before uninstalling,
read [backup and recovery](<15 - Cloud Backup, Verification, and Restore.md>).

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
