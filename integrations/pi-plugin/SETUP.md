# Setup — Ormah-Pi

End-to-end install for a **user**. Two pieces: the Ormah runtime (local memory server) and the Pi extension (the client that whispers + exposes tools).

## 1. Install the Ormah runtime

```bash
bash <(curl -fsSL https://ormah.me/install.sh)          # installs ormah + runs setup
# or, plugin-safe (server + models + autostart only, no Claude/Codex wiring):
# bash <(curl -fsSL https://ormah.me/install.sh) --no-setup
# ormah setup --skip-client-setup
```

`ormah setup` detects Pi on your PATH and wires it automatically: it installs `npm:ormah-pi`, writes the Ormah guidance block into `~/.pi/agent/AGENTS.md`, and installs the `ormah-maintenance` prompt into `~/.pi/agent/agents/`. If you used `--skip-client-setup`, install the extension manually and run step 3's `/ormah:setup` inside Pi instead.

Verify: `ormah server status` → running; graph UI at `http://localhost:8787`.

## 2. Install the Pi extension manually (only if setup skipped client wiring)

**From the package gallery (pi.dev/packages):**

```bash
pi install npm:ormah-pi
```

This writes `npm:ormah-pi` to `~/.pi/agent/settings.json` under `packages` and loads it. (`ormah-pi` appears on the gallery because its `package.json` carries the `pi-package` keyword.)

**From git (any tag/commit):**

```bash
pi install git:github.com/r-spade/ormah@main
# or a dedicated repo: pi install git:github.com/<owner>/ormah-pi@v0.1.0
```

**Try without installing** (one-off, current run only):

```bash
pi -e npm:ormah-pi
# or a local checkout:
pi -e ./integrations/pi-plugin/ormah-pi.ts
```

Project-local install (shared with your team via `.pi/settings.json`):

```bash
pi install npm:ormah-pi -l
```

Then `/reload` in Pi (or restart). Confirm with `/ormah:status` — it should report `connected · N mem`.

## 3. Finish wiring inside Pi (if you used `--skip-client-setup`)

```text
/ormah:setup
```

Runs `ormah setup --skip-client-setup` (server + models + autostart), then asks `ormah pi-md install --scope user` to write the canonical guidance block into `~/.pi/agent/AGENTS.md`. Then `/reload`. (If step 1's `ormah setup` already wired Pi, skip this.)

## 4. (Optional) Enable an LLM provider

Required only for transcript extraction (whisper store) and LLM-backed maintenance classification. Local recall, whisper, and the memory tools work without it.

```bash
# ~/.config/ormah/.env
ORMAH_LLM_PROVIDER=ollama            # or litellm
ORMAH_LLM_MODEL=llama3.2
ORMAH_LLM_BASE_URL=http://localhost:11434
# litellm example: ORMAH_LLM_MODEL=claude-haiku-4-5-20251001 + ANTHROPIC_API_KEY=...
```

Then `ormah server stop && ormah server start -d`.

## 5. (Optional) Enable agent-driven maintenance

```bash
ORMAH_CLAUDE_MAINTENANCE_ENABLED=true  # shared server setting for supported agents
```

The Ormah server appends `maintenance_due` when maintenance is due; the Pi extension relays that whisper without maintaining a second schedule. Run `/ormah:maintenance` (or let the agent run the tool when signaled) to perform the two-step `ormah_run_maintenance` flow. `ormah setup` can enable this setting interactively.

## Commands once installed

| Command | What it does |
|---|---|
| `/ormah:setup` | Install/repair Ormah runtime + write Pi guidance block |
| `/ormah:status` | Server health + memory stats |
| `/ormah:maintenance` | Run graph maintenance in-session |
| `/ormah:upgrade` | `uv tool upgrade ormah` + restart server |
| `/ormah:reload` | Reload Pi extensions/skills/prompts/themes |

## Troubleshooting

- **`/ormah:status` says "down"** — `ormah server start -d`, then re-check.
- **Extension not loaded after install** — run `/reload` in Pi, or restart Pi. `pi list` should show `ormah-pi`.
- **Whisper is silent** — expected on a fresh graph with no relevant memories; store some via `ormah_remember` and retry.
- **No memory extraction on session end** — confirm an LLM provider is configured and the session had ≥ `ORMAH_WHISPER_OUT_MIN_TURNS` (default 3) user turns.
- **Uninstall** — `pi remove npm:ormah-pi`; to tear down the runtime too, `ormah uninstall`.
