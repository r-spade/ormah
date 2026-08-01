# Ormah Desktop

A ruthlessly minimal Tauri v2 app. It is **onboarding + visibility**, not a new
product:

1. **Bundled runtime** — ships a `uv` sidecar that installs `ormah` from PyPI on
   first launch. No curl, no terminal. `~/.local/bin/ormah` is created
   automatically by `uv tool install`, giving hooks a stable binary path.
2. **One-click agent setup** — detects Claude Code / Claude Desktop / Codex and
   wires hooks/MCP via `ormah setup --json`.
3. **Menubar presence** — tray icon whose title is the weekly *whispers-used*
   count, with a dropdown for stats and actions.
4. **In-app graph** — the existing web UI loads inside the window.
5. **Auto-update** — Tauri updater (appcast).
6. **Trusted recovery handoff** — fixed native commands save or open the
   canonical recovery kit without exposing its bytes or location to React.

## Architecture

```
Tauri shell (Rust, src-tauri/)
  ├─ tray.rs      tray icon + title (counter) + dropdown menu
  ├─ stats.rs     polls GET /stats every 60s → tray title
  ├─ sidecar.rs   checks ormah on PATH; if absent, runs bundled uv to install it
  └─ commands.rs  setup_agents (ormah setup --json), open graph, onboarding marker
       ↓ installs (first launch only)
binaries/uv-<triple>        ← bundled uv binary (downloaded at CI build time)
       ↓ installs
~/.local/bin/ormah           ← stable ormah CLI (created by uv tool install)
       ↓ serves
http://127.0.0.1:8787        ← existing FastAPI app + web UI
```

The product webview receives only the purpose-built `desktop-product-bridge`
commands. Recovery-kit Save uses `tauri-plugin-dialog` only inside Rust; the
remote graph capability is not granted generic dialog, filesystem, or shell
access. Rust bounded-reads the fixed canonical kit, writes the selected file
with owner-only permissions on Unix, reopens it without following symlinks,
and sends only its SHA-256 digest to a fixed capability-authenticated local
endpoint. Python independently validates the canonical store and full active
identity set before recording readiness under the store lock. The direct
`sha2` and `libc` dependencies provide digesting and Unix `O_NOFOLLOW` support.

## Build & run (dev)

```bash
# 1. Download the uv sidecar for your arch (Linux x86_64 shown).
TRIPLE=$(rustc -Vv | sed -n 's/host: //p')
curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/uv-${TRIPLE}.tar.gz" \
  | tar -xz -C /tmp
cp /tmp/uv-${TRIPLE}/uv desktop/src-tauri/binaries/uv-${TRIPLE}
chmod +x desktop/src-tauri/binaries/uv-${TRIPLE}

# 2. Generate the full icon set from the source icon (one-time).
cd desktop/src-tauri && cargo tauri icon icons/icon.png && cd -

# 3. Run (tray appears; on first launch uv installs ormah; onboarding shows).
cd desktop && cargo build
./src-tauri/target/debug/ormah-desktop
```

Subsequent runs skip the install step since `ormah` is already on PATH.

## Release (CI — macOS + Linux)

`.github/workflows/desktop-release.yml` triggers on `desktop-v*` tags. It
verifies the tag matches `tauri.conf.json`'s version, builds per-arch on
macOS (aarch64) and Linux (x86_64) runners, downloads the matching `uv`
binary into `binaries/`, and attaches `.dmg`, `.AppImage`, and `.deb`
artifacts to the GitHub release.

Auto-update: `createUpdaterArtifacts` makes the build emit signed updater
bundles (`.AppImage.sig`, `.app.tar.gz` + `.sig`, signed with the
`TAURI_SIGNING_PRIVATE_KEY` secret). The publish job then generates
`latest.json` and uploads it to the rolling `desktop-latest` prerelease,
which the app's updater polls (`plugins.updater.endpoints` in
`tauri.conf.json`). The rolling release is a prerelease on purpose — GitHub's
`/releases/latest` alternates between desktop and Python releases, so the
feed needs its own stable URL.

The app pins which ormah Python package it installs: `build.rs` reads the
version from the repo's `pyproject.toml` at compile time (no manual sync).
On Linux, AppImage runs self-register an app-menu entry + icons under
`~/.local/share` on first launch; the `.deb` ships system-wide ones via dpkg.

macOS-only secrets required for signing/notarization:

| Secret | Purpose |
| --- | --- |
| `APPLE_CERTIFICATE` | base64 Developer ID Application `.p12` |
| `APPLE_CERTIFICATE_PASSWORD` | password for the `.p12` |
| `APPLE_SIGNING_IDENTITY` | e.g. `Developer ID Application: … (TEAMID)` |
| `APPLE_ID` | Apple ID email for notarization |
| `APPLE_APP_PASSWORD` | app-specific password for notarytool |
| `APPLE_TEAM_ID` | Apple Developer Team ID |
| `TAURI_SIGNING_PRIVATE_KEY` | Tauri updater private key |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | password for the updater key |

Also set `plugins.updater.pubkey` in `tauri.conf.json` and
`plugins.updater.endpoints`.

## Known follow-ups

- **Model downloads on first run** — embedding and whisper-reranker weights are
  downloaded by FastEmbed during server startup into Ormah's shared model cache at
  `~/.local/share/ormah/models`. Server readiness waits for each download attempt, so
  provisioning does not depend on whether desktop onboarding runs. If the reranker
  download fails, Ormah remains available with conservative embedding-only whisper and
  retries provisioning on the next server start.
- **Python runtime updates** — each desktop build pins an Ormah package version. On
  launch the app installs a missing/older version with `uv tool install`, then restarts
  the daemon so the new Python code and migrations are active. It never downgrades a
  newer installed runtime.
- **Updater appcast** — the CI `publish` job attaches artifacts but the
  `latest.json` regeneration/upload is a TODO.

## Backend coverage

- `GET /stats` — `tests/test_api/test_stats.py`
- `GET /agent/clients` + `ormah setup --json` — `tests/test_setup_json.py`
