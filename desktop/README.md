# Ormah Desktop

A ruthlessly minimal Tauri v2 app. It is **onboarding + visibility**, not a new
product:

1. **Bundled runtime** — ships a `uv` sidecar that installs `ormah` from PyPI on
   first launch. No curl, no terminal. `~/.local/bin/ormah` is created
   automatically by `uv tool install`, giving hooks a stable binary path.
2. **One-click agent setup** — detects Claude Code / Claude Desktop / Codex / Pi and
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

## Runtime setup and diagnostics

The bundled app installs the exact package version from `pyproject.toml` with
`uv tool install --reinstall --force --python 3.11 ormah==<pin>`. An older or
broken runtime is repaired; an exact matching uv environment on another Python
is rebuilt on 3.11. A newer installed package is still reused. `--force` replaces
conflicting tool entry points; uv's tool environment is separate from Ormah's
memory and configuration. The app does not edit PATH or shell profiles.

Transient network failures get five total attempts with waits of 10, 20, 30 and
45 seconds (105 seconds total). Each uv process has a fixed ten-minute timeout,
including a progressing download; there is no idle/progress heuristic. First
installs may download substantial files, so the UI has no installation deadline.
After server startup begins it allows ten minutes before offering recovery, and
continues checking health afterward. The same poll loop refreshes sequenced setup
snapshots as a fallback for unavailable lifecycle events. An already-healthy older daemon cannot bypass
an active install or retry. A failed daemon command is reported unless
the health endpoint confirms a running server. That endpoint cannot establish
which macOS user owns a healthy server; multi-user ownership is unchanged.

The setup failure screen offers **Try again**, **Copy details**, and **Show log**.
Copy uses a native clipboard command, rather than browser clipboard availability.
Only the bundled bootstrap capability can copy or reveal setup diagnostics; the
remote graph receives no generic clipboard or filesystem access.

Setup diagnostics use Tauri's `app_log_dir()` and the bundle identifier
`dev.ormah.desktop`:

- macOS: `~/Library/Logs/dev.ormah.desktop/setup.log`
- Linux: `$XDG_DATA_HOME/dev.ormah.desktop/logs/setup.log` (normally
  `~/.local/share/dev.ormah.desktop/logs/setup.log`)

The log records Unix-millisecond timestamps, app/runtime/uv/platform versions,
Install/Upgrade/Reuse decisions, uv arguments, attempt timing and exit status,
classification, streamed uv output and daemon exit/output. Only proxy and UV
variable names are recorded. URL credentials and recognizable credentials/tokens
are redacted before writing. Partial output lines are buffered across events;
lines over 8 KiB are replaced with a redaction marker. A killed uv attempt flushes
its remaining partial line. Files are bounded to 1 MiB plus one `setup.log.1`
rotation, so very verbose runs can overwrite older evidence. Copy includes up to
24 KiB from each file, redacted again. Logging errors never prevent setup.

On Linux, a small internal output collector exits when daemon output writers
close. It can outlive the GUI because the CLI's fallback background server may
inherit stdout/stderr; waiting for pipe EOF in the GUI would otherwise hang
startup, and closing GUI-owned readers could break the independent daemon's
output. CLI exit status is collected independently. A stable `setup.lock` file
coordinates rotation between the GUI and collector. macOS launchd startup
captures the CLI's stdout/stderr directly.

The initial frameless window fits the primary monitor's logical work area with a
24-pixel margin on each edge, and its minimum size cannot exceed that area. If
monitor information is unavailable it retains the 1180×820 default.

### Verification

```sh
# Supply a local uv sidecar as described above before Rust tests.
cd desktop/src-tauri
CARGO_BUILD_JOBS=2 cargo test
cd ../ui
npm ci
npm test
npx tsc --noEmit
npm run build
```

The desktop frontend tests compile its pure setup model and bridge with the
existing TypeScript compiler and use Node's built-in test runner. They require no
additional test dependency. macOS Finder startup, native clipboard and window
placement need GUI verification; Linux compilation and browser layout checks do
not establish those behaviors. Reliability changes do not establish the cause
of the reported Mac install failure.
