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
# 1. Download the pinned, checksum-verified sidecar for your arch.
bash desktop/scripts/build-sidecar.sh

# 2. Generate the full icon set from the source icon (one-time).
cd desktop/src-tauri && cargo tauri icon icons/icon.png && cd -

# 3. Run (tray appears; on first launch uv installs ormah; onboarding shows).
cd desktop && cargo build
./src-tauri/target/debug/ormah-desktop
```

Subsequent runs skip the install step since `ormah` is already on PATH.

## Release (CI — macOS + Linux)

The [release runbook](../CONTRIBUTING.md#coupled-python-and-desktop-releases)
contains the complete sequence and commands. Each Python release requires a
reviewed desktop version bump, followed by a new desktop candidate. The desktop
workflow supports an owner dispatch and `desktop-v*` tags, keeps the bounded PyPI
propagation check, signs/notarizes macOS, signs Linux updater packages, and puts
all artifacts into a **draft** release with `candidate.json` checksums.

The updater feed is generated only by the separate owner promotion command,
after exact-artifact clean-VM acceptance and manual macOS/Linux checks pass.
`latest.json` includes macOS, AppImage and Debian entries and lives on the rolling
`desktop-latest` prerelease, which keeps its URL stable without taking over
GitHub's `/releases/latest`. A failed acceptance leaves that feed untouched.

uv is pinned in `desktop/uv-version`. The build script verifies Astral's SHA-256
before extraction, including repeat builds, so a stale cached sidecar is replaced.
Fresh terminal installs use the same version; existing user uv is preserved.

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
| `APPLE_API_ISSUER` | App Store Connect issuer ID |
| `APPLE_API_KEY` | App Store Connect key ID |
| `APPLE_API_KEY_P8` | base64 App Store Connect private key |
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
- **Release acceptance** — the first real Tart run and first signed draft using
  the new gate must be verified on the owner's Mac. Linux checks remain manual.

## Backend coverage

- `GET /stats` — `tests/test_api/test_stats.py`
- `GET /agent/clients` + `ormah setup --json` — `tests/test_setup_json.py`


## Clean-machine acceptance

Run on an Apple Silicon Mac with Tart installed and owner GitHub CLI login.
GitHub-hosted macOS runners cannot run this Tart test nested; no self-hosted runner
is registered. The host also needs Python 3.11+, OpenSSH, curl and shasum.

Prepare a stopped clean base once (do not install developer tooling in it):

```bash
tart clone ghcr.io/cirruslabs/macos-tahoe-vanilla:latest ormah-base
tart set ormah-base --display 1920x1200
tart run ormah-base
# Verify auto-login to admin, Remote Login/SSH enabled, admin/admin works,
# no uv, Homebrew, real Python, Xcode tools, Ormah or Ormah/model data.
# Verify sudo -n works for reboot. Shut down this base cleanly in the guest.
```

Record the source image digest and Tart/macOS versions with release evidence.
Never prepare the base by deleting an Ormah install from a used machine. The base
must auto-login to its GUI session. `open -a` over SSH is checked by verifying the
console user, the GUI launchd domain, the app process, and backend health; the first
real run still needs owner verification. The image defaults to Gatekeeper
assessments disabled (`spctl --status`); the script records this and applies
quarantine metadata, but cannot prove Gatekeeper first-open behavior.

From a checkout matching the candidate, download its draft assets with `gh release
download` and run:

```bash
bash scripts/acceptance/macos-vm.sh app --dmg /path/to/exact-draft.dmg \
  --expected-version X.Y.Z --desktop-version A.B.C --output /path/to/evidence/app
bash scripts/acceptance/macos-vm.sh terminal --output /path/to/evidence/terminal
# Optional: --base another-clean-base, --keep, --installer /path/or/https-url
```

Each invocation creates a unique clone, sets 1920x1200, boots, and waits up to
three minutes for SSH. Each check has a timeout (first model/backend readiness:
15 minutes; terminal installation/setup: 20 minutes). It records baseline tools,
Gatekeeper, exact input SHA-256, runtime/desktop/uv versions, Python 3.11, both
nonempty ONNX model weights, VM-user server ownership, a unique remember/recall,
and reboot/autostart/recall. App mode checks the process before and after reboot;
terminal mode checks a real new login shell before and after reboot. It does not
manually launch the app or daemon after reboot. Only the created clone is stopped
and deleted on success; `--keep` retains it. Any failure exits nonzero and retains
the clone with its unique name in `result.json`. Logs, timings, every check and
input artifact stay in the requested output directory. Passwords and known hosts
use private temporary files, never process arguments or the user's SSH files.
No host Ormah service or host localhost:8787 is contacted.

The terminal installer defaults to the public script, uses `--no-setup` then
`ormah setup --ci`, and needs no model API key. Interactive terminal setup is not
covered. All AI-client release checks use subscription sign-in only.

Manual release items (required in the owner promotion attestation):

- Download the exact candidate through Safari on a clean Mac with Gatekeeper
  **enabled**, then open DMG and app normally. Confirm signing/notarization and
  first-open UX. A disabled Gatekeeper VM cannot substitute for this check.
- Connect an AI tool using subscription sign-in, then remember/recall through it.
- Check onboarding at 1024x768 for clipped content and inaccessible controls.
- In a fresh clone, launch with HTTPS_PROXY/HTTP_PROXY/ALL_PROXY pointing at a dead
  VM-local port (for example 127.0.0.1:9). Verify install failure is shown, retry is
  available, and the UI stays responsive. Remove the proxy and confirm recovery.
  Apply proxies to the GUI launch environment with `launchctl setenv` inside the
  VM; setting only the SSH shell environment may not reach LaunchServices.
- Test exact Linux AppImage and deb packages on clean supported Linux machines,
  including startup, runtime version, models, persistence and login autostart.

If acceptance fails, preserve logs and VM, fix through review, and use a new
candidate version for changed bytes. See the runbook for transient retry, corrected
candidates, and recovery from a failed feed upload after publication. Real Tart
execution is **UNVERIFIED** by Linux-only development tests; stub tests exercise
script flow and evidence rejection, not virtualization or GUI behavior.
