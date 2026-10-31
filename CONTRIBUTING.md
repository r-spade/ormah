# Contributing to Ormah

Help improve an integration, clarify a guide, or report a reproducible bug. Use [GitHub Issues](https://github.com/r-spade/ormah/issues) for problems and proposals, or the [Ormah Discord](https://discord.gg/guBU6XweBu) for discussion.

## Development

Use Python 3.11+, Node.js/npm, and Make. Run the following in an activated Python virtual environment with pip available:


```bash
git clone https://github.com/r-spade/ormah.git
cd ormah
make install
python -m pytest
```

### Retrieval evals

Two golden-corpus eval harnesses measure retrieval quality from a source checkout:

```bash
make eval                       # whisper + recall evals with fail-below quality bars
ormah eval whisper run          # whisper pipeline eval (--show-failures, --category, --fail-below)
ormah eval recall run           # recall/retrieval eval (--fail-below)
ormah eval whisper mine         # mine provisional cases from your live whisper_log (read-only)
ormah eval whisper import-labels  # confirm mined labels after human review
```

Eval corpora are intentionally **local-only and gitignored**: they seed real
memories from the developer's own usage, so they never ship in the repo. See
`eval/whisper/corpus/README.md` for the case-design and honest-labeling rules.
CI runs tests and lint only; eval bars are enforced locally via `make eval`.

## Release Process


Releases are published through the manual GitHub Actions `Release` workflow. The workflow
is guarded so only the GitHub login listed in `RELEASE_ALLOWED_ACTOR` can run the release
path.

Before running the workflow:

1. Bump `pyproject.toml`.
2. Bump `integrations/claude-plugin/.claude-plugin/plugin.json` to the same version.
3. Merge the release PR to `main`.

Then open GitHub Actions, run `Release` from `main`, and enter the version without the
leading `v`, for example `0.12.0`.

The workflow verifies that the requested version matches `pyproject.toml`, verifies that
the Claude plugin manifest has the same version, runs the Python test suite with
`ORMAH_LLM_PROVIDER=none`, builds the UI, builds one wheel, publishes that wheel to PyPI,
creates `v<version>`, and creates the GitHub Release with generated notes.

PyPI publishing uses Trusted Publishing. No long-lived PyPI token is stored in the repo,
local `.env`, or developer machines. The PyPI project should trust this repository,
`.github/workflows/release.yml`, and the `pypi` GitHub environment.

Protect the `pypi` GitHub environment with the release manager as the required reviewer.
Do not enable prevent-self-review unless a second reviewer is intentionally required for
every release. If Trusted Publishing is not configured yet, use a project-scoped PyPI API
token only as a temporary environment-scoped GitHub Actions secret.


Use the GitHub Actions release workflow; do not publish releases through local scripts or the local Makefile release target.

### Coupled Python and desktop releases

Each Python release needs a new desktop build pinned to that exact Python version.
The pin comes from `pyproject.toml` via `desktop/src-tauri/build.rs`; desktop must
never install an unpinned latest package at runtime. Fresh terminal installs and
app builds share `desktop/uv-version` (the standalone installer embeds a tested
copy; `tests/test_install_uv.py` prevents drift). uv 0.13.6 was selected as the
[current stable Astral release](https://github.com/astral-sh/uv/releases/tag/0.13.6)
on October 31, 2026, with per-asset SHA-256 files.
The terminal installer reuses existing uv without upgrading it, prints its version,
and warns only below the supported minimum 0.10.9. Python remains `--python 3.11`.

```text
reviewed version PR → Python release → PyPI propagation confirmed
    → owner candidate command → signed/notarized DRAFT (macOS + Linux)
    → exact DMG clean-VM acceptance + manual checks for all platforms
    → owner promotion command → publish desktop → update desktop-latest
```

Main's branch protection was inspected read-only: a PR, one approval and code-owner
review are required. Do not push version bumps directly to main or bypass those
requirements. No bot push or runner registration is needed for this flow.

1. In each Python release preparation PR, also increment the independent desktop
   patch version in `desktop/src-tauri/tauri.conf.json`, `Cargo.toml`, and the
   `ormah-desktop` entry in `Cargo.lock`. Keep those three equal. Keep the Python
   and Claude plugin versions equal as above. The Python release workflow checks
   that the desktop version is synced and unused before publishing. Recent `chore(release)` commits use
   this convention. No version bump is included in the release-train implementation.
2. Merge through normal review. Run the Python `Release` workflow on main. After
   publishing, it polls PyPI latest plus the non-yanked wheel for up to five minutes,
   logging every observation and bypassing caches (#180). The desktop workflow
   repeats that guard. Network failures and stale responses retry, then fail closed.
3. From a clean checkout of that published `vX.Y.Z` tag (with tags fetched), run:

   ```bash
   python3 scripts/release/desktop.py candidate --ormah X.Y.Z
   ```

   This is the required manual handoff recorded in the Python workflow summary.
   It dispatches the desktop workflow on that exact tag and supplies its source SHA;
   it does not publish. A moved dispatch ref fails closed, and candidate source must
   already be merged into protected main. The
   build fails if the desktop version already has a release, if version files
   disagree, or if the source does not contain the published Python release commit.
   Checkout uses full history for the ancestry check. Tag pushes of `desktop-v*`
   still build candidates, verify the desktop tag, and follow the same draft gate.
4. The macOS build retains Developer ID signing, app notarization, and separate DMG
   notarization/stapling. Linux retains AppImage and Debian updater signatures.
   Both matrix jobs must succeed before one draft is created. `candidate.json`
   records the source commit, all version pins, and SHA-256 of **every** uploaded
   package, macOS updater bundle and signature. Never replace a candidate's assets.
   Build jobs cannot publish or change `desktop-latest`.
5. Download the draft assets using owner GitHub CLI subscription login:

   ```bash
   gh release download desktop-vA.B.C -R r-spade/ormah --dir candidate
   bash scripts/acceptance/macos-vm.sh app --dmg candidate/Ormah_A.B.C_aarch64.dmg \
     --expected-version X.Y.Z --desktop-version A.B.C --output evidence/app
   bash scripts/acceptance/macos-vm.sh terminal --expected-version X.Y.Z \
     --output evidence/terminal
   ```

   Use the actual DMG filename from the draft. Run from the candidate checkout so
   the uv pin also matches. Private draft URLs may require `gh release download`;
   the acceptance script also accepts public HTTPS URLs and local installer paths.
   The terminal test defaults to the public installer, and covers `--no-setup` then
   `ormah setup --ci`; it does not cover interactive setup. See
   [Tart preparation and acceptance](desktop/README.md#clean-machine-acceptance).
6. Finish the manual release items below and create `manual.json`. Copy `assets`
   verbatim from the downloaded `candidate.json`, including signatures; set a check
   to true **only after actually performing it against these assets**:

   ```json
   {
     "schema": 1,
     "tag": "desktop-vA.B.C",
     "assets": {"COPY EVERY FILENAME": "AND ITS SHA-256 FROM candidate.json"},
     "checks": {
       "gatekeeper_browser_first_open": true,
       "signed_in_ai_tool": true,
       "small_display": true,
       "network_fault": true,
       "linux_appimage": true,
       "linux_deb": true
     }
   }
   ```

   Linux acceptance is manual: on fresh supported Linux machines test each exact
   AppImage and deb, runtime version, Python 3.11, models, remember/recall, restart
   and login autostart, and signed updater artifact availability. Record distro,
   timings and observations alongside `manual.json`. Both packages are gated
   equally; a passing Mac result alone cannot publish either Linux artifact.
7. Only the authenticated owner `r-spade` can run the explicit promotion command:

   ```bash
   python3 scripts/release/desktop.py promote --tag desktop-vA.B.C \
     --acceptance evidence/app/result.json --manual manual.json
   ```

   Promotion re-downloads the candidate, checks every manifest hash, requires every
   app check and manual check, and verifies matching backend/desktop/uv versions.
   It refuses out-of-order promotion behind a newer published desktop. It records
   the authenticated actor, time, candidate and full evidence in the release's
   `promotion-record.json`, publishes the release, then regenerates `latest.json`
   for macOS, Linux AppImage and Linux deb on the rolling `desktop-latest`
   prerelease. There is no override. Owner-supplied JSON is an audited attestation,
   not a cryptographic proof of a Mac run. Keep logs and review them before promotion.
   Do not publish a draft via the GitHub web UI; that would bypass this process.

### Acceptance failure and recovery

The feed stays unchanged while a candidate is a draft or acceptance fails. Keep
the failed clone and evidence; inspect the reported VM with Tart and SSH. Delete
only that clone once its evidence is saved. Never run app acceptance using an
isolated HOME on a developer account (server stop matches UID), or another macOS
user (localhost is shared). All service operations in the script run inside the VM.

For a transient test/environment failure, rerun against the same downloaded DMG
in a fresh clone. For a code, pin or packaging correction, use a new reviewed PR
with a new desktop version. Keep the backend pinned to the released Python version
unless a Python correction also needs publishing. From that reviewed main commit,
run `candidate --ormah X.Y.Z --ref main`; it must contain the published Python tag
and still have `pyproject.toml` version X.Y.Z. Leave the rejected draft as evidence;
never overwrite its bytes or reuse its acceptance result.

If the Python workflow times out waiting for propagation, the PyPI wheel and
Python GitHub tag/release already exist. Once PyPI is visible, run the candidate
command; it repeats the same propagation guard. Do not republish the Python wheel.

A failed build before draft creation may be rerun safely. If promotion publishes
successfully but uploading the feed fails, the old feed remains active; rerun the
same promotion command with the same evidence. The command supports that recovery
and records the attempt again. It cannot promote an older version over a newer
published release. GitHub release publication and feed upload are two operations,
so publication can precede feed availability briefly; inspect the logs on failure.

### Installer delivery

`ormah.me/install.sh` redirects to `www.ormah.me/install.sh`, whose Vercel redirect
in [ormah-web/next.config.ts](https://github.com/r-spade/ormah-web/blob/main/next.config.ts)
points at `raw.githubusercontent.com/r-spade/ormah/main/install.sh`. This was checked
against the live redirect chain and website source. Merging the installer change
into main makes it available after upstream caches expire (raw GitHub currently
advertises a five-minute cache). There is no separate copy to deploy. Verify the
redirect and served script after merge; no website deployment is part of this PR.
