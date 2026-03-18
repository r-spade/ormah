#!/usr/bin/env bash
# Ormah — one-liner installer
# Usage: bash <(curl -fsSL https://www.ormah.me/install.sh)
#        bash <(curl -fsSL https://www.ormah.me/install.sh) --no-setup
set -euo pipefail

# ── Output helpers ──────────────────────────────────────────────────────────

_use_color=true
if [[ -n "${NO_COLOR:-}" ]] || [[ ! -t 1 && -z "${FORCE_COLOR:-}" ]]; then
    _use_color=false
fi

_c_reset="" _c_green="" _c_yellow="" _c_bold=""
if $_use_color; then
    _c_reset=$'\033[0m'
    _c_green=$'\033[32m'
    _c_yellow=$'\033[33m'
    _c_bold=$'\033[1m'
fi

info()  { printf '%s[..]%s %s\n' "$_c_bold"   "$_c_reset" "$*"; }
ok()    { printf '%s[ok]%s %s\n' "$_c_green"   "$_c_reset" "$*"; }
warn()  { printf '%s[!!]%s %s\n' "$_c_yellow"  "$_c_reset" "$*"; }
fail()  { printf '%s[xx]%s %s\n' "$_c_bold"    "$_c_reset" "$*" >&2; exit 1; }
step()  { printf '\n%s==>%s %s\n' "$_c_bold"   "$_c_reset" "$*"; }

# ── Argument parsing ────────────────────────────────────────────────────────

RUN_SETUP=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-setup) RUN_SETUP=false; shift ;;
        --help|-h)
            cat <<'USAGE'
Ormah installer — bootstraps uv, installs ormah, runs interactive setup.

Usage:
  bash <(curl -fsSL https://ormah.me/install.sh)
  bash <(curl -fsSL https://ormah.me/install.sh) --no-setup

Options:
  --no-setup   Install only, skip interactive setup (for CI/headless)
  --help       Show this message

Environment variables:
  ORMAH_INSTALL_SOURCE   Override PyPI package spec (e.g. a git URL)
  NO_COLOR               Disable colored output
USAGE
            exit 0
            ;;
        *) fail "Unknown option: $1 (try --help)" ;;
    esac
done

# ── OS detection ────────────────────────────────────────────────────────────

step "Detecting platform"
PLATFORM="$(uname -s)"
case "$PLATFORM" in
    Darwin) ok "macOS detected" ;;
    Linux)  ok "Linux detected" ;;
    *)      fail "Unsupported platform: $PLATFORM (macOS and Linux only)" ;;
esac

# ── Install uv ──────────────────────────────────────────────────────────────

step "Checking for uv"
if command -v uv >/dev/null 2>&1; then
    ok "uv already installed ($(uv --version))"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Get uv on PATH for this session
    if [[ -f "$HOME/.local/bin/env" ]]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    elif [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        . "$HOME/.cargo/env"
    fi

    # Fallback: add ~/.local/bin directly
    if ! command -v uv >/dev/null 2>&1; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    command -v uv >/dev/null 2>&1 || fail "uv installation failed — could not find uv on PATH"
    ok "uv installed ($(uv --version))"
fi

# ── Install ormah ───────────────────────────────────────────────────────────

step "Installing ormah"

INSTALL_SOURCE="${ORMAH_INSTALL_SOURCE:-ormah[litellm]}"
UV_FLAGS=(--python 3.11)

if uv tool list 2>/dev/null | grep -q '^ormah '; then
    info "Existing install found — upgrading"
    UV_FLAGS+=(--upgrade)
fi

_install_ok=false

info "Installing from: $INSTALL_SOURCE"
if uv tool install "$INSTALL_SOURCE" "${UV_FLAGS[@]}" 2>/dev/null; then
    _install_ok=true
else
    # Fallback to git if PyPI failed and user didn't override source
    if [[ -z "${ORMAH_INSTALL_SOURCE:-}" ]]; then
        warn "PyPI install failed — trying git source"
        GIT_SOURCE='ormah[litellm] @ git+https://github.com/r-spade/ormah.git'
        if uv tool install "$GIT_SOURCE" "${UV_FLAGS[@]}"; then
            _install_ok=true
        fi
    fi
fi

$_install_ok || fail "Failed to install ormah"

# Verify binary is available
if ! command -v ormah >/dev/null 2>&1; then
    # uv tool bin may not be on PATH yet
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v ormah >/dev/null 2>&1 || fail "ormah installed but not found on PATH"

ok "ormah installed ($(ormah --version 2>/dev/null || echo 'unknown version'))"

# ── Post-install note ───────────────────────────────────────────────────────

warn "First server start will download the embedding model (~500 MB one-time download)"

# ── Run setup ───────────────────────────────────────────────────────────────

if $RUN_SETUP; then
    step "Running ormah setup"
    info "This will configure your LLM, start the server, and set up integrations"
    echo ""
    ormah setup
else
    echo ""
    ok "Installation complete"
    info "Run ${_c_bold}ormah setup${_c_reset} when you're ready to configure"
fi
