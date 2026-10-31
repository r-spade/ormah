#!/usr/bin/env bash
# Downloads the uv binary for the current (or specified) Rust target triple
# and places it in desktop/src-tauri/binaries/ where Tauri's externalBin picks it up.
#
# Usage:
#   ./desktop/scripts/build-sidecar.sh                          # auto-detects target
#   ./desktop/scripts/build-sidecar.sh aarch64-apple-darwin     # explicit target
#
# Called automatically by CI before `cargo tauri build`.
# Run manually once before `cargo check` / `cargo tauri dev` on a dev machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARIES_DIR="$SCRIPT_DIR/../src-tauri/binaries"
mkdir -p "$BINARIES_DIR"

# Determine target triple
if [[ $# -ge 1 ]]; then
    TARGET="$1"
else
    TARGET="$(rustc -vV 2>/dev/null | awk '/host:/ {print $2}')"
fi

if [[ -z "$TARGET" ]]; then
    echo "error: could not determine Rust target triple (is rustc installed?)" >&2
    exit 1
fi

# Map Rust triple → uv release asset name
case "$TARGET" in
    aarch64-apple-darwin)   UV_ASSET="uv-aarch64-apple-darwin.tar.gz" ;;
    x86_64-apple-darwin)    UV_ASSET="uv-x86_64-apple-darwin.tar.gz" ;;
    x86_64-unknown-linux-gnu) UV_ASSET="uv-x86_64-unknown-linux-gnu.tar.gz" ;;
    aarch64-unknown-linux-gnu) UV_ASSET="uv-aarch64-unknown-linux-gnu.tar.gz" ;;
    *)
        echo "error: unsupported target '$TARGET'" >&2
        exit 1
        ;;
esac

DEST="$BINARIES_DIR/uv-$TARGET"

UV_VERSION="$(cat "$SCRIPT_DIR/../uv-version")"
[[ "$UV_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "invalid uv pin" >&2; exit 1; }

# Always verify the release archive, including on repeat/cross-target builds.
# Executing a cached target binary is not possible when cross-compiling.
URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ASSET"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading uv $UV_VERSION ($UV_ASSET)..."
curl -fsSL "$URL" -o "$TMP/$UV_ASSET"
curl -fsSL "$URL.sha256" -o "$TMP/$UV_ASSET.sha256"
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$TMP" && sha256sum -c "$UV_ASSET.sha256")
else
    (cd "$TMP" && shasum -a 256 -c "$UV_ASSET.sha256")
fi
tar --no-same-owner -xzf "$TMP/$UV_ASSET" -C "$TMP"
cp "$TMP/uv-$TARGET/uv" "$DEST"
chmod +x "$DEST"
echo "Installed uv $UV_VERSION sidecar → $DEST"
