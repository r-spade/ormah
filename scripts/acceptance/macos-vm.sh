#!/usr/bin/env bash
# Clean-machine install test in a throwaway Tart VM. Never touches the host's Ormah.
# Usage: macos-vm.sh app --dmg <path> | terminal [--installer <path>]   [--ormah X.Y.Z] [--base VM] [--keep]
set -euo pipefail

MODE="${1:-}"; shift || true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORMAH="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "$ROOT/pyproject.toml" | head -1)"
BASE=ormah-base DMG="" INSTALLER="" KEEP=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dmg) DMG="$2"; shift 2 ;;
        --installer) INSTALLER="$2"; shift 2 ;;
        --ormah) ORMAH="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --keep) KEEP=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$MODE" != terminal && ! ( "$MODE" == app && -f "$DMG" ) ]]; then
    sed -n 3p "$0" >&2; exit 2
fi

VM="ormah-acceptance-$MODE-$(date +%s)"
OUT="$PWD/$VM"; mkdir -p "$OUT"
WORK="$(mktemp -d)"
SHA=""
TOKEN="purple-giraffe-$VM"

finish() {
    local status=$?
    local result=failed
    [[ $status -eq 0 ]] && result=passed
    printf '{"mode": "%s", "status": "%s", "artifact_sha256": "%s", "ormah_version": "%s"}\n' \
        "$MODE" "$result" "$SHA" "$ORMAH" > "$OUT/result.json"
    rm -rf "$WORK"
    if [[ $status -eq 0 ]] && ! $KEEP; then
        tart stop "$VM" >/dev/null 2>&1 || true
        tart delete "$VM" || true
    elif [[ $status -ne 0 ]]; then
        echo "FAILED. VM kept for evidence: $VM" >&2
    fi
    echo "$result: $OUT/result.json"
}
trap finish EXIT

# The base image's public default login, passed through a file so it stays out of argv.
printf '#!/bin/sh\necho admin\n' > "$WORK/askpass"; chmod 700 "$WORK/askpass"
export SSH_ASKPASS="$WORK/askpass" SSH_ASKPASS_REQUIRE=force
SSH=(ssh -F /dev/null -o "UserKnownHostsFile=$WORK/known_hosts" -o StrictHostKeyChecking=accept-new
     -o ConnectTimeout=5 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

guest() { "${SSH[@]}" "admin@$(tart ip "$VM" --wait 30)" 'bash -se' 2>&1 | tee -a "$OUT/guest.log"; return "${PIPESTATUS[0]}"; }
wait_for_ssh() {
    local _
    for _ in $(seq 1 60); do
        "${SSH[@]}" "admin@$(tart ip "$VM" --wait 5 2>/dev/null)" true 2>/dev/null </dev/null && return 0
        sleep 5
    done
    echo "VM never accepted SSH" >&2; return 1
}

tart clone "$BASE" "$VM"
tart set "$VM" --display 1920x1200
tart run "$VM" > "$OUT/tart.log" 2>&1 &
wait_for_ssh

guest <<'GUEST'
echo "--- clean baseline"
for path in /Library/Developer/CommandLineTools /opt/homebrew /usr/local/bin/uv /usr/local/bin/python3 \
    "$HOME/.local/bin/uv" "$HOME/.local/share/uv" "$HOME/.local/share/ormah" /Applications/Ormah.app; do
    [[ ! -e "$path" ]] || { echo "Base VM is not clean: $path"; exit 1; }
done
if lsof -nP -iTCP:8787 -sTCP:LISTEN; then echo "Base VM is not clean: port 8787 in use"; exit 1; fi
spctl --status || true
GUEST

if [[ "$MODE" == app ]]; then
    SHA="$(shasum -a 256 "$DMG" | awk '{print $1}')"
    "${SSH[@]}" "admin@$(tart ip "$VM")" 'cat > /tmp/ormah.dmg' < "$DMG"
    guest <<'GUEST'
echo "--- install app"
hdiutil attach -quiet -nobrowse -mountpoint /tmp/ormah-dmg /tmp/ormah.dmg
ditto /tmp/ormah-dmg/Ormah.app /Applications/Ormah.app
hdiutil detach -quiet /tmp/ormah-dmg
open /Applications/Ormah.app
GUEST
else
    if [[ -n "$INSTALLER" ]]; then
        "${SSH[@]}" "admin@$(tart ip "$VM")" 'cat > /tmp/install.sh' < "$INSTALLER"
    else
        "${SSH[@]}" "admin@$(tart ip "$VM")" 'curl -fsSL https://ormah.me/install.sh -o /tmp/install.sh' </dev/null
    fi
    guest <<'GUEST'
echo "--- install from terminal"
bash /tmp/install.sh --no-setup
"$HOME/.local/bin/ormah" setup --ci
zsh -lic 'command -v ormah'
GUEST
fi

wait_healthy='
for _ in $(seq 1 180); do
    curl -fsS --max-time 3 http://127.0.0.1:8787/admin/health | grep -q "\"status\":\"ok\"" && exit 0
    sleep 5
done
echo "Server never became healthy"; exit 1'

guest <<< "$wait_healthy"
guest <<GUEST
echo "--- checks"
ormah="\$HOME/.local/bin/ormah"
[[ "\$("\$ormah" --version)" == "ormah $ORMAH" ]] || { "\$ormah" --version; exit 1; }
python="\$HOME/.local/share/uv/tools/ormah/bin/python"
"\$python" -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 11)'
"\$python" -c 'import importlib.util; assert importlib.util.find_spec("litellm"), "litellm missing"'
# Embedding and reranker: setup tolerates a failed reranker download, so check each.
for model in "\$HOME"/.local/share/ormah/models/models--*; do
    find "\$model/" -name '*.onnx' -size +0 | grep -q . || { echo "No weights in \$model"; exit 1; }
done
[[ \$(ls -d "\$HOME"/.local/share/ormah/models/models--* | wc -l) -ge 2 ]] || { echo "Models missing"; exit 1; }
"\$ormah" remember '$TOKEN' --title 'Acceptance test'
"\$ormah" recall '$TOKEN' | grep -F '$TOKEN'
GUEST

boot_time() { "${SSH[@]}" "admin@$(tart ip "$VM" --wait 5)" 'sysctl -n kern.boottime' </dev/null; }
BOOT_BEFORE="$(boot_time)"
echo "--- reboot" | tee -a "$OUT/guest.log"
# SSH drops when the VM goes down, so success is judged by the boot time changing.
"${SSH[@]}" "admin@$(tart ip "$VM")" 'echo admin | sudo -S shutdown -r now' </dev/null 2>&1 | tee -a "$OUT/guest.log" || true
sleep 30
wait_for_ssh
[[ "$(boot_time)" != "$BOOT_BEFORE" ]] || { echo "VM did not reboot" >&2; exit 1; }
guest <<< "$wait_healthy"
guest <<GUEST
echo "--- after reboot"
"\$HOME/.local/bin/ormah" recall '$TOKEN' | grep -F '$TOKEN'
GUEST
if [[ "$MODE" == app ]]; then
    guest <<'GUEST'
pgrep -f /Applications/Ormah.app/Contents/MacOS/ormah-desktop
GUEST
fi
