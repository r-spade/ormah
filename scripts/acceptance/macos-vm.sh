#!/usr/bin/env bash
# Clean-machine acceptance. Host operations are limited to a fresh Tart clone.
# Requires Tart, OpenSSH, Python 3.11+ and a prepared, stopped ormah-base VM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
[[ $# -gt 0 ]] && shift
BASE=ormah-base
ARTIFACT=""
INSTALLER=https://ormah.me/install.sh
EXPECTED="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "$ROOT/pyproject.toml" | head -1)"
DESKTOP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$ROOT/desktop/src-tauri/tauri.conf.json")"
UV_VERSION="$(cat "$ROOT/desktop/uv-version")"
KEEP=false
OUTPUT="$PWD/acceptance-$(date -u +%Y%m%dT%H%M%SZ)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dmg) ARTIFACT="$2"; shift 2 ;;
        --installer) INSTALLER="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --expected-version) EXPECTED="$2"; shift 2 ;;
        --desktop-version) DESKTOP="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --keep) KEEP=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$MODE" != app && "$MODE" != terminal ]] || [[ "$MODE" == app && -z "$ARTIFACT" ]]; then
    echo "Usage: $0 app --dmg <path|url> | terminal [--installer <path|url>]" >&2
    exit 2
fi
for version in "$EXPECTED" "$DESKTOP" "$UV_VERSION"; do
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Invalid version" >&2; exit 2; }
done
for tool in tart ssh python3 shasum curl; do
    command -v "$tool" >/dev/null || { echo "Missing host tool: $tool" >&2; exit 2; }
done
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
# Refuse overwriting evidence from a previous run.
[[ ! -e "$OUTPUT/result.json" && ! -e "$OUTPUT/checks.tsv" ]] || exit 2
WORK="$(mktemp -d)"
VM="ormah-acceptance-$(date +%s)-${WORK##*/}"
CREATED=false
BOOT_PID=""
ARTIFACT_SHA=""
STARTED=$(date +%s)
touch "$OUTPUT/checks.tsv"
finish() {
    local status=$?
    trap - EXIT
    if $CREATED && [[ $status -eq 0 ]] && ! $KEEP; then
        tart stop "$VM" >> "$OUTPUT/cleanup.log" 2>&1 || true
        if ! tart delete "$VM" >> "$OUTPUT/cleanup.log" 2>&1; then status=1; fi
    fi
    if [[ $status -ne 0 ]] && $CREATED; then
        echo "Failed; VM retained for evidence: $VM" >&2
    fi
    python3 - "$OUTPUT" "$MODE" "$VM" "$ARTIFACT_SHA" "$EXPECTED" "$DESKTOP" \
        "$UV_VERSION" "$STARTED" "$status" <<'PY'
import json
import pathlib
import sys
import time
out, mode, vm, sha, version, desktop, uv, started, status = sys.argv[1:]
root = pathlib.Path(out)
checks = {}
for row in (root / "checks.tsv").read_text().splitlines():
    name, seconds, passed = row.split("\t")
    checks[name] = {"passed": passed == "0", "seconds": int(seconds),
                    "output": (root / (name + ".log")).read_text(errors="replace")}
versions = {name: checks[name]["output"].strip() for name in
            ("ormah_version", "python_version", "uv_version", "desktop_version") if name in checks}
result = {"schema": 1, "status": "passed" if status == "0" else "failed", "mode": mode,
          "vm": vm, "artifact_sha256": sha, "expected_version": version,
          "expected_desktop_version": desktop, "expected_uv_version": uv, "versions": versions,
          "started_at": int(started), "seconds": int(time.time()) - int(started), "checks": checks,
          "manual_required": ["Safari download/first-open Gatekeeper (base assessments disabled)",
                              "AI tool connection with subscription sign-in", "1024x768 display",
                              "HTTPS_PROXY dead-port failure UI", "interactive terminal setup"]}
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(root / "result.json")
PY
    rm -rf "$WORK"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM

# Every external operation has a deadline. Logs and check status survive failure.
check() {
    local name="$1" limit="$2" begin pid watcher rc
    shift 2
    begin=$(date +%s)
    cat > "$WORK/input"
    "$@" < "$WORK/input" > "$OUTPUT/$name.log" 2>&1 &
    pid=$!
    (
        sleep "$limit" &
        sleeper=$!
        trap 'kill "$sleeper" 2>/dev/null || true; exit 0' TERM
        wait "$sleeper"
        kill -TERM "$pid" 2>/dev/null || exit 0
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
    ) &
    watcher=$!
    rc=0
    wait "$pid" || rc=$?
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    printf '%s\t%s\t%s\n' "$name" "$(($(date +%s) - begin))" "$rc" >> "$OUTPUT/checks.tsv"
    cat "$OUTPUT/$name.log"
    [[ $rc -eq 0 ]]
}
fetch() {
    case "$1" in
        https://*) curl --fail --location --proto '=https' --max-time 300 "$1" -o "$2" ;;
        *://*) echo "Only HTTPS URLs or local paths are supported" >&2; return 1 ;;
        *) cp "$1" "$2" ;;
    esac
}
if [[ "$MODE" == app ]]; then
    check fetch 310 fetch "$ARTIFACT" "$OUTPUT/artifact.dmg" </dev/null
    INPUT="$OUTPUT/artifact.dmg"
else
    check fetch 310 fetch "$INSTALLER" "$OUTPUT/install.sh" </dev/null
    INPUT="$OUTPUT/install.sh"
fi
ARTIFACT_SHA="$(shasum -a 256 "$INPUT" | awk '{print $1}')"
# Do not stop or alter BASE. `tart clone` fails if this unique name already exists.
check clone 120 tart clone "$BASE" "$VM" </dev/null
CREATED=true
check display 30 tart set "$VM" --display 1920x1200 </dev/null
tart run "$VM" > "$OUTPUT/tart.log" 2>&1 &
BOOT_PID=$!

# Password exists only in a private temp file; never in argv or user's SSH config.
printf '%s\n' 'admin' > "$WORK/password"
chmod 600 "$WORK/password"
cat > "$WORK/askpass" <<'ASKPASS'
#!/bin/sh
cat "$ORMAH_ACCEPTANCE_PASSWORD_FILE"
ASKPASS
chmod 700 "$WORK/askpass"
export SSH_ASKPASS="$WORK/askpass" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
export ORMAH_ACCEPTANCE_PASSWORD_FILE="$WORK/password"
SSH_OPTIONS=(-F /dev/null -o "UserKnownHostsFile=$WORK/known_hosts" -o StrictHostKeyChecking=accept-new
    -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=5 -o ConnectionAttempts=1
    -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o LogLevel=ERROR
    -o PubkeyAuthentication=no -o PreferredAuthentications=password -o NumberOfPasswordPrompts=1)
IP=""
wait_ssh() {
    local deadline=$((SECONDS + 180))
    while [[ $SECONDS -lt $deadline ]]; do
        IP="$(tart ip "$VM" 2>/dev/null || true)"
        # Never allow a malformed address to select localhost or inject SSH options.
        if [[ "$IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ && "$IP" != 127.* && "$IP" != 0.* ]]; then
            if ssh "${SSH_OPTIONS[@]}" "admin@$IP" true </dev/null; then return 0; fi
        fi
        sleep 3
    done
    return 1
}
# Run in foreground so the IP discovered in wait_ssh remains in this shell.
# The function is itself bounded, with per-SSH connection deadlines.
if wait_ssh > "$OUTPUT/ssh_ready.log" 2>&1; then rc=0; else rc=1; fi
printf 'ssh_ready\t%s\t%s\n' "$(( $(date +%s) - STARTED ))" "$rc" >> "$OUTPUT/checks.tsv"
[[ $rc -eq 0 ]]
guest() { exec ssh "${SSH_OPTIONS[@]}" "admin@$IP" 'bash -se'; }
check clean_baseline 60 guest <<'GUEST'
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$HOME/.local/bin:$HOME/.cargo/bin
id
xcode-select -p 2>&1 || true
python_path=$(command -v python3 || true)
echo "python3 path: $python_path"
[[ -z "$python_path" || "$python_path" == /usr/bin/python3 ]]
# /usr/bin/python3 is an Xcode installer stub when tools are absent; do not invoke it.
if xcode-select -p >/dev/null 2>&1; then python3 --version; else echo 'python3: Xcode stub, no real Python'; fi
spctl --status 2>&1 || true
for tool in uv brew ormah; do
    if command -v "$tool"; then echo "Base is not clean: $tool exists"; exit 1; fi
done
[[ ! -d /Library/Developer/CommandLineTools ]]
[[ ! -d /Applications/Ormah.app ]]
[[ ! -e "$HOME/.local/share/ormah" ]]
[[ ! -e "$HOME/.local/share/uv" ]]
# Listener checks happen only inside the VM.
! lsof -nP -iTCP:8787 -sTCP:LISTEN
GUEST
check gui_session 30 guest <<'GUEST'
[[ "$(stat -f %Su /dev/console)" == "$(id -un)" ]]
launchctl print "gui/$(id -u)" >/dev/null
GUEST
check transfer 300 ssh "${SSH_OPTIONS[@]}" "admin@$IP" 'cat > /tmp/ormah-acceptance-input' < "$INPUT"
check artifact_checksum 30 guest <<GUEST
[[ "\$(shasum -a 256 /tmp/ormah-acceptance-input | awk '{print \$1}')" == "$ARTIFACT_SHA" ]]
GUEST
if [[ "$MODE" == app ]]; then
    check install 180 guest <<'GUEST'
xattr -w com.apple.quarantine '0083;00000000;OrmahAcceptance;' /tmp/ormah-acceptance-input
mkdir "$HOME/ormah-acceptance-mount"
hdiutil attach -nobrowse -mountpoint "$HOME/ormah-acceptance-mount" /tmp/ormah-acceptance-input
trap 'hdiutil detach "$HOME/ormah-acceptance-mount"' EXIT
ditto "$HOME/ormah-acceptance-mount/Ormah.app" /Applications/Ormah.app
xattr -w com.apple.quarantine '0083;00000000;OrmahAcceptance;' /Applications/Ormah.app
xattr -p com.apple.quarantine /Applications/Ormah.app
open -a /Applications/Ormah.app
GUEST
    check desktop_version 30 guest <<GUEST
actual=\$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' /Applications/Ormah.app/Contents/Info.plist)
echo "\$actual"
[[ "\$actual" == "$DESKTOP" ]]
GUEST
else
    check install 1200 guest <<'GUEST'
/bin/bash /tmp/ormah-acceptance-input --no-setup
export PATH="$HOME/.local/bin:$PATH"
ormah setup --ci
GUEST
fi
ready_script=$(cat <<'GUEST'
deadline=$((SECONDS + 900))
while [[ $SECONDS -lt $deadline ]]; do
    if curl -fsS --max-time 5 http://127.0.0.1:8787/admin/health | tr -d '[:space:]' | grep -Fq '"status":"ok"'; then exit 0; fi
    sleep 5
done
exit 1
GUEST
)
check server_ready 920 guest <<< "$ready_script"
owner_script=$(cat <<'GUEST'
pids=$(lsof -t -nP -iTCP:8787 -sTCP:LISTEN)
[[ -n "$pids" ]]
for pid in $pids; do
    [[ "$(ps -o uid= -p "$pid" | tr -d " ")" == "$(id -u)" ]]
    ps -p "$pid" -o command= | grep -E "ormah.*server|python.*ormah"
done
GUEST
)
check server_owner 30 guest <<< "$owner_script"
check ormah_version 30 guest <<GUEST
actual=\$("\$HOME/.local/bin/ormah" --version)
echo "\$actual"
[[ "\$actual" == "ormah $EXPECTED" ]]
# Also inspect the interpreter/package used by the listening server, not just a CLI on PATH.
head -1 "\$HOME/.local/bin/ormah"
GUEST
# Use the installed CLI's shebang, never the macOS /usr/bin/python3 stub.
check python_version 30 guest <<'GUEST'
interpreter=$(head -1 "$HOME/.local/bin/ormah")
interpreter=${interpreter#\#!}
"$interpreter" -c 'import sys, importlib.metadata; print(sys.version); print(importlib.metadata.version("ormah")); assert sys.version_info[:2] == (3, 11)'
GUEST
check uv_version 30 guest <<GUEST
if [[ "$MODE" == app ]]; then
    uv=\$(find /Applications/Ormah.app/Contents -type f -name 'uv*' | head -1)
else
    uv="\$HOME/.local/bin/uv"
fi
[[ -n "\$uv" ]]
actual=\$("\$uv" --version)
echo "\$actual"
[[ "\$actual" == 'uv $UV_VERSION'* ]]
GUEST
check models 60 guest <<'GUEST'
interpreter=$(head -1 "$HOME/.local/bin/ormah"); interpreter=${interpreter#\#!}
"$interpreter" - <<'PY'
from pathlib import Path
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from ormah.config import settings
from ormah.embeddings.cache import get_fastembed_cache_dir, get_model_cache_dirname
for model, registry in ((settings.embedding_model, TextEmbedding.list_supported_models()),
                        (settings.whisper_reranker_model, TextCrossEncoder.list_supported_models())):
    name = get_model_cache_dirname(model, registry)
    assert name, model
    path = get_fastembed_cache_dir() / name
    # Directory existence alone can be a failed/incomplete download.
    weights = [p for p in path.rglob('*.onnx') if p.stat().st_size > 0]
    assert weights, (model, str(path))
    print(model, [(str(p), p.stat().st_size) for p in weights])
PY
GUEST
TOKEN="ormah-acceptance-${VM}"
check remember 90 guest <<GUEST
"\$HOME/.local/bin/ormah" remember '$TOKEN' --type fact --tags acceptance
GUEST
recall_script="\"\$HOME/.local/bin/ormah\" recall '$TOKEN' --json | grep -F '$TOKEN'"
check recall 90 guest <<< "$recall_script"
if [[ "$MODE" == app ]]; then
    check app_process 30 guest <<'GUEST'
pgrep -u "$(id -u)" -f '^/Applications/Ormah.app/Contents/MacOS/'
GUEST
else
    check login_shell 30 guest <<'GUEST'
# Do not inject PATH here: this must work as a real new login shell.
"$SHELL" -lic 'command -v ormah && ormah --version' </dev/null
GUEST
fi
check boot_before 30 ssh "${SSH_OPTIONS[@]}" "admin@$IP" 'sysctl -n kern.boottime' </dev/null
BOOT_BEFORE="$(cat "$OUTPUT/boot_before.log")"
[[ -n "$BOOT_BEFORE" ]]
check reboot 30 guest <<'GUEST'
sudo -n true
# Delay until SSH has acknowledged the request, then reboot the guest only.
nohup sh -c 'sleep 2; sudo -n /sbin/reboot' >/tmp/ormah-reboot.log 2>&1 </dev/null &
GUEST
# Tart versions may keep running across a reboot or exit at guest shutdown.
# Handle both and prove an actual reboot by comparing the guest kernel boot time.
wait_reboot() {
    local deadline=$((SECONDS + 180)) after
    while [[ $SECONDS -lt $deadline ]]; do
        if ! kill -0 "$BOOT_PID" 2>/dev/null; then
            tart run "$VM" >> "$OUTPUT/tart.log" 2>&1 &
            BOOT_PID=$!
        fi
        after="$(ssh "${SSH_OPTIONS[@]}" "admin@$IP" 'sysctl -n kern.boottime' 2>/dev/null || true)"
        if [[ -n "$after" && "$after" != "$BOOT_BEFORE" ]]; then
            echo "$after"
            return 0
        fi
        sleep 3
    done
    return 1
}
check reboot_completed 190 wait_reboot </dev/null
if wait_ssh > "$OUTPUT/ssh_after_reboot.log" 2>&1; then rc=0; else rc=1; fi
printf 'ssh_after_reboot\t0\t%s\n' "$rc" >> "$OUTPUT/checks.tsv"
[[ $rc -eq 0 ]]
check server_returns 920 guest <<< "$ready_script"
check server_owner_after_reboot 30 guest <<< "$owner_script"
check recall_after_reboot 90 guest <<< "$recall_script"
if [[ "$MODE" == app ]]; then
    check app_process_after_reboot 70 guest <<'GUEST'
deadline=$((SECONDS + 60))
until pgrep -u "$(id -u)" -f '^/Applications/Ormah.app/Contents/MacOS/'; do
    [[ $SECONDS -lt $deadline ]] || exit 1
    sleep 2
done
GUEST
else
    check login_shell_after_reboot 30 guest <<'GUEST'
"$SHELL" -lic 'command -v ormah && ormah --version' </dev/null
GUEST
fi
echo 'Manual release items remain: browser Gatekeeper first-open, signed-in AI tool, small display, network fault.'
echo 'Terminal mode covers --no-setup plus setup --ci; interactive setup is not covered.'
