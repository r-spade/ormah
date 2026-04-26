#!/usr/bin/env bash
set -u

REPO="${ORMAH_TEST_REPO:-$HOME/ormah-server-side-test}"
PORT="${ORMAH_TEST_PORT:-8791}"
LOG="$HOME/ormah-stop-autostart-check-$(date +%Y%m%d-%H%M%S).log"
ENV_FILE="$HOME/.config/ormah/.env"
BACKUP=""

exec > >(tee "$LOG") 2>&1

checkpoint() {
    printf '\n===== %s =====\n' "$1"
}

run() {
    printf '+ %s\n' "$*"
    "$@"
    rc=$?
    printf '[exit=%s]\n' "$rc"
    return "$rc"
}

checkpoint "identity"
run whoami
run id
run date
printf 'LOG=%s\n' "$LOG"

checkpoint "repo update"
cd "$REPO" || exit 1
run git branch --show-current
run git pull --ff-only
run git log --oneline -5

checkpoint "rebuild venv"
rm -rf .venv
UV_BIN="$(command -v uv)"
PY311="$(uv python find 3.11)"
env PATH="$(dirname "$UV_BIN"):$(dirname "$PY311")" \
    "$UV_BIN" run --python "$PY311" --extra litellm ormah --version || exit 1
export PATH="$PWD/.venv/bin:$PATH"
run which ormah
run ormah --version

checkpoint "backup config and force test port"
mkdir -p "$(dirname "$ENV_FILE")"
if [ -f "$ENV_FILE" ]; then
    BACKUP="$ENV_FILE.backup.$(date +%Y%m%d-%H%M%S)"
    cp "$ENV_FILE" "$BACKUP"
    printf 'Backed up %s to %s\n' "$ENV_FILE" "$BACKUP"
fi

tmp="$(mktemp)"
if [ -f "$ENV_FILE" ]; then
    grep -v '^ORMAH_PORT=' "$ENV_FILE" > "$tmp" || true
fi
printf 'ORMAH_PORT=%s\n' "$PORT" >> "$tmp"
mv "$tmp" "$ENV_FILE"
run grep -n '^ORMAH_PORT=' "$ENV_FILE"

checkpoint "pre-existing servers"
run pgrep -af 'ormah server start'
run pgrep -u "$USER" -af 'ormah server start'
ORMAH_PORT="$PORT" run ormah server status

checkpoint "manual start"
ORMAH_PORT="$PORT" nohup ormah server start > "/tmp/ormah-test-$PORT-manual.log" 2>&1 &
MANUAL_PID=$!
printf 'manual_pid=%s\n' "$MANUAL_PID"
sleep 5
ORMAH_PORT="$PORT" run ormah server status
run pgrep -u "$USER" -af 'ormah server start'

checkpoint "manual stop"
ORMAH_PORT="$PORT" run ormah server stop
sleep 2
ORMAH_PORT="$PORT" run ormah server status
run pgrep -u "$USER" -af 'ormah server start'

checkpoint "daemon start -d"
ORMAH_PORT="$PORT" run ormah server start -d
sleep 5
ORMAH_PORT="$PORT" run ormah server status
run pgrep -u "$USER" -af 'ormah server start'
run systemctl --user is-active ormah.service
run systemctl --user is-enabled ormah.service

checkpoint "daemon stop"
ORMAH_PORT="$PORT" run ormah server stop
sleep 2
ORMAH_PORT="$PORT" run ormah server status
run pgrep -u "$USER" -af 'ormah server start'
run systemctl --user is-active ormah.service
run systemctl --user is-enabled ormah.service

checkpoint "restore config"
if [ -n "$BACKUP" ] && [ -f "$BACKUP" ]; then
    cp "$BACKUP" "$ENV_FILE"
    printf 'Restored %s from %s\n' "$ENV_FILE" "$BACKUP"
else
    printf 'No previous config backup to restore\n'
fi

checkpoint "done"
printf 'Copy this log back: %s\n' "$LOG"
