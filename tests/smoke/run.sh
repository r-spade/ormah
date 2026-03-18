#!/usr/bin/env bash
set -euo pipefail

# Use the same output helpers as install.sh
ok()   { printf '[ok] %s\n' "$*"; }
fail() { printf '[xx] %s\n' "$*" >&2; exit 1; }
info() { printf '[..] %s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }

# Ensure ormah is on PATH
export PATH="$HOME/.local/bin:$PATH"

step "CLI basics"
ormah --version || fail "ormah --version failed"
ok "ormah $(ormah --version 2>&1)"
ormah --help > /dev/null || fail "ormah --help failed"
ok "CLI responds to --help"

step "Non-interactive setup"
ormah setup --ci || fail "ormah setup --ci failed"
ok "Setup completed"

step "Server health"
# Server should already be running from setup
MAX_WAIT=120
for i in $(seq 1 $MAX_WAIT); do
    if curl -sf http://localhost:8787/admin/health > /dev/null 2>&1; then
        ok "Server healthy after ${i}s"
        break
    fi
    if [ "$i" -eq "$MAX_WAIT" ]; then
        fail "Server not healthy after ${MAX_WAIT}s"
    fi
    sleep 1
done

step "Remember + Recall"
# Use curl with generous timeout — first embedding inference and sqlite-vec
# insert can be slow in Docker on ARM.
API="http://localhost:8787"

curl -sf -X POST "$API/agent/remember" \
    -H 'Content-Type: application/json' \
    -d '{"content":"The sky is blue","type":"fact","tier":"working"}' \
    --max-time 120 > /dev/null || fail "remember failed"
ok "Stored a memory"

sleep 2  # Give the index a moment

RESULT=$(curl -sf -X POST "$API/agent/recall" \
    -H 'Content-Type: application/json' \
    -d '{"query":"sky color"}' \
    --max-time 120 2>&1) || fail "recall failed"
echo "$RESULT" | grep -qi "blue" || fail "Recall didn't find 'blue' in: $RESULT"
ok "Recall found the memory"

step "Context retrieval"
curl -sf "$API/agent/context" --max-time 60 > /dev/null || fail "context failed"
ok "Context endpoint works"

step "Server stop"
ormah server stop 2>/dev/null || true
ok "Server stopped"

step "All smoke tests passed"
