"""Claude CLI LLM adapter — headless `claude -p` via subscription auth (no paid API)."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from ormah.background.llm.base import LLMAdapter

logger = logging.getLogger(__name__)

# Trust boundary: the transcript is UNTRUSTED input (prompt-injection vector). The child must
# only ever emit text, never act. We pass ONE --settings override whose `permissions` block
# fully replaces whatever the operator's ~/.claude/settings.json has (verified 2026-07-02 on
# claude 2.1.156):
#   defaultMode "default" -> escape an inherited defaultMode:bypassPermissions. This is the
#     load-bearing key: neither `--allowed-tools ""`, nor the `--permission-mode default` CLI
#     flag, nor a `deny` list under an inherited bypass, overrides it. Setting defaultMode
#     here, on the same key, does. In headless -p with mode "default" and no allowed tools,
#     every tool the model attempts needs permission that no one can grant -> auto-denied.
#   allow []              -> drop the operator's inherited allow rules (e.g. Bash, Edit(./**)).
#   deny  [tool names]    -> belt-and-suspenders. NOTE: use bare tool names only. A glob rule
#     like "*"/"mcp__*" is rejected as invalid on claude 2.1.156, which discards the WHOLE
#     --settings block and silently falls back to the operator's bypassPermissions (verified
#     fail-open). Bare names are the safe form; MCP tools are already gated by defaultMode.
#   disableAllHooks true  -> the operator's own hooks AND plugin hooks otherwise FIRE in this
#     child (verified: a user SessionStart hook ran despite a hooks:{} override, because hooks
#     MERGE across sources rather than being replaced). disableAllHooks is a boolean, so the
#     --settings override actually takes effect, and it turns every non-managed hook off — no
#     recursion, no side effects. (We keep --no-session-persistence for the transcript; the
#     alternative, --setting-sources, disables hooks too but re-enables session persistence on
#     this CLI, so it is NOT used.)
_DENY_TOOLS = [
    "Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Bash", "Glob", "Grep",
    "LS", "WebFetch", "WebSearch", "Task",
]
_HARDENED_SETTINGS = json.dumps({
    "disableAllHooks": True,
    "permissions": {"defaultMode": "default", "allow": [], "deny": _DENY_TOOLS},
})

# Bound concurrent `claude -p`: one shared semaphore per distinct max_concurrency value. All
# adapters built with the same max share a bound (today ingest + maintenance read the same
# claude_cli_max_concurrency, so it is effectively global); adapters with different maxes get
# independent semaphores.
_SEMAPHORES: dict[int, threading.Semaphore] = {}
_SEM_LOCK = threading.Lock()


# session_id comes from the CLI envelope (untrusted). Only ever treat it as an exact,
# well-formed id — never as a glob pattern (a "*"/"?"/"[" could expand and delete unrelated
# transcripts). Real Claude session ids are UUIDs; this also admits hex/dash/underscore.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")


def _cleanup_persisted_stub(session_id: str) -> None:
    """Best-effort: delete the child's own transcript stub. Even with
    --no-session-persistence, `claude -p` writes a tiny ai-title record at
    ~/.claude/projects/<encoded-cwd>/<session_id>.jsonl. It carries ZERO conversation turns, so
    the session watcher skips it (not ingestible) — but removing it keeps ~/.claude clean and
    avoids leaving a prompt-derived title on disk. The id is validated and matched as an EXACT
    filename (no glob interpolation), so no other session's transcript is ever touched."""
    if not _SESSION_ID_RE.match(session_id or ""):
        return
    target = f"{session_id}.jsonl"
    try:
        for proj_dir in (Path.home() / ".claude" / "projects").iterdir():
            stub = proj_dir / target
            if stub.is_file():
                stub.unlink()
                return
    except OSError:
        pass


def _semaphore(max_concurrency: int) -> threading.Semaphore:
    with _SEM_LOCK:
        sem = _SEMAPHORES.get(max_concurrency)
        if sem is None:
            sem = threading.Semaphore(max_concurrency)
            _SEMAPHORES[max_concurrency] = sem
        return sem


class ClaudeCliAdapter(LLMAdapter):
    def __init__(
        self,
        model: str,
        timeout: int = 120,
        bin_path: str | None = None,
        max_concurrency: int = 1,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.bin_path = bin_path or shutil.which("claude") or "claude"
        self.max_concurrency = max(1, max_concurrency)

    def generate(
        self,
        prompt: str,
        json_mode: bool = True,
        *,
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_hint_seconds: float | None = None,
    ) -> str | None:
        # A batching caller can hint a longer budget for a fatter combined prompt; a plain
        # call keeps the constructor default.
        timeout = timeout_hint_seconds or self.timeout
        # Force subscription auth: strip the API key so the child never bills the paid API.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        # Prompt on stdin (never argv) — avoids leaking transcript text to the process list and
        # ARG_MAX failures on large transcripts.
        schema = None
        if response_format and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
        argv = [
            self.bin_path, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--no-session-persistence",
            "--permission-mode", "default",
            "--settings", _HARDENED_SETTINGS,
        ]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema)]
        sem = _semaphore(self.max_concurrency)
        with sem:
            try:
                proc = subprocess.run(
                    argv, input=prompt, capture_output=True, text=True,
                    timeout=timeout, cwd=tempfile.gettempdir(), env=env,
                )
            except subprocess.TimeoutExpired:
                logger.warning("claude -p timed out after %ss", timeout)
                return None
            except Exception as e:  # binary missing, OSError, etc.
                logger.warning("claude -p failed to run: %s", e)
                return None
        if proc.returncode != 0:
            logger.warning("claude -p exited %s: %s", proc.returncode, proc.stderr[:300])
            return None
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            logger.warning("claude -p returned a non-JSON envelope")
            return None
        if not isinstance(envelope, dict):
            return None
        _cleanup_persisted_stub(str(envelope.get("session_id") or envelope.get("sessionId") or ""))
        if envelope.get("is_error"):
            logger.warning("claude -p returned is_error envelope: %s", str(envelope.get("subtype"))[:100])
            return None
        if schema is not None:
            structured = envelope.get("structured_output")
            if structured is not None:
                return json.dumps(structured)
            # Fallback: for some prompt+schema pairs the CLI answers in a single text turn,
            # emitting valid schema-conformant JSON in `result` (```json-fenced) with
            # structured_output=null. Callers run extract_json + json.loads, so hand them the
            # raw result to recover; enum-integrity is normalized per-site by callers.
            result = envelope.get("result")
            return result if isinstance(result, str) and result.strip() else None
        result = envelope.get("result")
        return result if isinstance(result, str) else None
