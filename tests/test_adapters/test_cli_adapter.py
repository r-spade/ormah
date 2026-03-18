"""Tests for the CLI adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from ormah.adapters.cli_adapter import main


def _run_cli(args: list[str], monkeypatch, stdin_text: str | None = None):
    """Run the CLI with given args, returning (exit_code, stdout, stderr)."""
    import io
    import sys

    monkeypatch.setattr("sys.argv", ["ormah-cli"] + args)
    if stdin_text is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)

    exit_code = 0
    try:
        main()
    except SystemExit as e:
        exit_code = e.code if e.code is not None else 0

    return exit_code, stdout.getvalue(), stderr.getvalue()


def _mock_response(data, status_code=200):
    """Create a mock httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "http://test"),
    )


# --- status ---


def test_status_ok(monkeypatch):
    def handler(request):
        return _mock_response({"status": "ok"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(["status"], monkeypatch)
    assert code == 0
    assert "ok" in out


def test_status_with_jobs(monkeypatch):
    def handler(request):
        return _mock_response({
            "status": "ok",
            "jobs": {"auto_linker": {"state": "idle", "last_run": "2026-01-01T00:00:00"}}
        })

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(["status"], monkeypatch)
    assert code == 0
    assert "auto_linker" in out
    assert "idle" in out


# --- context ---


def test_context_default(monkeypatch):
    def handler(request):
        assert request.url.path == "/agent/context"
        return _mock_response({"text": "# Context\nHello world"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["context"], monkeypatch)
    assert code == 0
    assert "Hello world" in out


def test_context_with_task_and_space(monkeypatch):
    def handler(request):
        params = dict(request.url.params)
        assert params["space"] == "myproj"
        assert params["task_hint"] == "working on tests"
        return _mock_response({"text": "filtered context"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(
        ["context", "--space", "myproj", "--task", "working on tests"],
        monkeypatch,
    )
    assert code == 0
    assert "filtered context" in out


# --- recall ---


def test_recall_text(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["query"] == "graph clustering"
        return _mock_response({"text": "Found 2 memories"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["recall", "graph clustering"], monkeypatch)
    assert code == 0
    assert "Found 2 memories" in out


def test_recall_json(monkeypatch):
    response_data = {"text": "Found 1", "results": [{"id": "abc"}]}

    def handler(request):
        return _mock_response(response_data)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["recall", "test", "--json"], monkeypatch)
    assert code == 0
    parsed = json.loads(out)
    assert parsed["results"][0]["id"] == "abc"


def test_recall_with_limit_and_types(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["limit"] == 5
        assert body["types"] == ["fact", "decision"]
        return _mock_response({"text": "ok"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(
        ["recall", "query", "--limit", "5", "--types", "fact,decision"],
        monkeypatch,
    )
    assert code == 0


def test_recall_passes_space(monkeypatch):
    def handler(request):
        params = dict(request.url.params)
        assert params["default_space"] == "myproj"
        return _mock_response({"text": "ok"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(
        ["recall", "test", "--space", "myproj"],
        monkeypatch,
    )
    assert code == 0


# --- remember ---


def test_remember_basic(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["content"] == "test memory"
        assert body["type"] == "fact"
        assert body["tier"] == "working"
        return _mock_response({"text": "Stored.", "node_id": "abc123"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["remember", "test memory"], monkeypatch)
    assert code == 0
    assert "Stored" in out


def test_remember_all_flags(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["title"] == "My Title"
        assert body["type"] == "decision"
        assert body["tier"] == "core"
        assert body["tags"] == ["a", "b"]
        assert body["about_self"] is True
        assert body["space"] == "proj"
        return _mock_response({"text": "Stored."})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(
        [
            "remember", "content here",
            "--title", "My Title",
            "--type", "decision",
            "--tier", "core",
            "--tags", "a,b",
            "--about-self",
            "--space", "proj",
        ],
        monkeypatch,
    )
    assert code == 0


def test_remember_from_stdin(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["content"] == "piped content\n"
        return _mock_response({"text": "Stored."})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["remember", "-"], monkeypatch, stdin_text="piped content\n")
    assert code == 0


# --- ingest ---


def test_ingest_from_file(monkeypatch, tmp_path):
    log_file = tmp_path / "conv.log"
    log_file.write_text("User: Hello\nAssistant: Hi")

    def handler(request):
        body = json.loads(request.content)
        assert "Hello" in body["content"]
        return _mock_response({
            "extracted": 1,
            "memories": [{"title": "Greeting", "node_id": "abcd1234-0000"}],
        })

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["ingest", str(log_file)], monkeypatch)
    assert code == 0
    assert "Extracted 1" in out
    assert "Greeting" in out


def test_ingest_from_stdin(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["content"] == "stdin conversation\n"
        return _mock_response({"extracted": 0})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["ingest", "-"], monkeypatch, stdin_text="stdin conversation\n")
    assert code == 0
    assert "No new memories" in out


def test_ingest_error_response(monkeypatch):
    def handler(request):
        return _mock_response({"status": "error", "result": "LLM failed"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr("ormah.adapters.cli_adapter.resolve_space", lambda x: None)
    code, out, err = _run_cli(["ingest", "-"], monkeypatch, stdin_text="text")
    assert code == 1
    assert "LLM failed" in err


# --- node ---


def test_node_text(monkeypatch):
    def handler(request):
        assert "/agent/recall/abc-123" in str(request.url)
        return _mock_response({"text": "[fact] My memory\nContent here"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(["node", "abc-123"], monkeypatch)
    assert code == 0
    assert "My memory" in out


def test_node_json(monkeypatch):
    response_data = {"text": "info", "node": {"id": "abc-123"}}

    def handler(request):
        return _mock_response(response_data)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(["node", "abc-123", "--json"], monkeypatch)
    assert code == 0
    parsed = json.loads(out)
    assert parsed["node"]["id"] == "abc-123"


# --- error handling ---


def test_server_not_running(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(["status"], monkeypatch)
    assert code == 1
    assert "not running" in err


def test_http_error(monkeypatch):
    def handler(request):
        return _mock_response({"detail": "bad request"}, status_code=400)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    code, out, err = _run_cli(["status"], monkeypatch)
    assert code == 1
    assert "400" in err


# --- argparse ---


# --- whisper inject ---


def test_whisper_inject_returns_context(monkeypatch):
    def handler(request):
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["prompt"] == "graph clustering"
        assert body["space"] == "ormah"
        return _mock_response({"text": "# Relevant memories\n- clustering uses Louvain"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "ormah",
    )
    hook_input = json.dumps({"prompt": "graph clustering", "cwd": "/path/to/ormah", "session_id": "test"})
    code, out, err = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Louvain" in parsed["hookSpecificOutput"]["additionalContext"]


def test_whisper_inject_empty_prompt(monkeypatch):
    hook_input = json.dumps({"prompt": "", "cwd": "/path/to/proj", "session_id": "test"})
    code, out, err = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    assert out.strip() == ""


def test_whisper_inject_malformed_json(monkeypatch):
    code, out, err = _run_cli(["whisper", "inject"], monkeypatch, stdin_text="not json")
    assert code == 0
    assert out.strip() == ""


def test_whisper_inject_server_down(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "proj",
    )
    hook_input = json.dumps({"prompt": "hello", "cwd": "/path/to/proj", "session_id": "test"})
    code, out, err = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    assert out.strip() == ""


def test_whisper_inject_empty_context(monkeypatch):
    def handler(request):
        return _mock_response({"text": ""})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "proj",
    )
    # Isolate from real cursor file to prevent flaky nudge triggers
    monkeypatch.setattr("ormah.adapters.cli_adapter._load_cursors", lambda: {})
    monkeypatch.setattr("ormah.adapters.cli_adapter._save_cursors", lambda c: None)
    hook_input = json.dumps({"prompt": "hello", "cwd": "/path", "session_id": "test"})
    code, out, err = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    assert out.strip() == ""


def test_whisper_inject_no_cwd(monkeypatch):
    """When cwd is missing, space should be None (no space key in body)."""
    def handler(request):
        assert request.method == "POST"
        body = json.loads(request.content)
        assert "space" not in body
        assert body["prompt"] == "hello"
        return _mock_response({"text": "some context"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    hook_input = json.dumps({"prompt": "hello", "session_id": "test"})
    code, out, err = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    parsed = json.loads(out)
    assert "additionalContext" in parsed["hookSpecificOutput"]


# --- whisper inject nudge ---


def test_whisper_inject_nudge_at_interval(monkeypatch, tmp_path):
    """Nudge appears at the Nth prompt (default 10)."""
    cursor_file = tmp_path / "whisper-cursors.json"
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_FILE", cursor_file)
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_DIR", tmp_path)
    monkeypatch.setattr("ormah.config.settings.whisper_nudge_interval", 3)

    def handler(request):
        return _mock_response({"text": "some context"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "proj",
    )

    session_id = "nudge-test-session"
    hook_input = json.dumps({"prompt": "hello", "cwd": "/path", "session_id": session_id})

    # Prompts 1 and 2 — no nudge
    for i in range(2):
        code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        parsed = json.loads(out)
        assert "remember tool" not in parsed["hookSpecificOutput"]["additionalContext"]

    # Prompt 3 — nudge should appear
    code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    parsed = json.loads(out)
    assert "remember tool" in parsed["hookSpecificOutput"]["additionalContext"]


def test_whisper_inject_nudge_disabled(monkeypatch, tmp_path):
    """Nudge never appears when interval is 0."""
    cursor_file = tmp_path / "whisper-cursors.json"
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_FILE", cursor_file)
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_DIR", tmp_path)
    monkeypatch.setattr("ormah.config.settings.whisper_nudge_interval", 0)

    def handler(request):
        return _mock_response({"text": "some context"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "proj",
    )

    hook_input = json.dumps({"prompt": "hello", "cwd": "/path", "session_id": "disabled-test"})

    for _ in range(15):
        code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        parsed = json.loads(out)
        assert "remember tool" not in parsed["hookSpecificOutput"]["additionalContext"]


def test_whisper_inject_nudge_resets_per_session(monkeypatch, tmp_path):
    """Each session_id gets its own counter."""
    cursor_file = tmp_path / "whisper-cursors.json"
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_FILE", cursor_file)
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_DIR", tmp_path)
    monkeypatch.setattr("ormah.config.settings.whisper_nudge_interval", 2)

    def handler(request):
        return _mock_response({"text": "some context"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "proj",
    )

    # Session A: prompt 1
    hook_a = json.dumps({"prompt": "hello", "cwd": "/path", "session_id": "session-a"})
    code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_a)
    assert code == 0
    parsed = json.loads(out)
    assert "remember tool" not in parsed["hookSpecificOutput"]["additionalContext"]

    # Session B: prompt 1 — counter is independent
    hook_b = json.dumps({"prompt": "hello", "cwd": "/path", "session_id": "session-b"})
    code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_b)
    assert code == 0
    parsed = json.loads(out)
    assert "remember tool" not in parsed["hookSpecificOutput"]["additionalContext"]

    # Session A: prompt 2 — should trigger nudge (interval=2)
    code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_a)
    assert code == 0
    parsed = json.loads(out)
    assert "remember tool" in parsed["hookSpecificOutput"]["additionalContext"]

    # Session B: prompt 2 — also triggers nudge independently
    code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_b)
    assert code == 0
    parsed = json.loads(out)
    assert "remember tool" in parsed["hookSpecificOutput"]["additionalContext"]


def test_whisper_inject_nudge_on_empty_whisper(monkeypatch, tmp_path):
    """Nudge appears even when the server returns empty whisper context."""
    cursor_file = tmp_path / "whisper-cursors.json"
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_FILE", cursor_file)
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_DIR", tmp_path)
    monkeypatch.setattr("ormah.config.settings.whisper_nudge_interval", 1)

    def handler(request):
        return _mock_response({"text": ""})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter._whisper_client",
        lambda: httpx.Client(transport=transport, base_url="http://test"),
    )
    monkeypatch.setattr(
        "ormah.adapters.cli_adapter.detect_space_from_dir",
        lambda path: "proj",
    )

    hook_input = json.dumps({"prompt": "hello", "cwd": "/path", "session_id": "empty-test"})
    code, out, _ = _run_cli(["whisper", "inject"], monkeypatch, stdin_text=hook_input)
    assert code == 0
    parsed = json.loads(out)
    assert "remember tool" in parsed["hookSpecificOutput"]["additionalContext"]


# --- whisper setup ---


def test_whisper_setup_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    code, out, err = _run_cli(["whisper", "setup"], monkeypatch)
    assert code == 0
    settings_path = tmp_path / ".claude" / "settings.local.json"
    assert settings_path.exists()
    data = json.loads(settings_path.read_text())
    assert "hooks" in data
    assert "UserPromptSubmit" in data["hooks"]
    hook = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert "whisper inject" in hook["command"]
    assert hook["timeout"] == 10


def test_whisper_setup_global(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    code, out, err = _run_cli(["whisper", "setup", "--global"], monkeypatch)
    assert code == 0
    settings_path = fake_home / ".claude" / "settings.json"
    assert settings_path.exists()
    data = json.loads(settings_path.read_text())
    assert "hooks" in data


def test_whisper_setup_merges_existing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    existing = {"preferredNotifChannel": "terminal", "other": True}
    (claude_dir / "settings.local.json").write_text(json.dumps(existing))

    code, out, err = _run_cli(["whisper", "setup"], monkeypatch)
    assert code == 0
    data = json.loads((claude_dir / "settings.local.json").read_text())
    assert data["other"] is True  # preserved
    assert "hooks" in data  # added


# --- argparse ---


def test_no_command(monkeypatch):
    code, out, err = _run_cli([], monkeypatch)
    assert code == 2  # argparse exits with 2 for missing required args
