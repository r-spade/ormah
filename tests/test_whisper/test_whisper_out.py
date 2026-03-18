"""Tests for whisper-out: involuntary memory storage on PreCompact."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
import pytest

from ormah.adapters.cli_adapter import main


def _run_cli(args: list[str], monkeypatch, stdin_text: str | None = None):
    """Run the CLI with given args, returning (exit_code, stdout, stderr)."""
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
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "http://test"),
    )


def _make_transcript(user_turns: int = 6) -> str:
    """Create a minimal JSONL transcript with the given number of user turns."""
    lines = []
    for i in range(user_turns):
        lines.append(json.dumps({
            "type": "user",
            "message": {"content": f"User message {i} about important architecture decisions"},
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": f"Response {i} with details"}]},
        }))
    return "\n".join(lines) + "\n"


@pytest.fixture(autouse=True)
def _isolate_cursors(tmp_path, monkeypatch):
    """Point cursor file to tmp_path so tests don't share state."""
    cursor_dir = tmp_path / "cursors"
    cursor_dir.mkdir()
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_DIR", cursor_dir)
    monkeypatch.setattr("ormah.adapters.cli_adapter._WHISPER_CURSOR_FILE", cursor_dir / "whisper-cursors.json")


class TestWhisperStoreBasic:
    def test_whisper_store_basic(self, monkeypatch, tmp_path):
        """Mock transcript + mock HTTP → memories extracted and stored."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        captured_requests = []

        def handler(request):
            captured_requests.append(request)
            return _mock_response({"status": "processed", "extracted": 2, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/Users/someone/Projects/myapp",
            "session_id": "abc123",
            "trigger": "auto",
        })

        code, out, err = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_requests) == 1

    def test_whisper_store_reads_transcript(self, monkeypatch, tmp_path):
        """Verify parse_transcript is called with correct path."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        captured_paths = []

        from ormah.transcript import parser as parser_mod
        original_parse = parser_mod.parse_transcript

        def tracking_parse(path, **kwargs):
            captured_paths.append(str(path))
            return original_parse(path, **kwargs)

        monkeypatch.setattr("ormah.transcript.parser.parse_transcript", tracking_parse)

        def handler(request):
            return _mock_response({"status": "processed", "extracted": 0, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "abc",
            "trigger": "auto",
        })

        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_paths) == 1
        assert captured_paths[0] == str(transcript)


class TestWhisperStoreSkips:
    def test_whisper_store_skips_short_session(self, monkeypatch, tmp_path):
        """< min_turns → silent exit, no HTTP call."""
        transcript = tmp_path / "short.jsonl"
        transcript.write_text(_make_transcript(2))  # Only 2 turns, below default 5

        captured_requests = []

        def handler(request):
            captured_requests.append(request)
            return _mock_response({"status": "processed", "extracted": 0, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "abc",
            "trigger": "auto",
        })

        code, out, err = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_requests) == 0  # No HTTP call made


class TestWhisperStoreSilentOnError:
    def test_whisper_store_silent_on_error(self, monkeypatch, tmp_path):
        """Server down → exit 0 (no crash)."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        def handler(request):
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "abc",
            "trigger": "auto",
        })

        code, out, err = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0


class TestWhisperStoreSpace:
    def test_whisper_store_resolves_space(self, monkeypatch, tmp_path):
        """cwd="/path/to/ormah" → space="ormah" in request params."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        captured_requests = []

        def handler(request):
            captured_requests.append(request)
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": str(tmp_path),  # space will be the dir name
            "session_id": "abc",
            "trigger": "auto",
        })

        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_requests) == 1

        url = str(captured_requests[0].url)
        assert "default_space=" in url


class TestWhisperStoreExtraTags:
    def test_whisper_store_passes_extra_tags(self, monkeypatch, tmp_path):
        """Verify "whisper-out" tag in request."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        captured_requests = []

        def handler(request):
            captured_requests.append(request)
            return _mock_response({"status": "processed", "extracted": 0, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "abc",
            "trigger": "auto",
        })

        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_requests) == 1

        url = str(captured_requests[0].url)
        assert "extra_tags=whisper-out" in url


class TestWhisperStoreCursor:
    def test_cursor_saves_after_success(self, monkeypatch, tmp_path):
        """After successful extraction, cursor is persisted."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        def handler(request):
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "sess1",
            "trigger": "auto",
        })

        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        cursors = json.loads(_WHISPER_CURSOR_FILE.read_text())
        assert "sess1" in cursors
        assert cursors["sess1"] > 0
        assert cursors["sess1"] == transcript.stat().st_size

    def test_cursor_skips_already_processed(self, monkeypatch, tmp_path):
        """Second run on unchanged file → no HTTP call."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))
        file_size = transcript.stat().st_size

        # Pre-seed cursor at end of file
        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE, _WHISPER_CURSOR_DIR
        _WHISPER_CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        _WHISPER_CURSOR_FILE.write_text(json.dumps({"sess1": file_size}))

        captured_requests = []

        def handler(request):
            captured_requests.append(request)
            return _mock_response({"status": "processed", "extracted": 0, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "sess1",
            "trigger": "auto",
        })

        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_requests) == 0  # Skipped — already processed

    def test_cursor_processes_only_new_content(self, monkeypatch, tmp_path):
        """After appending new turns, only new content is sent."""
        transcript = tmp_path / "session.jsonl"
        part1 = _make_transcript(6)
        transcript.write_text(part1)
        part1_size = transcript.stat().st_size

        # Pre-seed cursor at end of part1
        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE, _WHISPER_CURSOR_DIR
        _WHISPER_CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        _WHISPER_CURSOR_FILE.write_text(json.dumps({"sess1": part1_size}))

        # Append more turns
        part2 = _make_transcript(6)
        with open(transcript, "a") as f:
            f.write(part2)

        captured_bodies = []

        def handler(request):
            captured_bodies.append(json.loads(request.content))
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "sess1",
            "trigger": "auto",
        })

        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_bodies) == 1

        # The sent content should NOT contain part1's text
        sent_content = captured_bodies[0]["content"]
        assert "User message 0" not in sent_content or sent_content.count("User message 0") == 1
        # But should contain the new turns (part2 also starts at message 0,
        # so just verify it's much smaller than the full transcript)
        from ormah.transcript.parser import parse_transcript
        full = parse_transcript(transcript)
        assert len(sent_content) < len(full.conversation)

    def test_cursor_not_saved_on_error(self, monkeypatch, tmp_path):
        """On HTTP error, cursor is NOT updated."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        def handler(request):
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )

        hook_input = json.dumps({
            "transcript_path": str(transcript),
            "cwd": "/tmp",
            "session_id": "sess1",
            "trigger": "auto",
        })

        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        assert not _WHISPER_CURSOR_FILE.exists()


class TestIngestEndpointExtraTags:
    def test_ingest_endpoint_extra_tags(self, engine):
        """HTTP test: extra_tags query param applied to created memories."""
        fake_llm_response = json.dumps({
            "memories": [
                {
                    "content": "The project uses SQLite for storage",
                    "type": "fact",
                    "title": "SQLite storage",
                    "tags": ["architecture"],
                },
            ]
        })
        with patch("ormah.background.llm_client.llm_generate", return_value=fake_llm_response):
            result = engine.ingest_conversation(
                content="A conversation about database choices and architecture." * 10,
                extra_tags=["whisper-out"],
            )

        assert isinstance(result, list)
        assert len(result) == 1
        node_id = result[0]["node_id"]

        node = engine.file_store.load(node_id)
        assert node is not None
        assert "whisper-out" in node.tags
        assert "auto-ingested" in node.tags


class TestWhisperSetup:
    def test_whisper_setup_includes_precompact(self, monkeypatch, tmp_path):
        """Setup generates both UserPromptSubmit and PreCompact hooks when whisper_out_enabled."""
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter.settings",
            MagicMock(port=8787, whisper_out_enabled=True),
        )

        code, out, err = _run_cli(
            ["whisper", "setup", "--global"],
            monkeypatch,
        )

        assert code == 0
        assert "PreCompact" in out
        assert "UserPromptSubmit" in out

    def test_whisper_setup_always_registers_precompact(self, monkeypatch, tmp_path):
        """Setup always registers PreCompact hook (runtime flag gates execution, not registration)."""
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter.settings",
            MagicMock(port=8787, whisper_out_enabled=False, whisper_out_min_turns=5),
        )

        code, out, err = _run_cli(
            ["whisper", "setup", "--global"],
            monkeypatch,
        )

        assert code == 0
        assert "PreCompact" in out
        assert "UserPromptSubmit" in out
