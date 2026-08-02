"""Tests for whisper-out: involuntary memory storage on PreCompact."""

from __future__ import annotations

import io
import json
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
            "message": {"stop_reason": "end_turn",
                        "content": [{"type": "text", "text": f"Response {i} with details"}]},
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

    def test_whisper_store_resolves_transcript_from_session_id(self, monkeypatch, tmp_path):
        """When transcript_path is absent, resolve it from session_id."""
        transcript = tmp_path / "resolved.jsonl"
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
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._resolve_transcript_path",
            lambda session_id: transcript if session_id == "abc" else None,
        )

        hook_input = json.dumps({
            "cwd": "/tmp",
            "session_id": "abc",
            "trigger": "auto",
        })

        code, out, err = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        assert len(captured_requests) == 1


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


class TestResolveTranscriptPath:
    def test_resolve_transcript_path_prefers_claude_exact_match(self, monkeypatch, tmp_path):
        claude_root = tmp_path / ".claude" / "projects" / "proj"
        claude_root.mkdir(parents=True)
        transcript = claude_root / "sess-123.jsonl"
        transcript.write_text(_make_transcript(1))

        monkeypatch.setenv("HOME", str(tmp_path))

        from ormah.adapters.cli_adapter import _resolve_transcript_path

        assert _resolve_transcript_path("sess-123") == transcript

    def test_resolve_transcript_path_finds_codex_rollout_file(self, monkeypatch, tmp_path):
        codex_root = tmp_path / ".codex" / "sessions" / "2026" / "04" / "02"
        codex_root.mkdir(parents=True)
        transcript = codex_root / "rollout-2026-04-02T17-34-35-sess-456.jsonl"
        transcript.write_text(_make_transcript(1))

        monkeypatch.setenv("HOME", str(tmp_path))

        from ormah.adapters.cli_adapter import _resolve_transcript_path

        assert _resolve_transcript_path("sess-456") == transcript


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

    def test_cursor_holds_back_dangling_prompt(self, monkeypatch, tmp_path):
        """A store fired while a prompt's response is not written yet must not ingest the
        dangling prompt or advance the cursor past it — the prompt stays with its response
        on the next run (no split)."""
        monkeypatch.setattr("ormah.adapters.cli_adapter.settings.whisper_out_min_turns", 1)
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))  # 6 complete (end_turn) turns
        with transcript.open("a") as f:
            f.write(json.dumps({"type": "user",
                "message": {"content": "Dangling prompt about the new feature"}}) + "\n")

        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )
        hook_input = json.dumps({
            "transcript_path": str(transcript), "cwd": "/tmp",
            "session_id": "sess1", "trigger": "auto",
        })

        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        cursors = json.loads(_WHISPER_CURSOR_FILE.read_text())
        assert "Dangling prompt" not in bodies[0]["content"]       # held back
        assert cursors["sess1"] < transcript.stat().st_size        # cursor before it

        # The response arrives; the next run pairs the prompt with its response.
        with transcript.open("a") as f:
            f.write(json.dumps({"type": "assistant",
                "message": {"stop_reason": "end_turn",
                            "content": [{"type": "text", "text": "Answer to the new feature"}]}}) + "\n")
        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert "Dangling prompt" in bodies[1]["content"]
        assert "Answer to the new feature" in bodies[1]["content"]

    def test_legacy_mid_response_cursor_recovered(self, monkeypatch, tmp_path):
        """A whisper cursor left mid-response by an older version is recovered: the store
        re-parses from 0 so the dropped tail is sent with its prompt, not orphaned."""
        monkeypatch.setattr("ormah.adapters.cli_adapter.settings.whisper_out_min_turns", 1)
        transcript = tmp_path / "session.jsonl"
        records = [
            {"type": "user", "message": {"content": "Prompt about the architecture decision"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "text", "text": "First part"}]}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Second part answer"}]}},
        ]
        transcript.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        raw = transcript.read_bytes().splitlines(keepends=True)
        mid = len(raw[0]) + len(raw[1])  # cursor saved mid-response

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE, _WHISPER_CURSOR_DIR
        _WHISPER_CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        _WHISPER_CURSOR_FILE.write_text(json.dumps({"sess1": mid}))

        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )
        hook_input = json.dumps({
            "transcript_path": str(transcript), "cwd": "/tmp",
            "session_id": "sess1", "trigger": "auto",
        })
        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)

        assert "Prompt about the architecture decision" in bodies[0]["content"]
        assert "First part" in bodies[0]["content"]
        assert "Second part answer" in bodies[0]["content"]

    def test_api_error_orphan_advances_cursor_without_full_reextract(
        self, monkeypatch, tmp_path
    ):
        """ADR-0003 regression (bug #149, hook path): a false-positive leading_orphan —
        an assistant 'API Error' record right after the end_turn boundary the cursor is
        parked on — must not re-send the whole transcript. The orphan is dropped and only
        the tail past the cursor is sent; the cursor advances to EOF."""
        monkeypatch.setattr("ormah.adapters.cli_adapter.settings.whisper_out_min_turns", 1)
        transcript = tmp_path / "session.jsonl"
        first_turn = [
            {"type": "user", "message": {"content": "Prompt one"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Answer one"}]}},
        ]
        tail = [
            {"type": "assistant", "message": {"stop_reason": "stop_sequence",
                "content": [{"type": "text",
                    "text": "API Error: Connection closed mid-response."}]}},
            {"type": "user", "message": {"content": "continue"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Answer two"}]}},
        ]
        transcript.write_text("\n".join(json.dumps(r) for r in first_turn) + "\n")
        from ormah.transcript.parser import parse_transcript
        boundary = parse_transcript(transcript).safe_end_offset
        with transcript.open("a") as f:
            for r in tail:
                f.write(json.dumps(r) + "\n")

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_DIR, _WHISPER_CURSOR_FILE
        _WHISPER_CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        _WHISPER_CURSOR_FILE.write_text(json.dumps({"sess1": boundary}))

        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )
        hook_input = json.dumps({
            "transcript_path": str(transcript), "cwd": "/tmp",
            "session_id": "sess1", "trigger": "auto",
        })

        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)

        assert len(bodies) == 1
        assert "Answer one" not in bodies[0]["content"]  # no whole-file re-extract
        assert "API Error" not in bodies[0]["content"]   # orphan fragment dropped
        assert "continue" in bodies[0]["content"]        # stranded tail recovered
        cursors = json.loads(_WHISPER_CURSOR_FILE.read_text())
        assert cursors["sess1"] == transcript.stat().st_size  # cursor monotonic, at EOF

    def test_inflight_orphan_rewind_parks_cursor_unchanged(self, monkeypatch, tmp_path):
        """ADR-0003 critical regression (Codex review, #149): an orphan with NO forward
        progress whose rewind ALSO makes no progress — a still-open in-flight response, not
        a genuinely recoverable one — must park: no POST, cursor untouched."""
        monkeypatch.setattr("ormah.adapters.cli_adapter.settings.whisper_out_min_turns", 1)
        transcript = tmp_path / "session.jsonl"

        closed_turn = [
            {"type": "user", "message": {"content": "Prompt about the release plan"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Answer about the release plan"}]}},
        ]
        transcript.write_text("\n".join(json.dumps(r) for r in closed_turn) + "\n")
        from ormah.transcript.parser import parse_transcript
        boundary = parse_transcript(transcript).safe_end_offset

        with transcript.open("a") as f:
            f.write(json.dumps({"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "text", "text": "In-flight fragment"}]}}) + "\n")

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_DIR, _WHISPER_CURSOR_FILE
        _WHISPER_CURSOR_DIR.mkdir(parents=True, exist_ok=True)
        _WHISPER_CURSOR_FILE.write_text(json.dumps({"sess1": boundary}))

        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return _mock_response({"status": "processed", "extracted": 1, "memories": []})

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )
        hook_input = json.dumps({
            "transcript_path": str(transcript), "cwd": "/tmp",
            "session_id": "sess1", "trigger": "auto",
        })

        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)

        assert bodies == []  # never re-ingested
        cursors = json.loads(_WHISPER_CURSOR_FILE.read_text())
        assert cursors["sess1"] == boundary  # cursor left untouched

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

    def test_cursor_not_advanced_on_extraction_error(self, monkeypatch, tmp_path):
        """Server responds HTTP 200 with {"status":"error"} (e.g. claude_cli extraction
        failed on timeout/is_error) — cursor must NOT advance so the slice is retried,
        unlike a legitimate empty extraction (status:"processed", extracted:0)."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        def handler(request):
            return _mock_response(
                {"status": "error", "result": "boom", "extracted": 0, "memories": []}
            )

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

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        assert not _WHISPER_CURSOR_FILE.exists()

    def test_cursor_advances_on_empty_processed_extraction(self, monkeypatch, tmp_path):
        """status:"processed" with extracted==0 is a legitimate empty extraction and MUST
        still advance the cursor, else the same slice reprocesses forever."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

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
            "session_id": "sess1",
            "trigger": "auto",
        })

        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        cursors = json.loads(_WHISPER_CURSOR_FILE.read_text())
        assert cursors["sess1"] == transcript.stat().st_size

    def test_non_dict_200_body_does_not_crash_or_advance(self, monkeypatch, tmp_path):
        """A rogue proxy may return HTTP 200 with a valid-but-non-object JSON body
        (null / list / number). The hook must not raise (its contract is 'never block
        compaction, exit silently') and must not advance the cursor (unconfirmed success)."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        def handler(request):
            # A literal JSON null body: r.json() returns None (not an object), so
            # resp.get("status") would raise AttributeError unless guarded.
            return httpx.Response(200, content=b"null", headers={"content-type": "application/json"})

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

        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        assert not _WHISPER_CURSOR_FILE.exists()

    @pytest.mark.parametrize("body", [
        {"extracted": 0, "memories": []},            # missing status
        {"status": "queued", "extracted": 0},         # unrecognized status
        {"status": None},                              # null status
    ])
    def test_cursor_not_advanced_on_unknown_200_status(self, monkeypatch, tmp_path, body):
        """Only status:"processed" advances the cursor. A 200 dict with a missing or
        unrecognized status is not a confirmed success — do NOT advance, so the slice is
        retried rather than silently lost."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        transport = httpx.MockTransport(lambda request: _mock_response(body))
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )
        hook_input = json.dumps({
            "transcript_path": str(transcript), "cwd": "/tmp",
            "session_id": "sess1", "trigger": "auto",
        })
        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        assert not _WHISPER_CURSOR_FILE.exists()

    def test_cursor_not_advanced_on_client_timeout(self, monkeypatch, tmp_path):
        """If the ingest POST itself times out (claude_cli extraction outran the client
        budget), the hook exits silently and the cursor is NOT advanced, so the slice is
        retried on the next run instead of being lost."""
        transcript = tmp_path / "session.jsonl"
        transcript.write_text(_make_transcript(6))

        def handler(request):
            raise httpx.ReadTimeout("extraction outran the client timeout", request=request)

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(
            "ormah.adapters.cli_adapter._whisper_store_client",
            lambda: httpx.Client(transport=transport, base_url="http://test"),
        )
        hook_input = json.dumps({
            "transcript_path": str(transcript), "cwd": "/tmp",
            "session_id": "sess1", "trigger": "auto",
        })
        code, _, _ = _run_cli(["whisper", "store"], monkeypatch, stdin_text=hook_input)
        assert code == 0
        from ormah.adapters.cli_adapter import _WHISPER_CURSOR_FILE
        assert not _WHISPER_CURSOR_FILE.exists()

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
        with patch("ormah.background.llm_client.ingest_llm_generate", return_value=fake_llm_response):
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
