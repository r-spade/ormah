"""Tests for the Claude Code JSONL transcript parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ormah.transcript.parser import parse_transcript


def _write_jsonl(tmp_path: Path, lines: list[dict], name: str = "abc123.jsonl") -> Path:
    """Write a list of dicts as JSONL to a temp file and return the path."""
    path = tmp_path / name
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


class TestParseTranscript:
    def test_user_and_assistant_text_extracted(self, tmp_path):
        lines = [
            {"type": "user", "message": {"content": "Hello there"}},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Hi! How can I help?"},
                        {"type": "tool_use", "name": "read", "input": {}},
                    ]
                },
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "User: Hello there" in result.conversation
        assert "Assistant: Hi! How can I help?" in result.conversation
        assert "tool_use" not in result.conversation
        assert result.user_turn_count == 1

    def test_user_raw_string_content(self, tmp_path):
        lines = [
            {"type": "user", "message": {"content": "raw string message"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "User: raw string message" in result.conversation
        assert result.user_turn_count == 1

    def test_user_list_with_text_blocks(self, tmp_path):
        lines = [
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "text", "text": "first part"},
                        {"type": "text", "text": "second part"},
                    ]
                },
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "first part" in result.conversation
        assert "second part" in result.conversation
        assert result.user_turn_count == 1

    def test_user_tool_result_only_skipped(self, tmp_path):
        lines = [
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "xyz", "content": "ok"},
                    ]
                },
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.conversation == ""
        assert result.user_turn_count == 0

    def test_assistant_thinking_blocks_stripped(self, tmp_path):
        lines = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "let me think..."},
                        {"type": "text", "text": "Here is the answer"},
                    ]
                },
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "Here is the answer" in result.conversation
        assert "let me think" not in result.conversation

    def test_assistant_tool_use_stripped(self, tmp_path):
        lines = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "bash", "input": {"command": "ls"}},
                        {"type": "text", "text": "I ran the command"},
                    ]
                },
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "I ran the command" in result.conversation
        assert "bash" not in result.conversation
        assert "ls" not in result.conversation

    def test_progress_lines_skipped(self, tmp_path):
        lines = [
            {"type": "progress", "data": {"status": "running"}},
            {"type": "user", "message": {"content": "hello"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "progress" not in result.conversation.lower()
        assert result.user_turn_count == 1

    def test_system_lines_skipped(self, tmp_path):
        lines = [
            {"type": "system", "message": {"content": "system prompt here"}},
            {"type": "user", "message": {"content": "hello"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "system prompt" not in result.conversation
        assert result.user_turn_count == 1

    def test_file_history_snapshot_skipped(self, tmp_path):
        lines = [
            {"type": "file-history-snapshot", "data": {}},
            {"type": "user", "message": {"content": "hello"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.user_turn_count == 1

    def test_queue_operation_skipped(self, tmp_path):
        lines = [
            {"type": "queue-operation", "data": {}},
            {"type": "user", "message": {"content": "hello"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.user_turn_count == 1

    def test_user_turn_count_only_text_bearing(self, tmp_path):
        lines = [
            {"type": "user", "message": {"content": "real message"}},
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
                    ]
                },
            },
            {"type": "user", "message": {"content": "another real message"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.user_turn_count == 2

    def test_session_id_from_filename(self, tmp_path):
        path = _write_jsonl(
            tmp_path,
            [{"type": "user", "message": {"content": "hi"}}],
            name="031e557c-ca08-4835-b7f1-3cb52354decb.jsonl",
        )
        result = parse_transcript(path)
        assert result.session_id == "031e557c-ca08-4835-b7f1-3cb52354decb"

    def test_malformed_lines_handled_gracefully(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write("not valid json\n")
            f.write("\n")
            f.write(json.dumps({"type": "user", "message": {"content": "valid"}}) + "\n")
            f.write("{incomplete\n")
        result = parse_transcript(path)
        assert "User: valid" in result.conversation
        assert result.user_turn_count == 1

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = parse_transcript(path)
        assert result.conversation == ""
        assert result.user_turn_count == 0
        assert result.cleaned_chars == 0

    def test_total_chars_reflects_original_size(self, tmp_path):
        lines = [
            {"type": "user", "message": {"content": "hello"}},
            {"type": "progress", "data": {"x": "y" * 1000}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)
        assert result.total_chars == path.stat().st_size
        assert result.cleaned_chars < result.total_chars

    def test_alternating_turns_format(self, tmp_path):
        lines = [
            {"type": "user", "message": {"content": "Q1"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "A1"}]},
            },
            {"type": "user", "message": {"content": "Q2"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "A2"}]},
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.conversation == "User: Q1\n\nAssistant: A1\n\nUser: Q2\n\nAssistant: A2"
