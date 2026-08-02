"""Tests for agent JSONL transcript normalization."""

from __future__ import annotations

import json
from pathlib import Path

from ormah.transcript.parser import extract_user_prompts, parse_transcript, should_rewind


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
        assert result.source == "claude_code"
        assert [(turn.role, turn.text) for turn in result.turns] == [
            ("user", "Hello there"),
            ("assistant", "Hi! How can I help?"),
        ]

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
            {"type": "user", "message": {"content": "the question"}},
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
            {"type": "user", "message": {"content": "list the files"}},
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

    def test_codex_response_item_messages_extracted(self, tmp_path):
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "How does whisper work?"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "It injects relevant memory context."}],
                },
            },
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines, name="codex.jsonl"))
        assert "User: How does whisper work?" in result.conversation
        assert "Assistant: It injects relevant memory context." in result.conversation
        assert result.user_turn_count == 1
        assert result.source == "codex"
        assert [(turn.role, turn.text) for turn in result.turns] == [
            ("user", "How does whisper work?"),
            ("assistant", "It injects relevant memory context."),
        ]

    def test_codex_bootstrap_user_message_skipped(self, tmp_path):
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /tmp/demo\n\n<INSTRUCTIONS>...</INSTRUCTIONS>\n<environment_context>...</environment_context>",
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Please review the diff"}],
                },
            },
        ]
        path = _write_jsonl(tmp_path, lines, name="codex-bootstrap.jsonl")
        result = parse_transcript(path)
        prompts = extract_user_prompts(path)
        assert result.user_turn_count == 1
        assert result.conversation == "User: Please review the diff"
        assert prompts == ["Please review the diff"]

    def test_codex_turn_aborted_wrapper_skipped(self, tmp_path):
        lines = [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<turn_aborted>\nThe user interrupted the previous turn on purpose.\n</turn_aborted>",
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Show me the current metrics"}],
                },
            },
        ]
        path = _write_jsonl(tmp_path, lines, name="codex-turn-aborted.jsonl")
        result = parse_transcript(path)
        prompts = extract_user_prompts(path)
        assert result.user_turn_count == 1
        assert result.conversation == "User: Show me the current metrics"
        assert prompts == ["Show me the current metrics"]


class TestSafeBoundary:
    def test_safe_offset_and_payload_stop_at_last_complete_pair(self, tmp_path):
        """safe_* must exclude a dangling user turn; raw fields still include it."""
        lines = [
            {"type": "user",      "message": {"content": "Turn 1 user"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Turn 1 assistant"}]}},
            {"type": "user",      "message": {"content": "Turn 2 user"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Turn 2 assistant"}]}},
            {"type": "user",      "message": {"content": "Turn 3 dangling user"}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)

        assert result.user_turn_count == 3
        assert "Turn 3" in result.conversation

        assert result.safe_user_turn_count == 2
        assert "Turn 3" not in result.safe_conversation
        assert "Turn 2 assistant" in result.safe_conversation
        assert 0 < result.safe_end_offset < result.end_offset

        tail = parse_transcript(path, start_offset=result.safe_end_offset)
        assert tail.user_turn_count == 1
        assert "Turn 3" in tail.conversation

    def test_trailing_pair_is_unsafe_until_next_user(self, tmp_path):
        """A trailing pair with NO completion signal (no stop_reason field) is not safe
        yet — the response may still continue. It closes only at a terminal stop_reason,
        a Codex task_complete, or the next user turn."""
        lines = [
            {"type": "user",      "message": {"content": "U1"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "A1"}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)
        assert result.safe_end_offset == 0
        assert result.safe_user_turn_count == 0
        assert result.safe_conversation == ""
        # Raw fields still see the full pair.
        assert result.user_turn_count == 1
        assert "A1" in result.conversation

    def test_user_then_tooluse_then_text_is_one_pair(self, tmp_path):
        """tool_use followed by a text assistant must form ONE pair, not fragment.

        The pair becomes safe at the following user turn; both assistant records
        stay attached to the prompt.
        """
        lines = [
            {"type": "user",      "message": {"content": "Please read the file"}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "read", "input": {}}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Here is the summary"}]}},
            {"type": "user",      "message": {"content": "Thanks, next"}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)
        assert "Please read the file" in result.safe_conversation
        assert "Here is the summary" in result.safe_conversation
        assert "Thanks, next" not in result.safe_conversation
        assert result.safe_user_turn_count == 1

    def test_terminal_toolonly_assistant_does_not_advance_boundary(self, tmp_path):
        """A trailing tool-only assistant (no text) leaves the pair pending (known limitation)."""
        lines = [
            {"type": "user",      "message": {"content": "U1"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "A1"}]}},
            {"type": "user",      "message": {"content": "U2 asks for a tool"}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "read", "input": {}}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)
        assert result.safe_user_turn_count == 1
        assert "U2" not in result.safe_conversation
        assert result.safe_end_offset < result.end_offset

    def test_trailing_multi_record_assistant_is_unsafe_until_next_user(self, tmp_path):
        """A multi-record assistant response at EOF must not be committed mid-stream.

        The safe boundary advances only when the *next* user turn starts, so the
        cursor never lands between two assistant records of the same response.
        Otherwise the next incremental slice starts after the prompt and sees only
        the late assistant text, breaking whisper-feedback prompt<->response matching
        (r-spade PR #34 review, 2026-06-25).
        """
        lines = [
            {"type": "user",      "message": {"content": "the question"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "partial answer"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "final answer with the memory usage"}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)
        # No following user yet -> the whole block stays unsafe; cursor does not advance.
        assert result.safe_user_turn_count == 0
        assert result.safe_conversation == ""
        assert result.safe_end_offset == 0
        # Raw fields still include everything.
        assert "final answer" in result.conversation

    def test_multi_record_assistant_is_one_pair_after_next_user(self, tmp_path):
        """Once the next user turn arrives, the full multi-record response is one safe pair."""
        lines = [
            {"type": "user",      "message": {"content": "the question"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "partial answer"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "final answer with the memory usage"}]}},
            {"type": "user",      "message": {"content": "next question"}},
        ]
        path = _write_jsonl(tmp_path, lines)
        result = parse_transcript(path)
        assert result.safe_user_turn_count == 1
        assert "partial answer" in result.safe_conversation
        assert "final answer" in result.safe_conversation
        assert "next question" not in result.safe_conversation

    def test_terminal_stop_reason_closes_response_without_following_user(self, tmp_path):
        """A terminal stop_reason (Claude Code) closes the response immediately — the
        safe boundary covers the last pair even with no following user turn."""
        for reason in ("end_turn", "stop_sequence"):
            lines = [
                {"type": "user",      "message": {"content": "U1 question"}},
                {"type": "assistant", "message": {"stop_reason": reason,
                                                  "content": [{"type": "text", "text": "A1 answer"}]}},
            ]
            path = _write_jsonl(tmp_path, lines, name=f"{reason}.jsonl")
            result = parse_transcript(path)
            assert result.safe_end_offset == result.end_offset, reason
            assert result.safe_user_turn_count == 1, reason
            assert "A1 answer" in result.safe_conversation

    def test_tool_use_stop_reason_keeps_response_open_until_terminal(self, tmp_path):
        """A multi-record response (tool_use then end_turn) is one safe pair, never split
        — and the in-flight first record alone is held back."""
        # In flight: only the tool_use record so far -> nothing is safe yet.
        inflight = [
            {"type": "user",      "message": {"content": "the question"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                                              "content": [{"type": "text", "text": "partial answer"}]}},
        ]
        r1 = parse_transcript(_write_jsonl(tmp_path, inflight, name="inflight.jsonl"))
        assert r1.safe_user_turn_count == 0
        assert r1.safe_conversation == ""

        # Completed: the terminal record closes the whole response into one safe pair.
        done = inflight + [
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                                              "content": [{"type": "text", "text": "final answer"}]}},
        ]
        r2 = parse_transcript(_write_jsonl(tmp_path, done, name="done.jsonl"))
        assert r2.safe_user_turn_count == 1
        assert "partial answer" in r2.safe_conversation
        assert "final answer" in r2.safe_conversation
        assert r2.safe_end_offset == r2.end_offset

    def test_interrupt_closes_open_response_at_next_user(self, tmp_path):
        """If the user interrupts a non-terminal response, the next user turn still closes
        the open block (the partial response stays attached to its prompt)."""
        lines = [
            {"type": "user",      "message": {"content": "U1 question"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                                              "content": [{"type": "text", "text": "A1 partial"}]}},
            {"type": "user",      "message": {"content": "U2 interrupts"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        # U2 closes the U1 block; the partial A1 stays with U1, U2 is the new open block.
        assert result.safe_user_turn_count == 1
        assert "A1 partial" in result.safe_conversation
        assert "U2 interrupts" not in result.safe_conversation

    def test_slice_starting_with_orphan_assistant_is_dropped(self, tmp_path):
        """A slice that begins with assistant records (a cursor left mid-response by an
        older version) must not emit an assistant fragment without its prompt — the orphan
        is dropped and parsing resumes safely at the next user turn."""
        lines = [
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "orphan fragment"}]}},
            {"type": "user",      "message": {"content": "U1 real prompt"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "A1 answer"}]}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert "orphan fragment" not in result.safe_conversation
        assert "orphan fragment" not in result.conversation
        assert result.safe_user_turn_count == 1
        assert "U1 real prompt" in result.safe_conversation
        assert "A1 answer" in result.safe_conversation

    def test_orphan_only_slice_commits_nothing(self, tmp_path):
        """A slice with only orphan assistant content (no user) is held back entirely."""
        lines = [
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "orphan only"}]}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.safe_user_turn_count == 0
        assert result.safe_conversation == ""

    def test_legacy_mid_response_cursor_drops_orphan(self, tmp_path):
        """Incremental parse from a cursor an older version saved BETWEEN two assistant
        records (mid-response) must not emit the trailing fragment without its prompt — it
        is dropped and parsing resumes safely at the next user turn."""
        lines = [
            {"type": "user",      "message": {"content": "U1 prompt"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "text", "text": "first part of answer"}]}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "second part orphan"}]}},
            {"type": "user",      "message": {"content": "U2 next prompt"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "A2 answer"}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        raw = path.read_bytes().splitlines(keepends=True)
        legacy_offset = len(raw[0]) + len(raw[1])  # cursor after U1 + first assistant record
        result = parse_transcript(path, start_offset=legacy_offset)
        assert "second part orphan" not in result.safe_conversation
        assert result.safe_user_turn_count == 1
        assert "U2 next prompt" in result.safe_conversation
        assert "A2 answer" in result.safe_conversation
        # The slice is flagged so the caller re-parses from 0 to recover the dropped tail.
        assert result.leading_orphan is True
        # A normal incremental slice (starting on a user turn) is NOT flagged.
        clean = parse_transcript(path, start_offset=len(raw[0]) + len(raw[1]) + len(raw[2]))
        assert clean.leading_orphan is False

    def test_tool_result_user_before_assistant_is_not_orphan(self, tmp_path):
        """A slice starting on a tool_result user record (role=user, no text) followed by
        an assistant must NOT be flagged leading_orphan: a tool_result IS a user-role
        record, so the assistant is inside a proper turn, not stranded mid-response.

        Regression: tool_result users don't advance user_turn_count, so the old gate
        (user_turn_count == 0) false-flagged this slice as an orphan. The caller then reset
        the cursor to 0 every tick — an infinite re-ingest loop that never drained tool-heavy
        sessions and burned one extraction call per cycle (2026-07-03)."""
        lines = [
            {"type": "user",      "message": {"content": "U1 prompt"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "tool_use", "name": "read", "input": {}}]}},
            {"type": "user",      "message": {"content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "file body"}]}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "A1 answer after tool"}]}},
            {"type": "user",      "message": {"content": "U2 next prompt"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "A2 answer"}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        raw = path.read_bytes().splitlines(keepends=True)
        # cursor lands on the tool_result user line (line 3)
        offset = len(raw[0]) + len(raw[1])
        result = parse_transcript(path, start_offset=offset)
        assert result.leading_orphan is False
        # and the slice makes forward progress (no infinite reset-to-0 loop)
        assert result.safe_end_offset > offset

    def test_codex_task_complete_before_first_user_does_not_commit(self, tmp_path):
        """A Codex task_complete closing a turn whose user is before the cursor must not
        commit an orphan (no user in the slice)."""
        lines = [
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "orphan codex answer"}]}},
            {"type": "event_msg", "payload": {"type": "task_complete"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.safe_user_turn_count == 0
        assert result.safe_conversation == ""

    def test_max_tokens_and_refusal_are_terminal(self, tmp_path):
        """A response truncated by max_tokens (or refused) is complete — it must close
        the boundary, not be held back forever."""
        for reason in ("max_tokens", "refusal"):
            lines = [
                {"type": "user",      "message": {"content": "U1"}},
                {"type": "assistant", "message": {"stop_reason": reason,
                                                  "content": [{"type": "text", "text": "A1 truncated"}]}},
            ]
            result = parse_transcript(_write_jsonl(tmp_path, lines, name=f"{reason}.jsonl"))
            assert result.safe_end_offset == result.end_offset, reason
            assert result.safe_user_turn_count == 1, reason

    def test_codex_task_complete_closes_multirecord_turn(self, tmp_path):
        """A Codex turn (multi-record, no stop_reason) closes at its task_complete event,
        so the whole turn is one safe block — never split."""
        lines = [
            {"type": "response_item", "payload": {"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "the question"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "first part"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "second part"}]}},
            {"type": "event_msg", "payload": {"type": "task_complete"}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.source == "codex"
        assert result.safe_user_turn_count == 1
        assert "first part" in result.safe_conversation
        assert "second part" in result.safe_conversation
        assert result.safe_end_offset == result.end_offset

    def test_codex_turn_without_task_complete_is_held_back(self, tmp_path):
        """Without a task_complete (or following user), a Codex turn stays open: with no
        completion signal it is held back, never split."""
        lines = [
            {"type": "response_item", "payload": {"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "the question"}]}},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "an answer"}]}},
        ]
        result = parse_transcript(_write_jsonl(tmp_path, lines))
        assert result.safe_user_turn_count == 0
        assert result.safe_conversation == ""
        # The raw payload still has the content; it just isn't safe to commit yet.
        assert "an answer" in result.conversation


class TestShouldRewind:
    """ADR-0003: rewind only on NO forward progress; an orphan-with-progress is dropped."""

    def _api_error_lines(self):
        return [
            {"type": "user", "message": {"content": "Prompt one"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Answer one"}]}},
            {"type": "assistant", "message": {"stop_reason": "stop_sequence",
                "content": [{"type": "text",
                    "text": "API Error: Connection closed mid-response."}]}},
            {"type": "user", "message": {"content": "continue"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Answer two"}]}},
        ]

    def test_api_error_orphan_with_progress_does_not_rewind(self, tmp_path):
        """The #149 byte pattern: end_turn boundary, then an assistant 'API Error' record
        before the next user turn. The orphan flag fires (false positive) but the parse
        still advances the safe boundary — so no rewind, and the tail is usable."""
        lines = self._api_error_lines()
        path = _write_jsonl(tmp_path, lines[:2])
        boundary = parse_transcript(path).safe_end_offset
        assert boundary == path.stat().st_size  # cursor parks exactly on the end_turn boundary
        with open(path, "a") as f:
            for line in lines[2:]:
                f.write(json.dumps(line) + "\n")

        result = parse_transcript(path, start_offset=boundary)
        assert result.leading_orphan is True          # the false positive is still flagged
        assert result.safe_end_offset > boundary      # but the parse made forward progress
        assert should_rewind(result, boundary) is False
        assert result.safe_user_turn_count == 1
        assert "API Error" not in result.safe_conversation  # orphan fragment dropped

    def test_no_progress_orphan_rewinds(self, tmp_path):
        """A genuine legacy cursor parked mid-response: orphan AND no forward progress."""
        lines = [
            {"type": "user", "message": {"content": "Prompt one"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "text", "text": "First part"}]}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Second part"}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        raw = path.read_bytes().splitlines(keepends=True)
        mid = len(raw[0]) + len(raw[1])  # cursor saved mid-response by an older version

        result = parse_transcript(path, start_offset=mid)
        assert result.leading_orphan is True
        assert result.safe_end_offset == mid          # nothing closed: no progress
        assert should_rewind(result, mid) is True

    def test_no_orphan_never_rewinds_even_without_progress(self, tmp_path):
        """No-progress alone (in-flight tail) must not rewind — only orphan+no-progress does."""
        path = _write_jsonl(tmp_path, [{"type": "user", "message": {"content": "hi"}}])
        result = parse_transcript(path, start_offset=0)
        assert result.leading_orphan is False
        assert result.safe_end_offset == 0
        assert should_rewind(result, 0) is False

    def test_large_orphan_with_progress_does_not_rewind(self, tmp_path):
        """ADR-0003 large-orphan variant: a giant orphan fragment before the first user turn
        still yields progress (boundary reaches the closed turn after it) — no rewind."""
        big = "x" * 50_000
        prefix = self._api_error_lines()[:2]
        tail = [
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "text", "text": big}]}}
            for _ in range(3)
        ] + self._api_error_lines()[3:]  # user("continue") + assistant(end_turn)
        path = _write_jsonl(tmp_path, prefix)
        boundary = parse_transcript(path).safe_end_offset
        with open(path, "a") as f:
            for line in tail:
                f.write(json.dumps(line) + "\n")

        result = parse_transcript(path, start_offset=boundary)
        assert result.leading_orphan is True
        assert should_rewind(result, boundary) is False

    def test_legacy_orphan_with_later_turns_drops_fragment(self, tmp_path):
        """ADR-0003 accepted-loss pinning (council R1, Cursor+Codex): a GENUINE legacy
        cursor mid-response whose file also contains later closed turns. The fragment tail
        is dropped (its head was already ingested before the cursor; the loss is bounded
        and one-time) and the cursor advances — deliberately NO rewind. If this assertion
        ever needs to change, re-open ADR-0003 first."""
        lines = [
            {"type": "user", "message": {"content": "Prompt one"}},
            {"type": "assistant", "message": {"stop_reason": "tool_use",
                "content": [{"type": "text", "text": "First part"}]}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Second part"}]}},
            {"type": "user", "message": {"content": "Prompt two"}},
            {"type": "assistant", "message": {"stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Answer two"}]}},
        ]
        path = _write_jsonl(tmp_path, lines)
        raw = path.read_bytes().splitlines(keepends=True)
        mid = len(raw[0]) + len(raw[1])  # legacy cursor parked mid-response

        result = parse_transcript(path, start_offset=mid)
        assert result.leading_orphan is True
        assert result.safe_end_offset > mid                    # later turn closed → progress
        assert should_rewind(result, mid) is False             # ADR-0003: drop, don't rewind
        assert "Second part" not in result.safe_conversation   # the accepted, bounded loss
        assert "Prompt two" in result.safe_conversation        # later turn fully usable
        assert "Answer two" in result.safe_conversation
