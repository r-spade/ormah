"""Normalize supported agent JSONL transcripts into conversation text and turns."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptTurn:
    """One normalized text-bearing conversation turn."""

    role: str
    text: str


@dataclass
class TranscriptResult:
    """Result of parsing a supported agent transcript."""

    conversation: str  # "User: ...\n\nAssistant: ...\n\n..."
    user_turn_count: int  # User messages with actual text (not tool_result)
    total_chars: int  # Original JSONL size
    cleaned_chars: int  # After stripping
    session_id: str  # From filename stem (UUID)
    end_offset: int = 0  # Byte position after last line read
    # "safe" = the closed boundary: content proven complete, so the cursor may advance
    # past it without ever splitting a response. A response closes at an assistant record
    # with a terminal stop_reason (Claude Code), a Codex task_complete event, or the start
    # of the next user turn (universal — covers interrupts and any signal-less source).
    safe_end_offset: int = 0       # byte position after the last closed response
    safe_conversation: str = ""    # conversation text up to safe_end_offset only
    safe_user_turn_count: int = 0  # user turns within the safe boundary
    safe_turns: list[TranscriptTurn] = field(default_factory=list)
    # True when an incremental slice (start_offset > 0) began with text-bearing assistant
    # content before any user turn — i.e. a cursor an older version left mid-response. The
    # orphan is dropped here; the caller should re-parse from offset 0 to recover the
    # prompt and re-pair the response.
    leading_orphan: bool = False
    turns: list[TranscriptTurn] = field(default_factory=list)
    source: str = "agent_jsonl"
    # True when max_bytes stopped the parse before a turn that would have overshot the
    # byte budget — more closed content remains past safe_end_offset for the caller to
    # drain in a follow-up parse_transcript(start_offset=safe_end_offset, ...) call.
    capped: bool = False


def _extract_user_text(content) -> str | None:
    """Extract text from a user message content field.

    Returns the text if the message contains user text, or None if it's
    only tool_result blocks (which should be skipped entirely).
    """
    if isinstance(content, str):
        return content.strip() or None

    if isinstance(content, list):
        texts = []
        has_text = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("text", "input_text"):
                has_text = True
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        # If there were no text blocks at all (only tool_result), skip
        if not has_text:
            return None
        return "\n".join(texts) if texts else None

    return None


def _extract_assistant_text(content) -> str | None:
    """Extract only text blocks from an assistant message content list."""
    if not isinstance(content, list):
        return None

    texts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "output_text"):
            text = block.get("text", "").strip()
            if text:
                texts.append(text)

    return "\n".join(texts) if texts else None


def _coerce_entry(entry: dict) -> tuple[str | None, object | None]:
    """Normalize Claude Code and Codex transcript records to (role, content)."""
    entry_type = entry.get("type")
    if entry_type in ("user", "assistant"):
        message = entry.get("message")
        if not isinstance(message, dict):
            return None, None
        return entry_type, message.get("content")

    if entry_type == "response_item":
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            return None, None
        if payload.get("type") != "message":
            return None, None
        role = payload.get("role")
        if role not in ("user", "assistant"):
            return None, None
        return role, payload.get("content")

    return None, None


def _source_for_entry(entry: dict) -> str | None:
    """Return the agent/source label implied by a supported transcript record."""
    entry_type = entry.get("type")
    if entry_type in ("user", "assistant"):
        return "claude_code"
    if entry_type == "response_item":
        return "codex"
    return None


# A terminal stop_reason means the response is finished and no further record belongs
# to it (the cursor may advance past it). tool_use / pause_turn are non-terminal: a
# continuation follows, so advancing there would strand a later record from its prompt.
_TERMINAL_STOP_REASONS = ("end_turn", "stop_sequence", "max_tokens", "refusal")


def _assistant_is_terminal(entry: dict) -> bool:
    """True when an assistant record ends its response (reliable for Claude Code)."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return False
    return message.get("stop_reason") in _TERMINAL_STOP_REASONS


def _is_turn_complete_event(entry: dict) -> bool:
    """True for a Codex ``task_complete`` event_msg — a reliable end-of-turn signal.

    Codex (response_item) records have no stop_reason, but each turn ends with a
    ``task_complete`` event. Treating it as a closure lets the cursor advance past a
    finished Codex turn without waiting for the next user turn — and without splitting a
    multi-record Codex response.
    """
    if entry.get("type") != "event_msg":
        return False
    payload = entry.get("payload")
    return isinstance(payload, dict) and payload.get("type") == "task_complete"


def _is_bootstrap_user_text(text: str) -> bool:
    """Return True when text is client/bootstrap context, not a real user turn."""
    stripped = text.strip()
    return (
        stripped.startswith("# AGENTS.md instructions for ")
        or stripped.startswith("<environment_context>")
        or stripped.startswith("<turn_aborted>")
    )


def extract_user_prompts(path: Path, start_offset: int = 0) -> list[str]:
    """Extract only user text turns from a supported agent JSONL transcript."""
    path = Path(path)
    prompts: list[str] = []

    with open(path) as f:
        if start_offset > 0:
            f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            entry_type, content = _coerce_entry(entry)
            if entry_type != "user":
                continue
            if content is None:
                continue

            text = _extract_user_text(content)
            if text and not _is_bootstrap_user_text(text):
                prompts.append(text)

    return prompts


def _conversation_from_turns(turns: list[TranscriptTurn]) -> str:
    return "\n\n".join(
        f"{turn.role.title()}: {turn.text}"
        for turn in turns
    )


def parse_transcript(
    path: Path, start_offset: int = 0, max_bytes: int | None = None
) -> TranscriptResult:
    """Parse a supported JSONL transcript into cleaned conversation text.

    Reads line by line, extracting only user text and assistant text blocks.
    Skips tool_use, thinking, tool_result, progress, system, and other
    non-conversation content.

    When *start_offset* > 0, seeks to that byte position before reading.
    The caller must ensure the offset falls on a line boundary (e.g. from
    a previous call's ``end_offset``).

    When *max_bytes* is set, parsing stops BEFORE committing a turn that would push the
    closed slice (``safe_end_offset - start_offset``) past that budget — so a multi-turn
    slice never exceeds max_bytes. The caller re-parses from the new ``safe_end_offset``
    to drain the rest. A single turn larger than max_bytes is committed anyway (there is
    no smaller slice to make progress with).
    """
    path = Path(path)
    total_chars = path.stat().st_size

    turns: list[TranscriptTurn] = []
    user_turn_count = 0
    source = "agent_jsonl"

    # safe_* is the closed boundary: content proven complete, so the cursor may advance
    # past it without ever splitting a response. A response closes at a terminal
    # stop_reason (Claude Code), a Codex task_complete event, or the start of the next
    # user turn. A response with no completion signal yet stays open (held back).
    _safe_end = start_offset
    _safe_len = 0
    _safe_users = 0
    _seen_assistant_text = False  # a text-bearing assistant appeared in the current block
    _leading_orphan = False  # dropped assistant content before any user record (bad cursor)
    _saw_user_record = False  # any user-role record seen (incl. a text-less tool_result)
    _capped = False  # max_bytes stopped the parse before an overshooting turn

    def _would_overshoot(new_safe_end: int) -> bool:
        # Only refuse a candidate boundary once something is already committed — a first
        # turn alone can't be shrunk further, so it's always allowed through.
        return (
            max_bytes is not None
            and _safe_len > 0
            and (new_safe_end - start_offset) > max_bytes
        )

    with open(path) as f:
        if start_offset > 0:
            f.seek(start_offset)
        while True:
            pos_before = f.tell()  # byte offset at the start of this line
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            try:
                entry = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                continue

            entry_source = _source_for_entry(entry)
            if source == "agent_jsonl" and entry_source is not None:
                source = entry_source  # first-wins (preserved from original)

            if _is_turn_complete_event(entry):
                # Codex end-of-turn: the open response is complete, advance the closed
                # boundary past it (so a multi-record Codex turn is never split).
                if _seen_assistant_text:
                    if _would_overshoot(f.tell()):
                        _capped = True
                        break
                    _safe_end = f.tell()
                    _safe_len = len(turns)
                    _safe_users = user_turn_count
                    _seen_assistant_text = False
                continue

            entry_type, content = _coerce_entry(entry)
            if entry_type not in ("user", "assistant"):
                continue
            if content is None:
                continue

            if entry_type == "user":
                # Any user-role record — including a tool_result with no text — means this
                # slice sits inside a proper turn, not stranded mid-response. Track it so a
                # following assistant is not misread as a leading orphan: a tool-use chain
                # runs assistant -> user(tool_result, empty text) -> assistant, and those
                # text-less users don't advance user_turn_count.
                _saw_user_record = True
                text = _extract_user_text(content)
                if text and not _is_bootstrap_user_text(text):
                    # A new user turn definitively closes any still-open response (an
                    # interrupt, or a Codex turn with no stop_reason): the safe boundary
                    # advances to the start of this user line. This never splits a
                    # response — the whole prior block is on the closed side.
                    if _seen_assistant_text:
                        if _would_overshoot(pos_before):
                            _capped = True
                            break
                        _safe_end = pos_before  # boundary = start of this user line
                        _safe_len = len(turns)
                        _safe_users = user_turn_count
                        _seen_assistant_text = False
                    turns.append(TranscriptTurn(role="user", text=text))
                    user_turn_count += 1

            elif entry_type == "assistant":
                text = _extract_assistant_text(content)
                # Drop assistant content that precedes ANY user record in this slice: its
                # prompt lies before start_offset (a cursor left mid-response by an older
                # version), so committing it would emit an orphan fragment without its
                # prompt. Gate on _saw_user_record (not user_turn_count): a text-less
                # tool_result user still marks a proper turn boundary, so a tool-use chain
                # is not a false orphan. The caller re-parses from 0 (see leading_orphan)
                # to recover a genuinely dropped fragment with its prompt.
                if text and not _saw_user_record and start_offset > 0:
                    _leading_orphan = True
                if text and user_turn_count > 0:
                    if _assistant_is_terminal(entry) and _would_overshoot(f.tell()):
                        _capped = True
                        break
                    turns.append(TranscriptTurn(role="assistant", text=text))
                    if _assistant_is_terminal(entry):
                        # Reliable completion signal (Claude Code): the response is done,
                        # so the safe boundary may advance past it even with no following
                        # user turn. A non-terminal record (tool_use / streaming) leaves
                        # the block open so a later record of the same response is never
                        # stranded from its prompt.
                        _safe_end = f.tell()
                        _safe_len = len(turns)
                        _safe_users = user_turn_count
                        _seen_assistant_text = False
                    else:
                        _seen_assistant_text = True

        end_offset = f.tell()

    conversation = _conversation_from_turns(turns)
    safe_turns = turns[:_safe_len]
    return TranscriptResult(
        conversation=conversation,
        user_turn_count=user_turn_count,
        total_chars=total_chars,
        cleaned_chars=len(conversation),
        session_id=path.stem,
        end_offset=end_offset,
        safe_end_offset=_safe_end,
        safe_conversation=_conversation_from_turns(safe_turns),
        safe_user_turn_count=_safe_users,
        safe_turns=safe_turns,
        leading_orphan=_leading_orphan,
        turns=turns,
        source=source,
        capped=_capped,
    )


def should_rewind(result: TranscriptResult, start_offset: int) -> bool:
    """Gate the leading-orphan recovery on forward progress (ADR-0003, bug #149).

    Rewind (re-parse from offset 0) only when the flagged parse made no forward
    progress — the orphan consumed the whole slice, i.e. a genuine legacy
    mid-response cursor. When the safe boundary still advanced past the cursor,
    the orphan is a false positive (e.g. an "API Error" assistant record right
    after a terminal stop_reason): the fragment is dropped and the cursor moves
    on. Rewinding there re-ingests the whole file on every tick forever, because
    the trigger is a permanent property of the file's bytes.
    """
    return result.leading_orphan and result.safe_end_offset <= start_offset
