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
    turns: list[TranscriptTurn] = field(default_factory=list)
    source: str = "agent_jsonl"


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


def parse_transcript(path: Path, start_offset: int = 0) -> TranscriptResult:
    """Parse a supported JSONL transcript into cleaned conversation text.

    Reads line by line, extracting only user text and assistant text blocks.
    Skips tool_use, thinking, tool_result, progress, system, and other
    non-conversation content.

    When *start_offset* > 0, seeks to that byte position before reading.
    The caller must ensure the offset falls on a line boundary (e.g. from
    a previous call's ``end_offset``).
    """
    path = Path(path)
    total_chars = path.stat().st_size

    turns: list[TranscriptTurn] = []
    user_turn_count = 0
    source = "agent_jsonl"

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

            entry_source = _source_for_entry(entry)
            if source == "agent_jsonl" and entry_source is not None:
                source = entry_source

            entry_type, content = _coerce_entry(entry)
            if entry_type not in ("user", "assistant"):
                continue
            if content is None:
                continue

            if entry_type == "user":
                text = _extract_user_text(content)
                if text and not _is_bootstrap_user_text(text):
                    turns.append(TranscriptTurn(role="user", text=text))
                    user_turn_count += 1

            elif entry_type == "assistant":
                text = _extract_assistant_text(content)
                if text:
                    turns.append(TranscriptTurn(role="assistant", text=text))

        end_offset = f.tell()

    conversation = _conversation_from_turns(turns)
    return TranscriptResult(
        conversation=conversation,
        user_turn_count=user_turn_count,
        total_chars=total_chars,
        cleaned_chars=len(conversation),
        session_id=path.stem,
        end_offset=end_offset,
        turns=turns,
        source=source,
    )
