"""Parse Claude Code JSONL session transcripts into clean conversation text."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    """Result of parsing a Claude Code JSONL transcript."""

    conversation: str  # "User: ...\n\nAssistant: ...\n\n..."
    user_turn_count: int  # User messages with actual text (not tool_result)
    total_chars: int  # Original JSONL size
    cleaned_chars: int  # After stripping
    session_id: str  # From filename stem (UUID)
    end_offset: int = 0  # Byte position after last line read


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
            if block.get("type") == "text":
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
        if block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                texts.append(text)

    return "\n".join(texts) if texts else None


def parse_transcript(path: Path, start_offset: int = 0) -> TranscriptResult:
    """Parse a Claude Code JSONL transcript into cleaned conversation text.

    Reads line by line, extracting only user text and assistant text blocks.
    Skips tool_use, thinking, tool_result, progress, system, and other
    non-conversation content.

    When *start_offset* > 0, seeks to that byte position before reading.
    The caller must ensure the offset falls on a line boundary (e.g. from
    a previous call's ``end_offset``).
    """
    path = Path(path)
    total_chars = path.stat().st_size

    turns: list[str] = []
    user_turn_count = 0

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

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            message = entry.get("message")
            if not isinstance(message, dict):
                continue

            content = message.get("content")
            if content is None:
                continue

            if entry_type == "user":
                text = _extract_user_text(content)
                if text:
                    turns.append(f"User: {text}")
                    user_turn_count += 1

            elif entry_type == "assistant":
                text = _extract_assistant_text(content)
                if text:
                    turns.append(f"Assistant: {text}")

        end_offset = f.tell()

    conversation = "\n\n".join(turns)
    return TranscriptResult(
        conversation=conversation,
        user_turn_count=user_turn_count,
        total_chars=total_chars,
        cleaned_chars=len(conversation),
        session_id=path.stem,
        end_offset=end_offset,
    )
