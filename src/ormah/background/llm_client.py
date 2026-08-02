"""Shared LLM facade for background tasks.

All callers import ``llm_generate`` from here — the function signature is
unchanged.  Internally we delegate to the adapter returned by
``get_adapter(settings)``.
"""

from __future__ import annotations

import json
import logging
import re

from ormah.background.llm import LLMAdapter, get_adapter

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json(raw: str) -> str:
    """Extract a JSON document from an LLM response.

    Thinking-capable models (e.g. qwen3.5) wrap their output in markdown
    ``` fences or surround it with prose even when asked for JSON mode, which
    makes a naive ``json.loads(raw)`` fail. This recovers the embedded JSON so
    callers parse it instead of discarding a valid response.
    """
    stripped = raw.strip()

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    for match in _FENCE_RE.finditer(raw):
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(raw[start:])
            return raw[start : start + end]
        except json.JSONDecodeError:
            continue

    return stripped

_cached_adapter: LLMAdapter | None = None
_adapter_initialised: bool = False


def reset_adapter() -> None:
    """Clear the cached adapter (useful for test isolation)."""
    global _cached_adapter, _adapter_initialised
    _cached_adapter = None
    _adapter_initialised = False


def _get_or_create_adapter(settings) -> LLMAdapter | None:
    global _cached_adapter, _adapter_initialised
    if not _adapter_initialised:
        _cached_adapter = get_adapter(settings)
        _adapter_initialised = True
    return _cached_adapter


def llm_generate(
    settings,
    prompt: str,
    json_mode: bool = True,
    *,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout_hint_seconds: float | None = None,
) -> str | None:
    """Call configured LLM. Returns raw response text, or None on failure."""
    adapter = _get_or_create_adapter(settings)
    if adapter is None:
        return None
    return adapter.generate(
        prompt,
        json_mode=json_mode,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_hint_seconds=timeout_hint_seconds,
    )
