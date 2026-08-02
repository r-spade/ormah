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

_cached_ingest_adapter: LLMAdapter | None = None
_ingest_adapter_initialised: bool = False


def reset_adapter() -> None:
    """Clear the cached adapters (useful for test isolation)."""
    global _cached_adapter, _adapter_initialised, _cached_ingest_adapter, _ingest_adapter_initialised
    _cached_adapter = None
    _adapter_initialised = False
    _cached_ingest_adapter = None
    _ingest_adapter_initialised = False


def _get_or_create_adapter(settings) -> LLMAdapter | None:
    global _cached_adapter, _adapter_initialised
    if not _adapter_initialised:
        _cached_adapter = get_adapter(settings)
        _adapter_initialised = True
    return _cached_adapter


def _resolve_ingest_provider(settings) -> str | None:
    return getattr(settings, "ingest_llm_provider", None) or getattr(settings, "llm_provider", None)


def _resolve_ingest_model(settings) -> str | None:
    return getattr(settings, "ingest_llm_model", None) or getattr(settings, "llm_model", None)


def _get_or_create_ingest_adapter(settings) -> LLMAdapter | None:
    global _cached_ingest_adapter, _ingest_adapter_initialised
    if not _ingest_adapter_initialised:
        _cached_ingest_adapter = get_adapter(
            settings,
            provider=_resolve_ingest_provider(settings),
            model=_resolve_ingest_model(settings),
        )
        _ingest_adapter_initialised = True
    return _cached_ingest_adapter


def ingest_llm_generate(settings, prompt: str, json_mode: bool = True, **kwargs) -> str | None:
    """Generate for server-side extraction, using ingest_llm_provider/model (not the
    maintenance-path llm_provider/llm_model)."""
    adapter = _get_or_create_ingest_adapter(settings)
    if adapter is None:
        return None
    return adapter.generate(prompt, json_mode=json_mode, **kwargs)


def ingest_provider_configured(settings) -> bool:
    """True when a server-side extraction adapter is available (ingest provider != none).

    Lets callers tell "no provider" (a global, temporary state) apart from "the call failed"
    (a timeout/error while a provider IS configured) — both of which surface as a None from
    ``ingest_llm_generate``."""
    return _get_or_create_ingest_adapter(settings) is not None


def llm_generate(
    settings,
    prompt: str,
    json_mode: bool = True,
    *,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
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
    )
