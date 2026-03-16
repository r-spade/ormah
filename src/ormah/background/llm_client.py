"""Shared LLM facade for background tasks.

All callers import ``llm_generate`` from here — the function signature is
unchanged.  Internally we delegate to the adapter returned by
``get_adapter(settings)``.
"""

from __future__ import annotations

import logging

from ormah.background.llm import LLMAdapter, get_adapter

logger = logging.getLogger(__name__)

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


def llm_generate(settings, prompt: str, json_mode: bool = True) -> str | None:
    """Call configured LLM. Returns raw response text, or None on failure."""
    adapter = _get_or_create_adapter(settings)
    if adapter is None:
        return None
    return adapter.generate(prompt, json_mode=json_mode)
