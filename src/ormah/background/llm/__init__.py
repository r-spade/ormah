"""LLM adapter package — pluggable backends for background jobs."""

from __future__ import annotations

from ormah.background.llm.base import LLMAdapter
from ormah.background.llm.normalize import normalize_conflict_type, normalize_link_type

__all__ = [
    "LLMAdapter",
    "get_adapter",
    "normalize_conflict_type",
    "normalize_link_type",
]


def get_adapter(settings) -> LLMAdapter | None:
    """Build an adapter from the application settings.

    Returns ``None`` when ``llm_provider`` is ``"none"``.
    """
    provider = settings.llm_provider
    timeout = getattr(settings, "llm_timeout_seconds", 60)

    if provider == "ollama":
        from ormah.background.llm.ollama_adapter import OllamaAdapter

        return OllamaAdapter(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=timeout,
        )

    if provider == "litellm":
        from ormah.background.llm.litellm_adapter import LiteLLMAdapter

        return LiteLLMAdapter(model=settings.llm_model, timeout=timeout)

    if provider == "none":
        return None

    raise NotImplementedError(f"LLM provider {provider!r} not implemented")
