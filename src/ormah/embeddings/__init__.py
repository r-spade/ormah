"""Embedding adapter package — pluggable backends for vector encoding."""

from __future__ import annotations

from ormah.embeddings.base import EmbeddingAdapter

__all__ = [
    "EmbeddingAdapter",
    "get_adapter",
]


def get_adapter(settings) -> EmbeddingAdapter:
    """Build an embedding adapter from the application settings."""
    provider = settings.embedding_provider

    if provider == "local":
        from ormah.embeddings.local_adapter import LocalAdapter

        return LocalAdapter(model_name=settings.embedding_model)

    if provider == "ollama":
        from ormah.embeddings.ollama_adapter import OllamaEmbeddingAdapter

        return OllamaEmbeddingAdapter(
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            dim=settings.embedding_dim,
        )

    if provider == "litellm":
        from ormah.embeddings.litellm_adapter import LiteLLMEmbeddingAdapter

        return LiteLLMEmbeddingAdapter(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )

    raise NotImplementedError(f"Embedding provider {provider!r} not implemented")
