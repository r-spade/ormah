"""Thin cached facade over pluggable embedding adapters."""

from __future__ import annotations

import logging
from functools import lru_cache

from ormah.embeddings.base import EmbeddingAdapter

logger = logging.getLogger(__name__)

# Cache keyed by id(settings) is not reliable; use a module-level dict instead.
_adapter_cache: dict[int, EmbeddingAdapter] = {}


def get_encoder(settings=None) -> EmbeddingAdapter:
    """Get or create a cached embedding adapter.

    Args:
        settings: Application Settings object. When ``None``, uses the
            global default settings (backwards compatibility).

    Returns:
        A cached :class:`EmbeddingAdapter` instance.
    """
    if settings is None:
        from ormah.config import settings as default_settings

        settings = default_settings

    key = id(settings)
    if key not in _adapter_cache:
        from ormah.embeddings import get_adapter

        _adapter_cache[key] = get_adapter(settings)
    return _adapter_cache[key]
