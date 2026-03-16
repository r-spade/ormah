"""Local fastembed embedding adapter (CPU-only, no PyTorch/CUDA required)."""

from __future__ import annotations

import logging

import numpy as np

from ormah.embeddings.base import EmbeddingAdapter

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


class LocalAdapter(EmbeddingAdapter):
    """Wraps fastembed with lazy loading and caching."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError(
                    "fastembed is required for local embeddings. "
                    "Install ormah normally — it should be included."
                )

            if self.model_name in _model_cache:
                self._model = _model_cache[self.model_name]
            else:
                logger.info("Loading embedding model (~420MB, first time only)...")
                self._model = TextEmbedding(self.model_name)
                _model_cache[self.model_name] = self._model
                logger.info("Embedding model ready.")
        return self._model

    def encode(self, text: str) -> np.ndarray:
        vec = next(iter(self.model.embed([text])))
        if self._dim is None:
            self._dim = vec.shape[0]
        return vec

    def encode_query(self, text: str) -> np.ndarray:
        # fastembed's query_embed handles model-specific query prefixes automatically
        embed_fn = getattr(self.model, "query_embed", self.model.embed)
        vec = next(iter(embed_fn([text])))
        if self._dim is None:
            self._dim = vec.shape[0]
        return vec

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts, batch_size=batch_size)))
        if self._dim is None and len(vecs) > 0:
            self._dim = vecs.shape[1]
        return vecs

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        # Probe the model to detect dimension
        self.encode("")
        return self._dim  # type: ignore[return-value]
