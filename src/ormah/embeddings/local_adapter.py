"""Local fastembed embedding adapter (CPU-only, no PyTorch/CUDA required)."""

from __future__ import annotations

import logging

import numpy as np

from ormah.embeddings.base import EmbeddingAdapter
from ormah.embeddings.cache import get_fastembed_cache_dir
from ormah.embeddings.runtime import MAX_BATCH_SIZE, local_inference

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


class LocalAdapter(EmbeddingAdapter):
    """Wraps fastembed with lazy loading and caching."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    @local_inference
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
                self._model = TextEmbedding(
                    self.model_name,
                    cache_dir=str(get_fastembed_cache_dir()),
                    enable_cpu_mem_arena=False,
                )
                _model_cache[self.model_name] = self._model
                logger.info("Embedding model ready.")
        return self._model

    @local_inference
    def encode(self, text: str) -> np.ndarray:
        vec = next(iter(self.model.embed([text])))
        if self._dim is None:
            self._dim = vec.shape[0]
        return vec

    @local_inference
    def encode_query(self, text: str) -> np.ndarray:
        # fastembed's query_embed handles model-specific query prefixes automatically
        model = self.model
        embed_fn = getattr(model, "query_embed", model.embed)
        vec = next(iter(embed_fn([text])))
        if self._dim is None:
            self._dim = vec.shape[0]
        return vec

    @local_inference
    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vecs = np.array(list(self.model.embed(
            texts, batch_size=min(batch_size, MAX_BATCH_SIZE),
        )))
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
