"""Abstract base class for embedding adapters."""

from __future__ import annotations

import abc

import numpy as np


class EmbeddingAdapter(abc.ABC):
    """Interface that all embedding backends must implement."""

    @abc.abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a normalized vector."""

    @abc.abstractmethod
    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of texts to normalized vectors."""

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a search query. Override to add model-specific query prefixes."""
        return self.encode(text)

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
