"""Ollama embedding adapter — calls /api/embed endpoint."""

from __future__ import annotations

import logging

import numpy as np

from ormah.embeddings.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """Produces embeddings via a local Ollama instance."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import httpx

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            embeddings = resp.json()["embeddings"]
            all_embeddings.extend(embeddings)

        vecs = np.array(all_embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim
