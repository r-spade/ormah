"""LiteLLM embedding adapter — supports OpenAI, Gemini, Voyage, Mistral, etc."""

from __future__ import annotations

import logging

import numpy as np

from ormah.embeddings.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


class LiteLLMEmbeddingAdapter(EmbeddingAdapter):
    """Produces embeddings via litellm.embedding()."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        self.model = model
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        try:
            import litellm
        except ImportError:
            logger.error(
                "litellm is not installed. Install it with: pip install 'ormah[litellm]'"
            )
            raise

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = litellm.embedding(model=self.model, input=batch)
            for item in response.data:
                all_embeddings.append(item["embedding"])

        vecs = np.array(all_embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim
