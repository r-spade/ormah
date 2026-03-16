"""Tests for embedding adapters and the provider registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ormah.config import Settings


# ---------------------------------------------------------------------------
# LocalAdapter
# ---------------------------------------------------------------------------


class TestLocalAdapter:
    def test_encode_produces_correct_dim(self):
        from ormah.embeddings.local_adapter import LocalAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768, dtype=np.float32)

        adapter = LocalAdapter(model_name="test-model")
        adapter._model = mock_model

        vec = adapter.encode("hello world")
        assert vec.shape == (768,)
        mock_model.encode.assert_called_once_with("hello world", normalize_embeddings=True)

    def test_dim_auto_detected(self):
        from ormah.embeddings.local_adapter import LocalAdapter

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(768, dtype=np.float32)

        adapter = LocalAdapter(model_name="test-model")
        adapter._model = mock_model

        # dim should be auto-detected after first encode
        adapter.encode("probe")
        assert adapter.dim == 768


# ---------------------------------------------------------------------------
# OllamaEmbeddingAdapter
# ---------------------------------------------------------------------------


class TestOllamaAdapter:
    def test_encode_single(self):
        from ormah.embeddings.ollama_adapter import OllamaEmbeddingAdapter

        adapter = OllamaEmbeddingAdapter(model="nomic-embed-text", dim=384)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[1.0, 0.0, 0.0] + [0.0] * 381]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            vec = adapter.encode("hello")
            assert vec.shape == (384,)
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/api/embed" in call_args[0][0]
            assert call_args[1]["json"]["model"] == "nomic-embed-text"

    def test_encode_batch(self):
        from ormah.embeddings.ollama_adapter import OllamaEmbeddingAdapter

        adapter = OllamaEmbeddingAdapter(model="nomic-embed-text", dim=4)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            vecs = adapter.encode_batch(["hello", "world"])
            assert vecs.shape == (2, 4)
            # Check normalization
            norms = np.linalg.norm(vecs, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_dim_property(self):
        from ormah.embeddings.ollama_adapter import OllamaEmbeddingAdapter

        adapter = OllamaEmbeddingAdapter(dim=512)
        assert adapter.dim == 512


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingAdapter
# ---------------------------------------------------------------------------


class TestLiteLLMAdapter:
    def test_encode(self):
        from ormah.embeddings.litellm_adapter import LiteLLMEmbeddingAdapter

        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small", dim=4)

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [1.0, 0.0, 0.0, 0.0]}]

        with patch.dict("sys.modules", {"litellm": MagicMock()}):
            import sys

            mock_litellm = sys.modules["litellm"]
            mock_litellm.embedding.return_value = mock_response

            vec = adapter.encode("hello")
            assert vec.shape == (4,)
            mock_litellm.embedding.assert_called_once()

    def test_dim_property(self):
        from ormah.embeddings.litellm_adapter import LiteLLMEmbeddingAdapter

        adapter = LiteLLMEmbeddingAdapter(dim=1536)
        assert adapter.dim == 1536


# ---------------------------------------------------------------------------
# Registry (get_adapter)
# ---------------------------------------------------------------------------


class TestGetAdapter:
    def test_routes_to_local(self, tmp_path):
        from ormah.embeddings import get_adapter
        from ormah.embeddings.local_adapter import LocalAdapter

        settings = Settings(memory_dir=tmp_path, embedding_provider="local")
        adapter = get_adapter(settings)
        assert isinstance(adapter, LocalAdapter)

    def test_routes_to_ollama(self, tmp_path):
        from ormah.embeddings import get_adapter
        from ormah.embeddings.ollama_adapter import OllamaEmbeddingAdapter

        settings = Settings(
            memory_dir=tmp_path,
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
        )
        adapter = get_adapter(settings)
        assert isinstance(adapter, OllamaEmbeddingAdapter)

    def test_routes_to_litellm(self, tmp_path):
        from ormah.embeddings import get_adapter
        from ormah.embeddings.litellm_adapter import LiteLLMEmbeddingAdapter

        settings = Settings(
            memory_dir=tmp_path,
            embedding_provider="litellm",
            embedding_model="text-embedding-3-small",
        )
        adapter = get_adapter(settings)
        assert isinstance(adapter, LiteLLMEmbeddingAdapter)

    def test_invalid_provider_raises(self, tmp_path):
        from ormah.embeddings import get_adapter

        settings = Settings.__new__(Settings)
        object.__setattr__(settings, "embedding_provider", "invalid")
        with pytest.raises(NotImplementedError):
            get_adapter(settings)


# ---------------------------------------------------------------------------
# Facade (get_encoder caching)
# ---------------------------------------------------------------------------


class TestGetEncoderCaching:
    def test_caches_adapter(self, tmp_path):
        from ormah.embeddings.encoder import get_encoder, _adapter_cache

        settings = Settings(memory_dir=tmp_path, embedding_provider="local")
        # Clear cache
        _adapter_cache.clear()

        enc1 = get_encoder(settings)
        enc2 = get_encoder(settings)
        assert enc1 is enc2


# ---------------------------------------------------------------------------
# Dimension mismatch detection
# ---------------------------------------------------------------------------


class TestDimensionMismatch:
    def test_mismatch_drops_and_recreates(self, tmp_path):
        """Changing dim should drop and recreate the vec table."""
        from ormah.index.db import Database

        db = Database(tmp_path / "index.db")
        db.init_schema()

        # Create with dim=4
        try:
            db.init_vec_table(dim=4)
        except Exception:
            pytest.skip("sqlite-vec not available")

        # Insert a vector
        import struct

        vec_bytes = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        db.conn.execute(
            "INSERT INTO node_vectors (id, embedding) VALUES (?, ?)",
            ("test-node", vec_bytes),
        )
        db.conn.commit()

        # Now init with dim=8 — should drop and recreate
        db.init_vec_table(dim=8)

        # Old data should be gone
        row = db.conn.execute(
            "SELECT id FROM node_vectors WHERE id = 'test-node'"
        ).fetchone()
        assert row is None

        db.close()
