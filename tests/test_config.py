"""Tests for config validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ormah.config import Settings


def _settings(**overrides) -> Settings:
    """Create settings with overrides, using a temp dir for memory_dir."""
    defaults = {"memory_dir": "/tmp/ormah_test"}
    defaults.update(overrides)
    return Settings(**defaults)


# --- Port ---

def test_valid_port():
    s = _settings(port=3000)
    assert s.port == 3000


def test_port_too_low():
    with pytest.raises(ValidationError, match="port must be 1"):
        _settings(port=0)


def test_port_too_high():
    with pytest.raises(ValidationError, match="port must be 1"):
        _settings(port=70000)


# --- LLM provider ---

def test_valid_providers():
    for p in ("ollama", "litellm", "none"):
        s = _settings(llm_provider=p)
        assert s.llm_provider == p


def test_invalid_provider():
    with pytest.raises(ValidationError, match="llm_provider must be one of"):
        _settings(llm_provider="gpt4all")


# --- LLM timeout ---

def test_timeout_zero():
    with pytest.raises(ValidationError, match="llm_timeout_seconds must be >= 1"):
        _settings(llm_timeout_seconds=0)


# --- Embedding dim ---

def test_embedding_dim_zero():
    with pytest.raises(ValidationError, match="embedding_dim must be >= 1"):
        _settings(embedding_dim=0)


# --- Intervals ---

def test_interval_zero():
    with pytest.raises(ValidationError, match="interval must be >= 1"):
        _settings(auto_link_interval_minutes=0)


def test_decay_hours_zero():
    with pytest.raises(ValidationError, match="decay_interval_hours must be >= 1"):
        _settings(decay_interval_hours=0)


# --- Core cap ---

def test_core_cap_zero():
    with pytest.raises(ValidationError, match="core_memory_cap must be >= 1"):
        _settings(core_memory_cap=0)


# --- Search weights ---

def test_negative_fts_weight():
    with pytest.raises(ValidationError, match="search weight must be >= 0"):
        _settings(fts_weight=-0.1)


def test_zero_fts_weight_ok():
    s = _settings(fts_weight=0.0)
    assert s.fts_weight == 0.0


# --- Thresholds ---

def test_threshold_above_one():
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(similarity_threshold=1.5)


def test_threshold_negative():
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(auto_merge_threshold=-0.1)


# --- Activation decay ---

def test_activation_decay_zero():
    with pytest.raises(ValidationError, match="activation_decay must be"):
        _settings(activation_decay=0.0)


def test_activation_decay_one_ok():
    s = _settings(activation_decay=1.0)
    assert s.activation_decay == 1.0


# --- llm_enabled property ---

def test_llm_enabled_true():
    s = _settings(llm_provider="ollama")
    assert s.llm_enabled is True


def test_llm_enabled_false():
    s = _settings(llm_provider="none")
    assert s.llm_enabled is False


# --- Importance weights ---

def test_importance_weight_valid():
    s = _settings(importance_access_weight=0.5)
    assert s.importance_access_weight == 0.5


def test_importance_weight_negative():
    with pytest.raises(ValidationError, match="importance weight must be 0"):
        _settings(importance_access_weight=-0.1)


def test_importance_weight_above_one():
    with pytest.raises(ValidationError, match="importance weight must be 0"):
        _settings(importance_edge_weight=1.5)


# --- Importance recompute interval ---

def test_importance_interval_zero():
    with pytest.raises(ValidationError, match="interval must be >= 1"):
        _settings(importance_recompute_interval_minutes=0)


# --- Context max nodes ---

def test_context_max_nodes_valid():
    s = _settings(context_max_nodes=10)
    assert s.context_max_nodes == 10


def test_context_max_nodes_zero():
    with pytest.raises(ValidationError, match="context_max_nodes must be >= 1"):
        _settings(context_max_nodes=0)


# --- Consolidation interval ---

def test_consolidation_interval_zero():
    with pytest.raises(ValidationError, match="interval must be >= 1"):
        _settings(consolidation_interval_minutes=0)
