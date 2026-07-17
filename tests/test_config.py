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


def test_llm_provider_defaults_to_none():
    s = _settings()
    assert s.llm_provider == "none"


def test_invalid_provider():
    with pytest.raises(ValidationError, match="llm_provider must be one of"):
        _settings(llm_provider="gpt4all")


def test_valid_llm_api_key_env_var():
    s = _settings(llm_api_key_env_var="ANTHROPIC_API_KEY")
    assert s.llm_api_key_env_var == "ANTHROPIC_API_KEY"


def test_invalid_llm_api_key_env_var():
    with pytest.raises(ValidationError, match="llm_api_key_env_var must be one of"):
        _settings(llm_api_key_env_var="AWS_SECRET_ACCESS_KEY")


# --- LLM timeout ---

def test_timeout_zero():
    with pytest.raises(ValidationError, match="llm_timeout_seconds must be >= 1"):
        _settings(llm_timeout_seconds=0)


def test_llm_num_predict_default():
    s = _settings()
    assert s.llm_num_predict == 4096


def test_llm_num_predict_env(monkeypatch):
    monkeypatch.setenv("ORMAH_LLM_NUM_PREDICT", "1024")
    s = Settings(memory_dir="/tmp/ormah_test")
    assert s.llm_num_predict == 1024


def test_llm_num_predict_zero():
    with pytest.raises(ValidationError, match="llm_num_predict must be >= 1"):
        _settings(llm_num_predict=0)


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


def test_backup_defaults():
    s = _settings()
    assert s.backup_enabled is True
    assert s.backup_interval_hours == 24
    assert s.backup_retention_count == 10


def test_backup_interval_zero():
    with pytest.raises(ValidationError, match="backup_interval_hours must be >= 1"):
        _settings(backup_interval_hours=0)


def test_cloud_backup_interval_zero():
    with pytest.raises(ValidationError, match="cloud_backup_interval_hours must be >= 1"):
        _settings(cloud_backup_interval_hours=0)


def test_backup_retention_zero():
    with pytest.raises(ValidationError, match="backup_retention_count must be >= 1"):
        _settings(backup_retention_count=0)


def test_whisper_log_cleanup_defaults():
    s = _settings()
    assert s.whisper_log_rejected_retention_days == 30
    assert s.whisper_log_cleanup_interval_hours == 24
    assert s.whisper_log_cleanup_batch_size == 1000


@pytest.mark.parametrize(
    "field",
    [
        "whisper_log_rejected_retention_days",
        "whisper_log_cleanup_interval_hours",
        "whisper_log_cleanup_batch_size",
    ],
)
def test_whisper_log_cleanup_settings_must_be_positive(field):
    with pytest.raises(ValidationError, match="whisper log cleanup settings must be >= 1"):
        _settings(**{field: 0})


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


# --- Consolidation interval ---

def test_consolidation_interval_zero():
    with pytest.raises(ValidationError, match="interval must be >= 1"):
        _settings(consolidation_interval_minutes=0)


# --- Affinity and exploration defaults ---

def test_affinity_defaults():
    s = _settings()
    assert s.affinity_similarity_threshold == 0.70
    assert s.affinity_half_life_days == 30.0
    assert s.affinity_max_boost == 0.15
    assert s.affinity_implicit_weight == 0.8
    assert s.whisper_exploration_enabled is True
    assert s.feedback_llm_judge_enabled is False
    assert s.feedback_llm_judge_min_confidence == 0.75


def test_feedback_llm_judge_min_confidence_range():
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(feedback_llm_judge_min_confidence=1.5)


# --- Consolidation limits (#89) ---

def test_consolidation_max_clusters_negative():
    with pytest.raises(ValidationError, match="consolidation_max_clusters_per_run must be >= 0"):
        _settings(consolidation_max_clusters_per_run=-1)


def test_consolidation_min_cluster_size_below_two():
    with pytest.raises(ValidationError, match="consolidation_min_cluster_size must be >= 2"):
        _settings(consolidation_min_cluster_size=1)


def test_consolidation_threshold_out_of_range():
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(consolidation_cluster_threshold=1.5)
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(consolidation_cluster_threshold=-0.1)


def test_consolidation_threshold_non_finite():
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(consolidation_cluster_threshold=float("nan"))
    with pytest.raises(ValidationError, match="threshold must be 0"):
        _settings(consolidation_cluster_threshold=float("inf"))


def test_consolidation_max_nodes_zero_rejected():
    # The destructive misconfig Codex flagged: max_nodes=0 slips past the
    # runtime guard and emits single-node clusters. Reject it at construction.
    with pytest.raises(ValidationError, match="consolidation_max_cluster_nodes"):
        _settings(consolidation_max_cluster_nodes=0)


def test_consolidation_inverted_bounds_rejected():
    with pytest.raises(ValidationError, match="consolidation_max_cluster_nodes"):
        _settings(consolidation_min_cluster_size=3, consolidation_max_cluster_nodes=2)


# --- Embedding backfill / vector-store reconciliation (#32) ---

def test_embedding_backfill_settings_defaults():
    s = _settings()
    assert s.embedding_backfill_interval_minutes == 60
    assert s.embedding_index_max_retries == 2
    assert s.embedding_index_retry_backoff_seconds == 0.5


def test_embedding_backfill_interval_rejects_zero():
    with pytest.raises(ValidationError):
        _settings(embedding_backfill_interval_minutes=0)


def test_embedding_index_max_retries_rejects_negative():
    with pytest.raises(ValidationError):
        _settings(embedding_index_max_retries=-1)


def test_embedding_index_retry_backoff_rejects_negative():
    with pytest.raises(ValidationError):
        _settings(embedding_index_retry_backoff_seconds=-0.1)
