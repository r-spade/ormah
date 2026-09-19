"""Environment-independent Settings overrides shared by the eval harnesses.

A bare ``Settings()`` absorbs the developer's ``ORMAH_*`` environment and
``~/.config/ormah/.env`` values, so retrieval knobs would silently vary from
machine to machine and baselines would not be comparable. These dicts pin
every retrieval-relevant knob to its production default (the values in
``src/ormah/config.py``), so the eval measures the *shipped* configuration
rather than whatever the local box happens to be tuned to.
"""
from __future__ import annotations

# Retrieval-relevant knobs pinned to their production defaults (config.py).
# Shared by both the recall and whisper eval harnesses so each one measures the
# shipped configuration regardless of local env/.env overrides.
RETRIEVAL_EVAL_SETTINGS_OVERRIDES: dict = {
    # LLM extraction is never exercised by retrieval evals.
    "llm_provider": "none",
    # Embeddings
    "embedding_provider": "local",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_dim": 768,
    # Hybrid search fusion
    "fts_weight": 0.4,
    "vector_weight": 0.6,
    "similarity_threshold": 0.4,
    "rrf_k": 60,
    "fts_only_dampening": 0.5,
    "min_result_score": 0.1,
    "rrf_min_spread_ratio": 0.05,
    # Question-query adjustments
    "question_fts_weight_scale": 0.3,
    "question_vector_weight_scale": 1.5,
    "question_similarity_blend_weight": 0.85,
    "similarity_blend_weight": 0.5,
    # Title / length scoring
    "title_match_boost": 2.0,
    "length_penalty_threshold": 300,
    # Scoring signals
    "recency_boost": 0.05,
    "recency_half_life_days": 7.0,
    "access_boost": 0.05,
    "tier_boost_core": 0.1,
    "tier_boost_working": 0.0,
    "tier_boost_archival": -0.1,
    # Space prioritization
    "space_boost_global": 1.0,
    "space_boost_other": 0.6,
    # Absolute gates
    "recall_min_relevance_score": 0.35,
    "whisper_injection_gate": 0.45,
    # Recall eval never whispers: no reranker, no involuntary storage, no
    # Claude-in-the-loop maintenance. The whisper eval re-enables the reranker.
    "whisper_reranker_enabled": False,
    "whisper_out_enabled": False,
    "claude_maintenance_enabled": False,
}


WHISPER_EVAL_SETTINGS_OVERRIDES = {
    # Shared environment-independent retrieval pins (see eval/settings.py).
    **RETRIEVAL_EVAL_SETTINGS_OVERRIDES,
    # Whisper pipeline (re-enables the reranker the shared base disables)
    "whisper_max_nodes": 6,
    "whisper_min_relevance_score": 0.45,
    "whisper_candidate_pool_multiplier": 5,
    "whisper_injected_content_max_chars": 600,
    "whisper_reranker_enabled": True,
    "whisper_reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
    "whisper_reranker_min_score": 0.40,
    "whisper_reranker_blend_alpha": 0.6,
    "whisper_reranker_max_doc_chars": 512,
    "whisper_context_buffer_size": 5,
    "whisper_session_gap_minutes": 10,
    "whisper_intent_threshold": 0.65,
    "whisper_topic_shift_enabled": True,
    "whisper_topic_shift_threshold": 0.75,
    "whisper_injection_gate": 0.45,
    "whisper_no_overlap_ce_floor": 0.45,
    "whisper_no_overlap_cosine_floor": 0.70,
    "whisper_preference_applicability_enabled": True,
    "whisper_preference_applicability_gate": 0.40,
    "whisper_preference_max_nodes": 2,
    "whisper_exploration_enabled": True,
    # Ranking adjustments used by whisper post-processing
    "affinity_similarity_threshold": 0.70,
    "affinity_half_life_days": 30.0,
    "affinity_max_boost": 0.15,
    "affinity_implicit_weight": 0.8,
}
