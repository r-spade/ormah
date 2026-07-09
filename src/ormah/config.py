"""Application configuration via environment variables and .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


_ENV_FILES = [
    Path.home() / ".config" / "ormah" / ".env",  # Fixed global config
    Path(".env"),  # Local override (cwd)
]
# pydantic-settings reads later files with higher priority
_EXISTING_ENV_FILES = [str(p) for p in _ENV_FILES if p.exists()]


class Settings(BaseSettings):
    model_config = {"env_prefix": "ORMAH_", "env_file": _EXISTING_ENV_FILES or ".env", "extra": "ignore"}

    # Server
    host: str = "127.0.0.1"
    port: int = 8787
    log_format: str = "text"  # "text" or "json"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Paths
    memory_dir: Path = Path.home() / ".local" / "share" / "ormah" / "memory"

    # Local memory backups. Only source-of-truth memory files are backed up;
    # SQLite/vector indexes are derived and rebuilt after restore.
    backup_enabled: bool = True
    backup_dir: Path = Path.home() / ".local" / "share" / "ormah" / "backups"
    backup_interval_hours: int = 24
    backup_retention_count: int = 10

    # Embeddings
    embedding_provider: str = "local"  # "local", "ollama", "litellm"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # LLM for extraction. Disabled by default; setup can opt into cloud/local providers.
    llm_provider: str = "none"
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: int = 60
    llm_num_predict: int = 4096
    llm_api_key_env_var: str | None = None
    llm_inherit_api_key: bool = False

    # Background intervals (LLM-dependent tasks default to daily to keep costs low)
    auto_link_interval_minutes: int = 1440
    decay_interval_hours: int = 24
    conflict_check_interval_minutes: int = 1440
    conflict_check_all_spaces: bool = False
    duplicate_check_interval_minutes: int = 1440
    auto_cluster_interval_minutes: int = 60

    # Hippocampus (file watching & auto-ingestion)
    hippocampus_watch_dirs: list[Path] = []
    hippocampus_debounce_seconds: float = 2.0
    hippocampus_enabled: bool = True
    hippocampus_ignore_patterns: list[str] = []

    # Session watcher (auto-ingest agent transcripts; default path is Claude Code)
    session_watcher_enabled: bool = False
    session_watcher_dir: Path = Path("~/.claude/projects")
    session_watcher_debounce_seconds: float = 60.0
    session_watcher_min_turns: int = 5
    session_watcher_lookback_hours: int = 72
    session_watcher_idle_threshold: float = 30.0

    # Tier limits
    core_memory_cap: int = 50
    working_decay_days: int = 14  # Deprecated: superseded by FSRS-based decay

    # FSRS spaced repetition decay
    fsrs_initial_stability: float = 1.0    # days; starting stability for new nodes
    fsrs_decay_threshold: float = 0.3      # R below this = decay candidate
    fsrs_stability_growth: float = 1.5     # base multiplier on access
    fsrs_max_stability: float = 365.0      # cap at 1 year

    # Search
    fts_weight: float = 0.4
    vector_weight: float = 0.6
    similarity_threshold: float = 0.4
    rrf_k: int = 60

    # Embedding truncation
    embedding_max_content_chars: int = 512

    # Score blending
    similarity_blend_weight: float = 0.5
    fts_only_dampening: float = 0.5
    min_result_score: float = 0.1
    rrf_min_spread_ratio: float = 0.05

    # Question query adjustments
    question_fts_weight_scale: float = 0.3
    question_vector_weight_scale: float = 1.5
    question_similarity_blend_weight: float = 0.85

    # Title and length scoring
    title_match_boost: float = 2.0  # Multiplicative bonus for query terms in title (0 = disabled)
    length_penalty_threshold: int = 300  # Content length above which vector similarity is penalized

    # Scoring signals
    recency_boost: float = 0.05
    recency_half_life_days: float = 7.0
    access_boost: float = 0.05
    tier_boost_core: float = 0.1
    tier_boost_working: float = 0.0
    tier_boost_archival: float = -0.1

    # Spreading activation
    activation_decay: float = 0.5
    activation_seed_count: int = 5
    activation_max_per_seed: int = 3

    # Auto-link
    auto_link_similarity_threshold: float = 0.65
    auto_link_cross_space_penalty: float = 0.1  # subtracted from similarity for cross-space pairs
    auto_link_max_edges_per_run: int = 500
    auto_link_max_nodes_per_run: int = 500  # cursor batch: nodes scanned per run

    # Auto-merge
    auto_merge_threshold: float = 0.85

    # Importance scoring weights (3 dynamic signals)
    importance_access_weight: float = 0.34
    importance_edge_weight: float = 0.33
    importance_recency_weight: float = 0.33
    importance_recompute_interval_minutes: int = 120

    # Importance: absolute normalization references
    importance_access_reference: int = 50
    importance_edge_reference: int = 20

    # Importance: recency half-life (separate from search recency)
    importance_recency_half_life_days: float = 14.0

    # Decay: skip nodes above this importance
    decay_importance_threshold: float = 0.5

    # Whisper-out (involuntary storage on compaction / session end)
    whisper_out_enabled: bool = True
    whisper_out_min_turns: int = 3
    whisper_out_interval: int = 10  # extract every N user prompts (0 = disabled)

    # Whisper nudge (periodic reminder to use ormah)
    whisper_nudge_interval: int = 10  # Nudge every N prompts (0 = disabled)

    # --- Score contract ------------------------------------------------
    # Two kinds of scores flow through retrieval; every threshold below
    # documents which kind it cuts:
    #   RANK-RELATIVE (ordering only): the blended hybrid `score` — RRF is
    #     min-max normalized per query, so any query's best candidate scores
    #     ~1.0 regardless of absolute quality. Absolute thresholds on it are
    #     meaningless across queries; use it only to order candidates.
    #   ABSOLUTE (gating): `ce_absolute` (cross-encoder score linearly
    #     rescaled from [-12, +6] to [0, 1]) and `raw_cosine` (pre-penalty
    #     vector similarity). Safe to compare against fixed thresholds.

    # Whisper (involuntary recall)
    whisper_max_nodes: int = 6
    # Pre-rerank noise trim: a candidate reaches the reranker if either its
    # RANK-RELATIVE blended score or its ABSOLUTE raw cosine clears this.
    # Its job is only to spare the cross-encoder obvious junk — the absolute
    # injection gate does the real cutting after reranking.
    whisper_min_relevance_score: float = 0.45
    # Candidate pool fed to the reranker/gate = whisper_max_nodes * this
    # multiplier. Retrieve-then-rerank needs a deep pool so the cross-encoder
    # can rescue memories the bi-encoder under-ranked; final injection is
    # still capped at whisper_max_nodes.
    whisper_candidate_pool_multiplier: int = 5
    # Max characters of node content injected for the top full-content
    # whispers; truncated at a word boundary. Full content stays one
    # recall_node call away (the whisper framing says so).
    whisper_injected_content_max_chars: int = 600

    # Whisper reranking (cross-encoder with linear-rescale blended scoring)
    whisper_reranker_enabled: bool = True
    whisper_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Post-affinity-boost floor on the RANK-RELATIVE blended score; defines
    # which candidates enter whisper_log and the exploration pool.
    whisper_reranker_min_score: float = 0.40
    whisper_reranker_blend_alpha: float = 0.6
    whisper_reranker_max_doc_chars: int = 512

    # Whisper context buffer (session-aware search enhancement)
    whisper_context_buffer_size: int = 5  # max recent prompts to keep per session
    whisper_session_gap_minutes: int = 10  # prune prompts older than this

    # Whisper intent classification
    whisper_intent_threshold: float = 0.65  # min cosine similarity for intent match

    # Whisper topic-shift detection (skip injection when topic unchanged)
    whisper_topic_shift_enabled: bool = True
    whisper_topic_shift_threshold: float = 0.75  # cosine sim above this = same topic

    # Whisper injection gate — cuts the ABSOLUTE gate score (ce_absolute
    # when the reranker ran, raw_cosine otherwise, plus any affinity delta).
    # 0.45 on the ce_absolute scale ≙ raw cross-encoder score −3.9: real
    # paraphrase matches land around raw −3 (≈0.49) while true noise sits
    # below raw −5 (≤0.39); tuned against eval/whisper (gate sweep, 2026-07).
    whisper_injection_gate: float = 0.45

    # Topical-filter vouchers for candidates sharing NO token with the prompt
    # (the fail-closed path): such a candidate survives only with an ABSOLUTE
    # relevance signal. The CE floor matches the injection gate (its added
    # value is keeping no-overlap junk out of the exploration pool); the
    # cosine floor applies when the reranker didn't run.
    whisper_no_overlap_ce_floor: float = 0.45
    whisper_no_overlap_cosine_floor: float = 0.70

    # Injection gate when the reranker did not run (unavailable, still
    # downloading, or disabled): the gate then cuts raw_cosine, a weaker
    # absolute signal, so demand a higher bar — degraded mode is more
    # conservative, never noisier. COSINE scale (bge noise floor ~0.5).
    whisper_injection_gate_no_reranker: float = 0.60

    # Deliberate recall floor: results below this are dropped rather than
    # padding to `limit` (recency-vouched temporal supplements exempt).
    # Cuts the RANK-RELATIVE blended score — pragmatic: observed cross-space
    # padding noise scores ~0.30 while relevant results score 0.6+. More
    # permissive than whisper's gate by design; recall is a deliberate act.
    recall_min_relevance_score: float = 0.35

    # Affinity boost (adaptive feedback loop)
    affinity_similarity_threshold: float = 0.70
    affinity_half_life_days: float = 30.0
    affinity_max_boost: float = 0.15
    affinity_implicit_weight: float = 0.8
    whisper_exploration_enabled: bool = True
    feedback_llm_judge_enabled: bool = False
    feedback_llm_judge_min_confidence: float = 0.75

    # Space prioritization
    space_boost_global: float = 1.0
    space_boost_other: float = 0.6

    # Ingestion
    ingest_max_content_chars: int = 100000

    # Consolidation
    consolidation_interval_minutes: int = 1440
    # Consolidator per-run limits (#89) — defaults preserve the previous
    # hardcoded behavior exactly.
    consolidation_max_clusters_per_run: int = 10
    consolidation_min_cluster_size: int = 2
    consolidation_cluster_threshold: float = 0.6
    consolidation_max_cluster_nodes: int = 5

    # Claude-in-the-loop maintenance
    claude_maintenance_enabled: bool = False
    claude_maintenance_interval_hours: int = 24  # hours between maintenance runs
    claude_maintenance_batch_size: int = 25  # candidates per type per run

    # --- Validators ---

    @field_validator("port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be 1–65535, got {v}")
        return v

    @field_validator("log_format")
    @classmethod
    def _log_format_enum(cls, v: str) -> str:
        allowed = {"text", "json"}
        if v not in allowed:
            raise ValueError(f"log_format must be one of {allowed}, got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _log_level_enum(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @field_validator("llm_provider")
    @classmethod
    def _llm_provider_enum(cls, v: str) -> str:
        allowed = {"ollama", "litellm", "none"}
        if v not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got {v!r}")
        return v

    @field_validator("llm_api_key_env_var")
    @classmethod
    def _llm_api_key_env_var_allowed(cls, v: str | None) -> str | None:
        if v in (None, ""):
            return None
        allowed = {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
            "COHERE_API_KEY",
            "AZURE_API_KEY",
        }
        if v not in allowed:
            raise ValueError(f"llm_api_key_env_var must be one of {allowed}, got {v!r}")
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _embedding_provider_enum(cls, v: str) -> str:
        allowed = {"local", "ollama", "litellm"}
        if v not in allowed:
            raise ValueError(f"embedding_provider must be one of {allowed}, got {v!r}")
        return v

    @field_validator("llm_timeout_seconds")
    @classmethod
    def _llm_timeout_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"llm_timeout_seconds must be >= 1, got {v}")
        return v

    @field_validator("llm_num_predict")
    @classmethod
    def _llm_num_predict_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"llm_num_predict must be >= 1, got {v}")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def _embedding_dim_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {v}")
        return v

    @field_validator(
        "auto_link_interval_minutes",
        "conflict_check_interval_minutes",
        "duplicate_check_interval_minutes",
        "auto_cluster_interval_minutes",
    )
    @classmethod
    def _interval_minutes_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"interval must be >= 1 minute, got {v}")
        return v

    @field_validator("hippocampus_debounce_seconds")
    @classmethod
    def _debounce_min(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError(f"hippocampus_debounce_seconds must be >= 0.1, got {v}")
        return v

    @field_validator("session_watcher_debounce_seconds")
    @classmethod
    def _session_watcher_debounce_min(cls, v: float) -> float:
        if v < 10.0:
            raise ValueError(f"session_watcher_debounce_seconds must be >= 10.0, got {v}")
        return v

    @field_validator("decay_interval_hours")
    @classmethod
    def _decay_hours_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"decay_interval_hours must be >= 1, got {v}")
        return v

    @field_validator("backup_interval_hours")
    @classmethod
    def _backup_interval_hours_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"backup_interval_hours must be >= 1, got {v}")
        return v

    @field_validator("backup_retention_count")
    @classmethod
    def _backup_retention_count_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"backup_retention_count must be >= 1, got {v}")
        return v

    @field_validator("core_memory_cap")
    @classmethod
    def _core_cap_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"core_memory_cap must be >= 1, got {v}")
        return v

    @field_validator("fts_weight", "vector_weight")
    @classmethod
    def _search_weight_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"search weight must be >= 0, got {v}")
        return v

    @field_validator("rrf_k")
    @classmethod
    def _rrf_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"rrf_k must be >= 1, got {v}")
        return v

    @field_validator("rrf_min_spread_ratio")
    @classmethod
    def _rrf_min_spread_ratio_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"rrf_min_spread_ratio must be 0–1, got {v}")
        return v

    @field_validator(
        "similarity_threshold",
        "auto_link_similarity_threshold",
        "auto_merge_threshold",
        "feedback_llm_judge_min_confidence",
    )
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"threshold must be 0–1, got {v}")
        return v

    @field_validator("activation_decay")
    @classmethod
    def _activation_decay_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"activation_decay must be (0, 1], got {v}")
        return v

    @field_validator(
        "importance_access_weight",
        "importance_edge_weight",
        "importance_recency_weight",
    )
    @classmethod
    def _importance_weight_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"importance weight must be 0–1, got {v}")
        return v

    @field_validator(
        "importance_access_reference",
        "importance_edge_reference",
    )
    @classmethod
    def _importance_reference_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"importance reference must be >= 1, got {v}")
        return v

    @field_validator("decay_importance_threshold", "fsrs_decay_threshold")
    @classmethod
    def _decay_threshold_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"threshold must be 0–1, got {v}")
        return v

    @field_validator("fsrs_initial_stability", "fsrs_stability_growth")
    @classmethod
    def _fsrs_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"FSRS parameter must be > 0, got {v}")
        return v

    @field_validator("fsrs_max_stability")
    @classmethod
    def _fsrs_max_stability_positive(cls, v: float) -> float:
        if v < 1:
            raise ValueError(f"fsrs_max_stability must be >= 1, got {v}")
        return v

    @field_validator("ingest_max_content_chars")
    @classmethod
    def _ingest_max_content_chars_min(cls, v: int) -> int:
        if v < 1000:
            raise ValueError(f"ingest_max_content_chars must be >= 1000, got {v}")
        return v

    @field_validator("importance_recompute_interval_minutes", "consolidation_interval_minutes")
    @classmethod
    def _enrichment_interval_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"interval must be >= 1 minute, got {v}")
        return v

    @property
    def llm_enabled(self) -> bool:
        """True when an LLM provider is configured (not ``"none"``)."""
        return self.llm_provider != "none"

    @property
    def nodes_dir(self) -> Path:
        return self.memory_dir / "nodes"

    @property
    def db_path(self) -> Path:
        return self.memory_dir / "index.db"


settings = Settings()
