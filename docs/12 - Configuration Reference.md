# Configuration Reference

Verified against the current repository state on 2026-06-09.

All settings live in `src/ormah/config.py` and use the `ORMAH_` prefix.

Settings are loaded from:

1. `~/.config/ormah/.env`
2. local `.env`

Later files override earlier ones.

## Server

| Setting | Default |
|---|---|
| `host` | `127.0.0.1` |
| `port` | `8787` |
| `log_format` | `text` |
| `log_level` | `INFO` |

## Paths

| Setting | Default |
|---|---|
| `memory_dir` | `~/.local/share/ormah/memory` |
| `backup_dir` | `~/.local/share/ormah/backups` |

## Backups

| Setting | Default |
|---|---|
| `backup_enabled` | `true` |
| `backup_interval_hours` | `24` |
| `backup_retention_count` | `10` |

Automatic backups are local and timestamped. The server creates them when memory
nodes exist and the newest backup is older than the configured interval. They
include source-of-truth memory files from `nodes/` and `deleted/`, but exclude
derived indexes, logs, config, and API keys. Use `ormah backup create`,
`ormah backup list`, `ormah backup status`, and `ormah backup restore <backup>`
for manual backup workflows.

## Embeddings

| Setting | Default |
|---|---|
| `embedding_provider` | `local` |
| `embedding_model` | `BAAI/bge-base-en-v1.5` |
| `embedding_dim` | `768` |
| `embedding_max_content_chars` | `512` |

## LLM

| Setting | Default |
|---|---|
| `llm_provider` | `none` |
| `llm_model` | `claude-haiku-4-5-20251001` |
| `llm_base_url` | `http://localhost:11434` |
| `llm_timeout_seconds` | `60` |
| `llm_num_predict` | `4096` |
| `llm_api_key_env_var` | unset |
| `llm_inherit_api_key` | `false` |

Note: setup may persist different values in `.env`. For remote providers, Ormah stores only key policy, such as `ORMAH_LLM_API_KEY_ENV_VAR=ANTHROPIC_API_KEY`; it does not store API key values. `llm_num_predict` maps to Ollama's `options.num_predict` request field.

Cheap JSON-capable models are usually enough for classification-style background jobs such
as feedback judging. Good starting points are local `llama3.2` through Ollama, or remote
low-cost models such as `gpt-4o-mini` or Claude Haiku through LiteLLM.

## Background Intervals

| Setting | Default |
|---|---|
| `auto_link_interval_minutes` | `1440` |
| `decay_interval_hours` | `24` |
| `conflict_check_interval_minutes` | `1440` |
| `conflict_check_all_spaces` | `false` |
| `duplicate_check_interval_minutes` | `1440` |
| `auto_cluster_interval_minutes` | `60` |
| `consolidation_interval_minutes` | `1440` |
| `importance_recompute_interval_minutes` | `120` |

## Hippocampus

| Setting | Default |
|---|---|
| `hippocampus_watch_dirs` | `[]` |
| `hippocampus_debounce_seconds` | `2.0` |
| `hippocampus_enabled` | `true` |
| `hippocampus_ignore_patterns` | `[]` |

Operational note: default-enabled does not mean active without configured watch dirs.

## Session Watcher

| Setting | Default |
|---|---|
| `session_watcher_enabled` | `false` |
| `session_watcher_dir` | `~/.claude/projects` |
| `session_watcher_debounce_seconds` | `60.0` |
| `session_watcher_min_turns` | `5` |
| `session_watcher_lookback_hours` | `72` |
| `session_watcher_idle_threshold` | `30.0` |

`session_watcher_dir` is the primary watch directory and remains the historical Claude Code
default. When it is left at the default, Ormah also watches `~/.codex/sessions` if that
directory exists.

### Feedback signal mining

| Setting | Default |
|---|---|
| `feedback_llm_judge_enabled` | `false` |
| `feedback_llm_judge_min_confidence` | `0.75` |

The session watcher always records free/local heuristic feedback signals for injected
whispers. When `feedback_llm_judge_enabled` is true and `llm_provider != "none"`, it also
asks the configured LLM to judge ambiguous turns. Confident `used` verdicts become positive
affinity; confident `irrelevant` verdicts become negative affinity; uncertain or
low-confidence verdicts remain observational `signals` rows only. The judge requests
compact JSON Schema output when available and falls back to JSON-object mode for providers
that reject schema output.

## Search

| Setting | Default |
|---|---|
| `fts_weight` | `0.4` |
| `vector_weight` | `0.6` |
| `similarity_threshold` | `0.4` |
| `rrf_k` | `60` |
| `similarity_blend_weight` | `0.5` |
| `fts_only_dampening` | `0.5` |
| `min_result_score` | `0.1` |
| `rrf_min_spread_ratio` | `0.05` |

### Question-query tuning

| Setting | Default |
|---|---|
| `question_fts_weight_scale` | `0.3` |
| `question_vector_weight_scale` | `1.5` |
| `question_similarity_blend_weight` | `0.85` |

### Post-retrieval tuning

| Setting | Default |
|---|---|
| `title_match_boost` | `2.0` |
| `length_penalty_threshold` | `300` |
| `recency_boost` | `0.05` |
| `recency_half_life_days` | `7.0` |
| `access_boost` | `0.05` |
| `tier_boost_core` | `0.1` |
| `tier_boost_working` | `0.0` |
| `tier_boost_archival` | `-0.1` |

Important note: current search applies tier as a multiplicative factor after confidence adjustment, while recency and access are additive proportional bonuses.

## Spreading Activation

| Setting | Default |
|---|---|
| `activation_decay` | `0.5` |
| `activation_seed_count` | `5` |
| `activation_max_per_seed` | `3` |

## Auto-Link / Merge / Importance

| Setting | Default |
|---|---|
| `auto_link_similarity_threshold` | `0.65` |
| `auto_link_cross_space_penalty` | `0.1` |
| `auto_link_max_edges_per_run` | `500` |
| `auto_merge_threshold` | `0.85` |
| `importance_access_weight` | `0.34` |
| `importance_edge_weight` | `0.33` |
| `importance_recency_weight` | `0.33` |
| `importance_access_reference` | `50` |
| `importance_edge_reference` | `20` |
| `importance_recency_half_life_days` | `14.0` |
| `decay_importance_threshold` | `0.5` |

`decay_importance_threshold` no longer affects `working -> archival` decay, which
depends on retrievability alone (#222). It remains a documented setting with no
effect in this version; bounded forgetting reintroduces a consumer.

## Whisper-Out and Nudge

| Setting | Default |
|---|---|
| `whisper_out_enabled` | `true` |
| `whisper_out_min_turns` | `3` |
| `whisper_out_interval` | `10` |
| `whisper_nudge_interval` | `10` |

## Whisper Recall

| Setting | Default |
|---|---|
| `whisper_max_nodes` | `6` |
| `whisper_min_relevance_score` | `0.45` |
| `whisper_reranker_enabled` | `true` |
| `whisper_reranker_model` | `Xenova/ms-marco-MiniLM-L-6-v2` |
| `whisper_reranker_min_score` | `0.40` |
| `whisper_reranker_blend_alpha` | `0.6` |
| `whisper_reranker_max_doc_chars` | `512` |
| `whisper_context_buffer_size` | `5` |
| `whisper_session_gap_minutes` | `10` |
| `whisper_intent_threshold` | `0.65` |
| `whisper_topic_shift_enabled` | `true` |
| `whisper_topic_shift_threshold` | `0.75` |
| `whisper_injection_gate` | `0.50` |
| `whisper_exploration_enabled` | `true` |

## Affinity

| Setting | Default |
|---|---|
| `affinity_similarity_threshold` | `0.70` |
| `affinity_half_life_days` | `30.0` |
| `affinity_max_boost` | `0.15` |
| `affinity_implicit_weight` | `0.8` |

## Space Prioritization

| Setting | Default |
|---|---|
| `space_boost_global` | `1.0` |
| `space_boost_other` | `0.6` |

## FSRS / Tier Limits / Ingestion

| Setting | Default |
|---|---|
| `core_memory_cap` | `50` |
| `working_decay_days` | `14` |
| `fsrs_initial_stability` | `1.0` |
| `fsrs_decay_threshold` | `0.3` |
| `fsrs_max_stability` | `365.0` |
| `fsrs_growth_factor` | `0.5` |
| `fsrs_growth_exponent` | `0.5` |
| `fsrs_spacing_cap` | `2.0` |
| `fsrs_reinforcement_cooldown_days` | `1.0` |
| `ingest_max_content_chars` | `100000` |

Reinforcement is bounded and diminishing (#221):

```text
spacing = min(R^-0.2, fsrs_spacing_cap)
S'      = min(S * (1 + fsrs_growth_factor * S^-fsrs_growth_exponent * spacing),
              fsrs_max_stability)
```

`fsrs_spacing_cap` keeps a very old memory from reaching the ceiling in a single
use, and `fsrs_growth_exponent` shrinks each step as stability rises — roughly 74
eligible updates take a node from `1.0` to `fsrs_max_stability`.
`fsrs_reinforcement_cooldown_days` allows at most one numeric stability update per
node per window; use still advances `last_accessed` on every event.
`fsrs_reinforcement_cooldown_days = 0` is a legal value that disables the cooldown
entirely, which allows unbounded-feeling growth within a single session — 74
recalls in one sitting reach `fsrs_max_stability`. This reproduces #221's
user-visible symptom and is currently legal and undocumented outside this note.

Because of that, `last_review` can lag real use by a full cooldown window, and any
job that treats it as a recency signal will read an actively used memory as stale.
Decay and importance therefore anchor on `last_accessed`. If you add a consumer
that anchors on `last_review` instead, keep
`fsrs_reinforcement_cooldown_days < -ln(fsrs_decay_threshold) x stability`
(about `1.2` days at the default threshold and an initial stability of `1.0`), or
that consumer will treat the cooldown lag as decay.
`fsrs_stability_growth` was removed in #221: it was a base multiplier, and the new
`fsrs_growth_factor` is an additive term with different semantics.

Upgrading does not rescale `stability` values written by the old unbounded
formula. Only future growth is bounded; a memory the old formula pushed to
`fsrs_max_stability` stays there and will not decay for roughly
`-ln(fsrs_decay_threshold) x fsrs_max_stability` days of disuse (about 440 at
the defaults). Correcting existing values would require a rescaling migration,
deliberately not done here.

Downgrading to a build older than #221 after upgrading is **not supported**. The
store records which lifecycle model wrote its `stability` values, and an older
binary does not know that key: it keeps writing with the unbounded formula while
leaving the version marker at the newer value, so a later upgrade trusts a marker
that no longer describes the data. Roll the binary back only together with a
backup taken before the upgrade.

## Agent-Backed Maintenance

| Setting | Default |
|---|---|
| `claude_maintenance_enabled` | `false` |
| `claude_maintenance_interval_hours` | `24` |
| `claude_maintenance_batch_size` | `25` |
| `claude_maintenance_cluster_max_chars` | `24000` |

## Code Anchor

- `src/ormah/config.py`
