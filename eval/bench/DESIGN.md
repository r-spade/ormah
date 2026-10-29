# Public memory benchmarks for Ormah (LongMemEval-S, LoCoMo)

Status: design approved 2026-09-18 (lives here because `docs/plans/` is
gitignored). Implementation delegated to a Codex job on ai-agents on branch
`feat/memory-benchmarks`. Owner reviews before any full LLM run.

## Why

Competitors publish LoCoMo / LongMemEval numbers (Mem0: 92.5 / 94.4 on their
managed platform, top-200 retrieval budget; Synap: 93.2 / 92, unverified).
Ormah has only private, local-only whisper/recall evals. We want honest,
reproducible public numbers with raw per-question artifacts, methodology,
model versions, cost and latency, and stated limitations. This also feeds a
possible research write-up.

## Budget and hard constraints

- Total paid LLM spend for the first published run: **under USD 15**.
  Usage credits are scarce (~EUR 50 left). Every paid phase must be gated
  behind an explicit flag, print a pre-run cost estimate, and abort when the
  running tally exceeds `--max-usd`.
- Long-running work runs on the always-on ai-agents node (4 vCPU, 6 GiB RAM,
  no GPU, no Ollama). The Mac may sleep. Nothing in the harness may assume the
  laptop stays awake.
- No private data, no golden corpora, no API keys in the repo. Datasets and
  run artifacts are gitignored (`eval/bench/data/`, `eval/bench/artifacts/`).
  Only `eval/bench/results/*.json` summaries and per-question JSONL for
  published runs are committed.
- Delegated agents never call paid APIs. The API key is only set by the owner
  when launching a paid phase.

## Scope of the first published run

| Track | Dataset | Store mode | Paid phases | Est. cost (Haiku 4.5) |
|---|---|---|---|---|
| A | LoCoMo (10 convs, 1,540 scored QA, cats 1-4) | `extract` via Ormah `ingest_conversation` | extract (~0.2M tok) + answer + judge | ~USD 8 |
| B | LongMemEval-S (500 q) | `raw` (no LLM, one memory per turn) | answer + judge | ~USD 2.5 |
| C | both | both | retrieval-only metrics | USD 0 |

Track B in `extract` mode is out of budget (49M input tokens, ~19.8k unique
sessions). The harness must still support it (cached, resumable) so it can be
run later, e.g. through a subscription-billed CLI on ai-agents, but it is not
part of the first published run.

## LLM providers (decided 2026-09-18, supersedes the cost table above)

The owner is on a Claude subscription plan and has ample Codex subscription
usage, so the first run should need **zero API spend**. Every LLM-touching
phase (extract, answer, judge) takes a provider:

| Provider | Mechanism | Billing | Notes |
|---|---|---|---|
| `claude-cli` | `claude -p <prompt> --model <m> --output-format json`, tools disabled | Claude subscription weekly limit | Logged in on ai-agents. Default for extract and answer. |
| `codex` | `codex exec` subprocess, reads prompt on stdin, `--output-last-message` | Codex subscription | Default for judge (GPT judge is closest to the official LongMemEval GPT-4o judge). |
| `anthropic` | Anthropic Python SDK | API credits | Optional. Only when `ANTHROPIC_API_KEY` is set; still gated by `--max-usd`. |

Extraction in `extract` mode must still go through Ormah's real
`ingest_conversation` prompt. Implement bench-only `LLMAdapter` subclasses
(in `eval/bench/`) for `claude-cli` and `codex`, and install them into the
engine for the run (a small `set_adapter()` helper on
`ormah.background.llm_client` is acceptable). Production providers are
unchanged.

CLI subprocess calls are slow (seconds each); run them with a bounded
worker pool (`--workers`, default 4), retry transient failures with
backoff, and treat a non-zero exit or unparsable output as a recorded
per-question error, never a crash of the run. Record the CLI model id and
version in the manifest. Token usage from CLI providers is recorded when the
CLI reports it and marked unknown otherwise; USD cost applies to the
`anthropic` provider only.

Default models: `claude-cli` uses `sonnet` for extract and answer;
`codex` uses the node's configured model for judge. Model ids are run
parameters and recorded in every artifact.

## Datasets

- LongMemEval-S: HF `xiaowu0162/longmemeval`, file `longmemeval_s` (278 MB
  JSON, 500 questions). Fields: `question_id`, `question_type` (6 types),
  `question`, `answer`, `question_date`, `haystack_dates`,
  `haystack_session_ids`, `haystack_sessions` (list of `{role, content}`
  turns), `answer_session_ids`. 30 abstention questions (`question_id`
  ending `_abs`). Sessions are shared across questions (25,112 refs, 19,829
  unique) so extraction/embedding caches must be keyed by session id.
- LoCoMo: `snap-research/locomo` `data/locomo10.json`, 10 conversations,
  1,986 QA. Each conversation has `session_N` turn lists with `dia_id`,
  `speaker`, `text` and `session_N_date_time`. QA has `question`, `answer`,
  `evidence` (dia_ids), `category` (1 multi-hop, 2 temporal, 3 open-domain,
  4 single-hop, 5 adversarial). Category 5 is excluded from the J-score and
  reported separately as an abstention rate. Category 3 gold answers are cut
  at the first `;` (Mem0 convention).

A `download` subcommand fetches both files into `eval/bench/data/` and
records their SHA-256 in the run manifest.

## Architecture (`eval/bench/`)

Mirror `eval/recall/` conventions: in-process `MemoryEngine` on an isolated
`eval/bench/eval_db/` directory, settings pinned with
`eval.settings.RETRIEVAL_EVAL_SETTINGS_OVERRIDES`, CLI handlers imported
lazily from `src/ormah/cli.py` with the same "not installed in published
runtime" guard. Tests under `tests/test_eval_bench/` with fakes, no network.

Pipeline phases, each resumable and cached on disk under
`eval/bench/artifacts/<run_id>/`:

1. **load**: dataset adapter yields `Question` and `Session` records in one
   common shape (`session_id`, `date`, `turns[{speaker, text, turn_id}]`).
2. **store**: build the memory set for a haystack.
   - `raw`: one memory per turn. Title = speaker + first ~80 chars. Content =
     the turn text (LoCoMo: prefix with speaker name so entity attribution
     survives). `created` = session date. Tags carry `bench:<dataset>`,
     `session:<id>`, `turn:<id>`. No LLM.
   - `extract`: `engine.ingest_conversation(session_text, dry_run=True)` per
     session, output cached as JSONL keyed by session id + prompt hash, then
     seeded as nodes with `created` = session date and the same tags. Uses
     Ormah's real ingest prompt so we measure the shipped extractor.
   - Embeddings are cached by `sha256(embedding_text)` in a local sqlite/npz
     cache and written with `VectorStore.upsert_batch`, so re-seeding a
     haystack costs no re-encoding. Seed via the same path as
     `eval/recall/seeder.py` (file store + index + vectors).
3. **retrieve**: per question, clear the eval DB, seed its haystack, call
   `engine.recall_search_structured(question, limit=k, min_relevance=0.0,
   auto_temporal=False)`. Record ranked node ids, scores, sources and
   latency. `auto_temporal` is off because the classifier resolves phrases
   like "last week" against wall-clock now, not `question_date`; note this as
   a limitation. Also record how many of the top-k pass the production
   `recall_min_relevance_score` so we can report a "production gate" variant.
   LoCoMo: the whole conversation is the haystack; questions of one
   conversation share a seed (seed once per conversation).
4. **answer** (paid): Anthropic SDK, `claude-haiku-4-5`, temperature 0,
   `max_tokens` 1024. Prompt adapted from Mem0's memory-benchmarks answer
   prompts (cite the source file in a comment), includes `question_date`
   and each memory's date. Records `usage` tokens and latency per call.
5. **judge** (paid or codex): LongMemEval uses the official per-type yes/no
   prompts from `xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py`
   including the abstention prompt. LoCoMo uses Mem0's unified
   CORRECT/WRONG prompt with JSON output. Judge provider is pluggable:
   `anthropic` (default) or `codex` (`codex exec` subprocess).
6. **report**: `summary.json` + text table. Metrics:
   - LongMemEval: accuracy overall and per `question_type`; abstention
     accuracy separately; retrieval `recall@k` and `ndcg@k` at session level
     (a hit is any retrieved memory tagged with a gold `answer_session_id`).
   - LoCoMo: J-score (cats 1-4) overall and per category; cat 5 abstention
     rate; retrieval `recall@k` at turn level against `evidence` dia_ids.
   - Cost: input/output tokens and USD per phase from a small price table
     keyed by model id; wall time per phase; p50/p95 retrieval latency.
   - Manifest: ormah version, git sha + dirty flag, embedding model,
     pinned settings, dataset SHA-256, model ids, k, mode, run timestamp.

Per-question JSONL rows carry everything needed to re-judge offline:
question, gold, retrieved memory ids/titles/scores, answer text, judge
verdict and reasoning, tokens, latency.

## CLI

```
ormah eval bench download [--dataset longmemeval|locomo|all]
ormah eval bench run longmemeval --mode raw --k 30 --phase store|retrieve|answer|judge|report|all \
    [--limit N] [--question-type T] [--answer-model M] [--judge-model M] \
    [--judge-provider anthropic|codex] [--max-usd 5] [--run-id ID] [--resume]
ormah eval bench run locomo --mode extract --k 30 ...   # same flags, --category 1..5
ormah eval bench report <run_id> [--json]
```

`--phase` defaults to the free phases only (`store,retrieve,report`). Paid
phases run only when named explicitly and `ANTHROPIC_API_KEY` is set;
otherwise the command explains what it would cost and exits non-zero.

## Publication contract

A published run commits `eval/bench/results/<dataset>-<mode>-<date>.json`
(summary + manifest) and `<same>.jsonl` (per-question rows). A short
`eval/bench/README.md` documents methodology, differences from Mem0 and the
official LongMemEval setup (k, store mode, models, no reranker, no
temporal auto-filter), and known limitations. Numbers are never quoted
without the run id they came from.

## Out of scope for this iteration

- Whisper (involuntary recall) evaluation on these datasets.
- Reranker-on variants and k sweeps beyond one extra k.
- BEAM or other >1M-token benchmarks.
- LongMemEval-M / oracle variants.
