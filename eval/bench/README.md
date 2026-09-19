# Public memory benchmarks

A source-checkout harness for LongMemEval-S and LoCoMo. Read [DESIGN.md](DESIGN.md)
for the approved scope. The default run performs **store, retrieve and report**
in raw mode, with no LLM calls. Dataset files, caches, isolated databases and full
run artifacts remain gitignored. Tests use synthetic fixtures and fake providers.

```sh
uv run ormah eval bench download --dataset all
uv run ormah eval bench run locomo --mode raw --k 30
uv run ormah eval bench run longmemeval --mode raw --k 30 --limit 50
uv run ormah eval bench report RUN_ID
uv run ormah eval bench report RUN_ID --json
```

## Explicit LLM tracks

The first extraction track is LoCoMo; LongMemEval starts with raw turns because
extracting its large haystacks is costly. CLI providers bill the owner's
subscription, not API credits. These are the small smoke scopes, not published
accuracy estimates:

```sh
uv run ormah eval bench run locomo --mode extract --extract-provider claude-cli \
  --answer-provider claude-cli --judge-provider codex --phase all \
  --limit 10 --conversation 0 --run-id locomo-extract-smoke10
uv run ormah eval bench run longmemeval --mode raw --answer-provider claude-cli \
  --judge-provider codex --phase all --limit 10 --run-id longmemeval-raw-smoke10
```

Extraction uses the shipped `MemoryEngine.ingest_conversation(..., dry_run=True)`
prompt through a bench-only adapter. `--phase store` explicitly enables extraction
when mode is extract; omitting `--phase` in extract mode is rejected. Specify
comma-separated phases or `all`. Answer and judge never run implicitly.
`--workers` defaults to four and bounds CLI work. Calls time out after 300 seconds;
transient failures retry twice with backoff. Permanent errors and invalid verdicts
are recorded, never interpreted as correct answers.

`claude-cli` defaults to `sonnet`; the subprocess uses `--bare`, no session
persistence, empty tools, and no MCP servers. Some Claude CLI versions disable
subscription OAuth under `--bare`: that causes a recorded authentication error,
not fallback to API billing. `codex` uses the configured model name, passed
explicitly while ignoring the rest of user config; the pinned node executable is
`/root/agent-tools/codex-0.154.0/node_modules/.bin/codex`. It runs from an empty
temporary directory, read-only, ephemeral, with shell tools disabled. Provider
versions, requested models and reported resolved model IDs are recorded. Set
`--extract-model`, `--answer-model` and `--judge-model` to pin explicit IDs.
CLI providers use their native sampling/output controls; the API's temperature-zero
and `max_tokens` settings are not hard limits on CLI outputs. Token projections
for CLI runs are allowances, not guaranteed caps, and exclude CLI system overhead.

The optional `anthropic` provider requires `uv sync --extra bench` and an
owner-supplied `ANTHROPIC_API_KEY`. It prints a rough price-table estimate before
execution, uses temperature zero and maximum 1,024 answer tokens, and charges
reported input/output/cache tokens to a shared `--max-usd` budget (default $5).
API calls serialize around the budget so concurrent workers cannot issue calls
against an outdated tally. Estimated next-call cost is checked before a call;
actual usage is journaled before an over-budget abort. One unexpectedly expensive
in-flight call can cross the cap; subsequent calls are prevented. Subscription
`total_cost_usd` is recorded as an **estimate**, never API spend. Missing CLI usage
is unknown rather than zero. Harness development never calls the paid API.

## Resuming and artifacts

Repeat the original dataset, mode, filters, models and k with `--run-id ID
--resume`; optionally change `--phase` to run the next stage. For example:

```sh
uv run ormah eval bench run locomo --run-id example --mode raw --k 30
uv run ormah eval bench run locomo --run-id example --mode raw --k 30 --resume \
  --phase answer,judge,report
```

`questions.jsonl` is an append-only phase journal. The last row for a question is
its current state. Completed phases are skipped, failed phases retried. Only a
torn final line is repaired. `calls.jsonl` records each provider attempt and its
usage, including prior attempts across resumes. Each haystack is checkpointed
separately for retrieval-only resumption. Session extraction caches include
session ID, full effective prompt hash, provider and model; embedding caches are
SHA-256 of the actual production embedding text, namespaced by embedding model
and dimension. Changed dataset checksums or run parameters cannot silently reuse
a run. Do not run two processes against the same run ID.

## Methodology

* LongMemEval streams one question/haystack at a time. LoCoMo shares a haystack
  across all questions in a conversation. Selection is dataset order, not random.
  `--conversation` is a zero-based LoCoMo index. The first ten questions are not a
  balanced sample; 50 LongMemEval questions also omit some question types.
  LongMemEval supplies its question date; LoCoMo uses the conversation's latest
  session date as the answer prompt's reference date because no query date is supplied.
* The isolated engine uses `eval.settings.RETRIEVAL_EVAL_SETTINGS_OVERRIDES`:
  local BGE-base-en-v1.5 embeddings and shipped hybrid retrieval settings. Raw
  mode creates one working-tier node per turn. LoCoMo prefixes speaker names;
  LongMemEval preserves role attribution in titles. Node creation timestamps are
  historical session dates; updated/accessed timestamps are current so FSRS does
  not discard the corpus merely because its dates are old. No auto-linking or
  production core-cap enforcement runs during seeding.
* Seeding writes the file store, SQLite index and batched vectors. Embeddings use
  production title/content formatting and truncation (512 content characters by
  default). CPU batches group texts by length (16 at a time) to reduce padding;
  content-hash lookup restores node order. Full stored text, not just the embedded
  prefix, reaches the answerer.
* Retrieval requests `limit=k`, `min_relevance=0.0`, `auto_temporal=False`, and
  `default_space=None`. Ranked IDs, scores, sources, complete answer context and
  retrieval latency are saved. The production gate variant filters those top-k
  results at the pinned `recall_min_relevance_score`; it does not refill them.
* LongMemEval retrieval recall is the fraction of gold `answer_session_ids` hit
  by the top-k memories. For nDCG, collapse duplicate sessions in first-hit order
  **after truncating memories to k**, then compute binary DCG against ideal gold
  sessions. Abstention questions (`_abs`) and missing gold are excluded from
  retrieval denominators; correctness includes abstentions and also reports them
  separately.
* LoCoMo retrieval recall compares exact turn `dia_id`s to evidence, macro-averaged
  per question, including category 5's annotated evidence. Category 5 is excluded
  from J-score, and its judged abstention rate is separate. Category 3 gold stops at the first semicolon.
  **Extracted facts lack exact turn provenance in the shipped ingest output:**
  LoCoMo turn recall is therefore unavailable in extract mode. We never credit an
  extracted fact with all the turns of its source session.
* Accuracy/J-score averages successful verdicts; `scored`, abstention counts and
  phase-error counts expose missing coverage. An incomplete run is not a valid
  published accuracy result. Retrieval p50/p95 measures the recall call only;
  per-phase wall time includes seeding separately. Startup, dataset loading and
  report generation add overhead to command wall time.

## Comparability and limitations

Answer prompts are condensed adaptations of Mem0's
[LoCoMo](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py)
and [LongMemEval](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/longmemeval/prompts.py)
prompts, with explicit reference dates and memory dates, concise answers, no
requested reasoning transcript, and abstention when facts are unavailable.
LoCoMo uses Mem0's unified no-evidence CORRECT/WRONG JSON judge rules: partial
list matches count, dates allow 14 days and durations allow 50 percent tolerance.
These permissive J-score rules should not be read as exact factual accuracy.
LongMemEval follows the official
[per-type and abstention judge](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py),
including complete-answer requirements, temporal off-by-one tolerance, updated
facts and personalization rubrics. Invalid judge output is an error.

This harness uses **k=30**, not Mem0's top-200 managed service; no reranker;
no automatic temporal filtering (the production classifier uses wall-clock now,
not historical `question_date`); raw LongMemEval in the first run; and subscription
CLI models instead of fixed API models/the official GPT-4o judge. CLI aliases can
change. Extract mode applies the shipped ingestion truncation at 100,000 characters;
cache records expose truncation. Image content in LoCoMo is not downloaded or
vision-encoded. No dataset gold or evidence enters extraction or answer generation.

Only reviewed summaries go under `results/` for these smoke runs. Raw journals,
datasets and caches must not be committed. A future published run needs complete
coverage, pinned models, methodology, the run ID, and owner review of its artifacts.
