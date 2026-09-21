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

The default recall track uses **k=30**, not Mem0's top-200 managed service; no reranker;
no automatic temporal filtering (the production classifier uses wall-clock now,
not historical `question_date`); raw LongMemEval in the first run; and subscription
CLI models instead of fixed API models/the official GPT-4o judge. CLI aliases can
change. Extract mode applies the shipped ingestion truncation at 100,000 characters;
cache records expose truncation. Image content in LoCoMo is not downloaded or
vision-encoded. No dataset gold or evidence enters extraction or answer generation.

Only reviewed summaries go under `results/` for these smoke runs. Raw journals,
datasets and caches must not be committed. A future published run needs complete
coverage, pinned models, methodology, the run ID, and owner review of its artifacts.

## Whisper retrieval track

Select `--retrieval whisper` to measure involuntary recall: the answering model
sees only the memories Ormah chooses to whisper for each question. The default
`--retrieval recall` retains deliberate top-k retrieval (k=30) and its existing
answer and judge prompts. Whisper uses the same haystack seeding, but applies the
full prompt classifier, cross-encoder reranker, production gates and six-node
cap. `--k` does not change the whisper cap or truncate `recall@whisper`.

```sh
uv run ormah eval bench run locomo --mode raw --retrieval whisper \
  --answer-provider codex --answer-model gpt-5.6-terra \
  --judge-provider codex --judge-model gpt-5.6-terra \
  --phase all --conversation 0 --limit 30 --workers 4 --run-id whisper-locomo-raw-c0-30
uv run ormah eval bench run longmemeval --mode raw --retrieval whisper \
  --answer-provider codex --answer-model gpt-5.6-terra \
  --judge-provider codex --judge-model gpt-5.6-terra \
  --phase all --limit 20 --workers 4 --run-id whisper-lme-raw-20
```

For a matched comparison, repeat with `--retrieval recall` and a separate run ID.
Strategy is recorded in the manifest and checked on resume; older manifests
without a strategy are treated as recall. Both tracks reuse identical cached
embeddings. Whisper settings come from `WHISPER_EVAL_SETTINGS_OVERRIDES`, shared
with the private whisper eval, including the shared retrieval pins. Startup must
load the reranker before any question is seeded. A failed model load or a logged
inference fallback aborts the run; `reranker_active` is recorded in the manifest
and each retrieval result. It means the model is available, not that every prompt
runs it: production can skip reranking for silence or identity-only intent.

Each question calls `get_whisper_context(question, space=None,
recent_prompts=None, session_id=None, _return_debug=True)`. Questions are independent:
no conversation history, topic suppression across questions, or learned feedback
is supplied. LoCoMo still shares one seeded haystack per conversation.

The ordered debug IDs are the memory allowlist. The adapter matches each ID's
rendered title and copies a content preview only if it appears immediately below
that title and matches the production truncator. Currently only the first two
memories have previews (up to 600 characters each); later memories retain titles
and empty content. No title-only memory is expanded from storage. The same dated
memory formatter and answer prompt are then used for both tracks. Memory dates
are added from stored metadata, as in recall. Unknown rendering formats fail
visibly. Framing, onboarding nudges and maintenance signals are excluded from
memory payloads and counts; raw output is retained separately for audit. A nudge
with zero debug IDs is **silence** and sends an empty memory list to the answerer.
There is no fallback retrieval or follow-up recall.

Whisper reports unchanged answer accuracy/J-score plus:

* `injection_rate`: fraction of successful retrievals with at least one memory.
* `mean_injected_memories` and `mean_whisper_context_chars`: mean counts and
  memory-only rendered whisper characters, including zero for silent questions.
* `mean_answer_context_chars`: actual dated memory payload characters sent to the
  answerer, excluding prompt instructions. Also reported for recall, along with
  its nonempty-context rate and memory count, for a direct context-budget comparison.
* `recall@whisper`: macro recall over the entire injected set, session-level for
  LongMemEval and exact turn-level for LoCoMo raw. The existing exclusions apply:
  no LongMemEval abstentions/missing gold, and no invented LoCoMo extract provenance.
* `abstention_silence_rate`: silence among successfully retrieved LongMemEval
  abstention questions, with `abstention_retrieved` exposing the denominator.
  This differs from judged abstention accuracy: unrelated injections can still
  lead to a correct abstention. A sample without abstentions reports null.

Failed retrievals are excluded from these means and exposed by error/coverage
counts; silence is a successful retrieval with zero context. Artifacts include
ordered IDs, raw and stripped whisper text, counts, context sizes, reference
date, reranker status and latency of the whisper call (excluding seeding and
adapter formatting).

### Historical dates

The production classifier's temporal and continuation intents use wall-clock
UTC. Automatic temporal parsing inside recall uses the same parser even if the
classifier did not identify temporal intent. For example, `last week` resolves
to the rolling window 14 to 7 days before now; historical 2023 memories would be
excluded on this node. A temporal classification without an explicit time phrase
falls back to the preceding three days; continuation also defaults to three days.

The benchmark scopes a context-local reference clock around the whisper call,
using LongMemEval's question date or LoCoMo's latest session date (its questions
have no query timestamps). Only these temporal interpretation sites use the
reference clock, and it resets even on exceptions. Production defaults, ranking,
gates, output, recency scoring and FSRS clocks are unchanged. Recall continues
to disable automatic temporal filtering. Thus the tracks differ in temporal
filtering as well as reranking and budget. Historical interpretation is still
limited by the shipped heuristics: windows are rolling rather than calendar
periods, classification can impose a recent window on a historical question,
and LoCoMo's inferred reference date may differ from the intended query time.

### What this number means

Deliberate retrieval benchmarks, including the top-k comparisons described above,
measure answering after an explicit memory search with a fixed retrieval budget.
This track measures answering from Ormah's automatic, gated pre-prompt injection,
including its decision to spend no context. It tests a different operating point;
it is not a like-for-like leaderboard comparison or a claim that no other system
has evaluated automatic injection. Report accuracy together with injection rate,
context size, retrieval recall, coverage, models and run ID. Tiny ordered smoke
samples diagnose behavior; they are not full-dataset accuracy estimates. This
track also differs from the private whisper eval, which scores expected injection
and suppression directly rather than downstream answers on public datasets.
