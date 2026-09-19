# Benchmark smoke validation — 2026-09-18

These are **smoke validation results, not published end-to-end accuracy claims**.
All measurements are from 2026-09-18 on the 4-vCPU CPU-only node, Ormah 0.15.1,
Python 3.11.2, SQLite 3.40.1, sqlite-vec 0.1.9 and fastembed 0.8.0, with
BAAI/bge-base-en-v1.5 embeddings. Source base was
`71094e83665da50d0e29b5d69cbd1470790e94c7` with implementation changes uncommitted.

## Free retrieval runs

`locomo-raw-free-v2`: all 10 conversations and 1,986 questions, zero errors.
Macro turn recall@30 **0.738223** on 1,982 labeled questions; four category-3
questions lack evidence. Retrieval p50/p95 **94.511 / 138.018 ms**.
Command wall **901.925 s (15.03 min)**; store **684.941 s**, retrieve **204.739 s**.
Peak RSS **811,728 KiB (792.7 MiB)**. Production-gate recall is identical.

| Category | Questions | Retrieval denominator | Recall@30 |
|---|---:|---:|---:|
| 1 | 282 | 282 | 0.589351 |
| 2 | 321 | 321 | 0.811526 |
| 3 | 96 | 92 | 0.454284 |
| 4 | 841 | 841 | 0.827784 |
| 5 | 446 | 446 | 0.669283 |

Category 5 contributes its evidence to retrieval recall but is excluded from
J-score. Categories 1–4 alone have recall@30 0.758241; that is not the overall
score. The initial `locomo-raw-free` attempt was discarded after vector-search
failures were discovered. Its cache partially warmed the valid run. Valid-run
ranked tags were restored from exact saved haystack node IDs after discovering
that structured SQL rows omit tags; rankings, text, scores and timings did not
change. This correction is recorded in the manifest. Initial tagless scores in
the command log are invalid. The valid LoCoMo run used embedding batches of 8.

`longmemeval-raw-free50`: 50 questions, zero errors. Session recall@30 **0.980000**,
nDCG@30 **0.900266**; retrieval p50/p95 **79.657 / 116.270 ms**. Production-gate
recall is identical. All selected questions are `single-session-user`; there are
no abstention questions in this sample. Other types are unmeasured.

| Question type | Questions | Recall@30 | nDCG@30 |
|---|---:|---:|---:|
| knowledge-update | 0 | N/A | N/A |
| multi-session | 0 | N/A | N/A |
| single-session-assistant | 0 | N/A | N/A |
| single-session-preference | 0 | N/A | N/A |
| single-session-user | 50 | 0.98 | 0.900266 |
| temporal-reasoning | 0 | N/A | N/A |

Cumulative completed phase wall **4,704.472 s (78.41 min)**: store 4,699.922 s,
retrieve 4.551 s. Throughput **38.261 questions/hour** (0.01063 questions/s).
Linear projection for 500 questions: **47,044.725 s (13.07 hours)** of phase work.
The resumed invocation took **4,476.606 s (74.61 min)**, peak RSS **1,099,016 KiB
(1,073.3 MiB)**. Original start to first final summary: **4,920.454 s (82.01 min)**;
that elapsed-time basis projects to **13.67 hours** for 500 questions.

This was not one uninterrupted cold run: after two completed questions and partial
third-haystack seeding, execution paused for a 64-turn CPU batch profile, then
resumed with length-sorted batches of 16 instead of 32, using identical text/model.
Completed-phase timing omits interrupted third-question work and the profiling
pause. Full-data shared-session cache reuse may reduce the projection; the sample
is not balanced by question type. The resumed benchmark peaked near 1.05 GiB; the initial interrupted process
was observed around 1.2 GiB, below the 3 GB limit.

## Subscription smoke attempts

Both exact requested ten-question scopes ran and exited nonzero on authentication
errors. Installed Claude Code **2.1.220** disables subscription OAuth under the
required `--bare`; calls returned `Not logged in · Please run /login`. No fallback
API call was made. Codex **0.154.0** had no successful answers to judge, so its
inference path remains unexercised. The final provider resolves the configured
judge model to `gpt-5.6-sol`; Claude was requested as `sonnet` with no resolved
model ID because inference never started.

| Run | Successful answers / scored | Provider attempts | Call p50 / p95 | Command wall |
|---|---:|---|---|---:|
| `locomo-extract-smoke10` | 0 / 0 | 6 failed extraction attempts, 5 sessions | 2.046 / 4.222 s (legacy timing caveat below) | 20.291 s |
| `longmemeval-raw-smoke10` | 0 / 0 | 10 failed answer attempts | 1.275 / 1.306 s | 33.571 s |

Accuracy/J-score on both scopes is **N/A**, not zero. LoCoMo has ten store failures
and dependent answer/judge errors. LongMemEval completed retrieval for ten questions
(recall@30 1.0, nDCG@30 0.926186), then ten answer failures and dependent judge errors.
API spend is **$0**. The new parser records LongMemEval's reported zero tokens and
zero subscription USD estimate; the older LoCoMo ledger marks usage unknown even
though its raw error payloads reported zero API duration. No successful-call
latency or full LLM runtime estimate can be inferred from authentication failures.

LoCoMo's second attempt recorded 4.927282 s cumulatively, including its previous
attempt and retry backoff, so its reported p95 is not a pure per-attempt latency.
A UUID containing `429` in the raw JSON error caused that spurious retry. Final
provider code extracts the result message and measures each attempt separately.
All per-attempt ledger measurements are retained in the corresponding summary JSON.

## Full-run token planning

These are character-count estimates (ceil(chars/4)), not measured LLM usage.
They exclude CLI system overhead and cache behavior. Output columns are allowances,
not CLI hard caps; native CLI generation can differ. No dataset session exceeds
the harness's 100,000-character ingest limit.

| Planned phase | Calls | Input token estimate | Output allowance |
|---|---:|---:|---:|
| LoCoMo extract | 272 | 592,265 | 1,114,112 |
| LoCoMo answer | 1,986 | 4,113,595 raw-context proxy; extract context unknown | 2,033,664 |
| LoCoMo judge, including category-5 abstention | 1,986 | 1,227,848 plus generated answers | 508,416 |
| LongMemEval raw answer | 500 | 1,238,740, projected from 123,874 in first 50 | 512,000 |
| LongMemEval judge | 500 | 86,342 plus generated answers | 128,000 |

The LoCoMo raw answer proxy does not measure the planned extracted-memory contexts.
LongMemEval answer projection uses only one question type. Full LongMemEval
extraction is outside first-run scope: 19,829 unique sessions, estimated 79,016,259
input tokens including repeated ingest instructions, and an 81,219,584-token output
allowance. The earlier design's 49M estimate used a different accounting basis.

## Commands executed

```sh
uv run ormah eval bench run locomo --mode raw --k 30 --run-id locomo-raw-free-v2
uv run ormah eval bench run longmemeval --mode raw --k 30 --limit 50 --run-id longmemeval-raw-free50
# Resumed after an intentional batching profile:
uv run ormah eval bench run longmemeval --mode raw --k 30 --limit 50 --run-id longmemeval-raw-free50 --resume
uv run ormah eval bench run locomo --mode extract --extract-provider claude-cli --answer-provider claude-cli --judge-provider codex --phase all --limit 10 --conversation 0 --run-id locomo-extract-smoke10
uv run ormah eval bench run longmemeval --mode raw --answer-provider claude-cli --judge-provider codex --phase all --limit 10 --run-id longmemeval-raw-smoke10
```

Artifacts and raw journals remain local under `eval/bench/artifacts/<run_id>/`;
only these summary JSON files and documentation are intended for version control.
Read [the methodology](../README.md) before comparing these numbers to other systems.
