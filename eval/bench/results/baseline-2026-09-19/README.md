# Ormah historical public benchmark baseline

Ormah's September 2026 explicit-recall baseline scored **81.8% (409/500)** on
LongMemEval-S and **71.75% (1,105/1,540)** on LoCoMo under the configurations below.
These are complete dataset runs, audited from saved per-question records on
2026-10-29. They measure an assistant using Ormah's memory retrieval and the
specified answering/judging model, not a model-independent memory score.

This is a historical baseline at Ormah 0.15.1, source
[`16d64577fb251fabcb0e53e60956a083aac83dc9`](https://github.com/r-spade/ormah/tree/16d64577fb251fabcb0e53e60956a083aac83dc9).
It is not a measurement of current production or automatic whisper. The two
tracks use different storage modes and judge rubrics; do not average them.
No claim of superiority to another system is made without a matched experiment.

## Results and coverage

| Track | Primary score | Coverage | Abstention score |
|---|---:|---|---:|
| LongMemEval-S, raw turns, recall top 30 | **81.8% (409/500)** | All 500 questions, all six types | 93.33% (28/30), included in primary score |
| LoCoMo, extracted memories, recall top 30 | **71.75% (1,105/1,540) J-score** | All 10 conversations, 1,986 questions processed | 92.60% (413/446), category 5 reported separately |

All questions completed store, retrieval, answer and judge with zero recorded
phase errors. LoCoMo's 446 adversarial questions are excluded from its primary
J-score; they were not dropped or unprocessed.

| LongMemEval question type | Correct / scored | Score |
|---|---:|---:|
| Knowledge update | 73/78 | 93.59% |
| Multi-session | 94/133 | 70.68% |
| Single-session assistant | 53/56 | 94.64% |
| Single-session preference | 22/30 | 73.33% |
| Single-session user | 69/70 | 98.57% |
| Temporal reasoning | 98/133 | 73.68% |

| LoCoMo category | Correct / scored | J-score |
|---|---:|---:|
| 1: multi-hop | 229/282 | 81.21% |
| 2: temporal | 197/321 | 61.37% |
| 3: open-domain | 48/96 | 50.00% |
| 4: single-hop | 631/841 | 75.03% |

LongMemEval session recall@30 was **97.74%** over 470 non-abstention questions.
Hitting a session does not prove the specific answer-bearing turn was retrieved.
Exact turn recall is unavailable for extracted LoCoMo memories, which retain
session-level provenance only.

## Configuration and meaning

Both tracks used BAAI/bge-base-en-v1.5 local embeddings (768 dimensions), historical
session dates, and structured hybrid recall with `k=30`, `min_relevance=0.0`,
`auto_temporal=False`, no reranker, and no whisper injection. The full effective
recorded settings are in each original manifest. Full retrieved content—not a
whisper preview—was supplied to the answerer.

LongMemEval stored one node per source turn. LoCoMo passed each of 272 sessions
through Ormah's shipped `ingest_conversation(..., dry_run=True)` prompt, producing
870 memories from 5,882 turns. Static seeding does not reproduce normal core-tier
assignment, about-self handling, graph linking, maintenance, or learned feedback.
This is not an evaluation of every part of Ormah's production lifecycle.

Every LLM phase requested `gpt-5.6-terra` through subscription Codex CLI, including
extraction for LoCoMo. The same requested model answered and judged, which can
introduce correlated errors. These historical provider records retain the requested
model name, not independently verified resolved backend model identifiers.

LongMemEval uses per-type and abstention prompts adapted from the
[official evaluator](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py),
but uses the recorded Codex model rather than the official GPT-4o judge.
LoCoMo uses the [Mem0-style J-score rubric](https://github.com/mem0ai/memory-benchmarks/blob/main/benchmarks/locomo/prompts.py),
which accepts partial list answers, dates within 14 days, durations within 50%,
and matching referents despite some description differences. **J-score is not
strict factual accuracy or the original LoCoMo paper's metric.** Exact prompt
file hashes are preserved in the manifests and match the recorded source commit.

Answerer inputs contain question/date and retrieved memory text, not gold answers
or evidence labels. The wrapper runs from a temporary directory with user config
ignored, ephemeral sessions, read-only sandbox, shell disabled, and instructions
not to use tools. Raw Codex event streams were not retained, so this package cannot
independently prove absence of every possible tool invocation.

## Context, latency and usage

| Measurement | LongMemEval raw | LoCoMo extract |
|---|---:|---:|
| Mean dated answer-memory payload | 10,262.81 characters | 8,966.78 characters |
| Retrieval p50 / p95 | 9.29 / 14.45 ms | 126.74 / 165.40 ms |
| Answer call p50 / p95 | 5.33 / 13.64 s | 5.99 / 14.21 s |
| Judge call p50 / p95 | 4.62 / 13.10 s | 5.00 / 9.00 s |
| Recorded invocation wall | 4,848.79 s | 9,402.80 s |
| Recorded peak RSS | 1,810,544 KiB | 814,404 KiB |
| Reported input / output tokens, all LLM phases | 15,119,898 / 31,270 | 63,797,740 / 232,213 |
| API spend | $0 | $0 |

Payload sizes were reconstructed from the exact dated memories in saved ranked
results, excluding question and instructions. They are characters, not measured
tokens. Call usage (including CLI overhead) is retained per call and per phase;
subscription usage is not free compute or zero token consumption.

LongMemEval ran on the Mac with Python 3.11.15, SQLite 3.53.1 and Codex CLI
0.155.0. LoCoMo ran on the CPU-only ai-agents node with Python 3.11.2, SQLite
3.40.1 and Codex CLI 0.154.0. Both recorded fastembed 0.8.0 and sqlite-vec 0.1.9,
with four LLM workers. Timing is descriptive of different machines and cache
states, not a controlled speed comparison. Each saved manifest contains one
invocation marked `resume=true`; these counters do not establish complete cold-run
cost.

## Evidence and offline verification

Run from a checkout containing this package, with Python 3.9+; no dependencies,
credentials, network or inference are needed:

```sh
python3 eval/bench/results/baseline-2026-09-19/verify.py
```

Expected: `409/500` and `1105/1540`, with all checks passed. The verifier checks
file hashes, unique question counts, phase completion, verdict parsing, primary
and abstention denominators, answer/judge ledger agreement, context sizes, and
zero API spend. It recomputes model-judged scores; it does not independently
establish that each model judgment is factually correct.

- [Audit](audit.json): independently recomputed counts, context sizes, source hashes,
  and dataset/provenance checks performed before export.
- [LongMemEval manifest](longmemeval/manifest.json) and [summary](longmemeval/summary.json).
- [LoCoMo manifest](locomo/manifest.json) and [summary](locomo/summary.json).
- [LongMemEval records](longmemeval/questions.jsonl.gz) and
  [LoCoMo records](locomo/questions.jsonl.gz): one final journal checkpoint
  per question, preserving question, gold, ordered retrieved memories/scores,
  answer, judgment, usage and latency. Earlier partial checkpoints are omitted;
  the original complete journal hash is in the audit.
- Each directory includes `calls.jsonl.gz`: the complete original provider ledger.
- `locomo/haystacks.json.gz` preserves the exact extracted corpus for all ten
  conversations. Its keys are original relative haystack paths. LongMemEval raw
  corpora can be reconstructed from the checksummed source dataset.
- [Checksums](SHA256SUMS.json) cover every package artifact except the checksum
  listing itself. Original manifests and summaries are copied without rewriting.

## Reproducing inference

Offline score verification above is deterministic. Fresh model inference is not:
subscription model aliases, CLI defaults and model behavior can change. These
legacy runs predate the stronger experiment/environment fingerprints added later;
the recorded source commit, prompt hashes and runtime fields are the provenance
available, not a claim that every dependency or environment variable was captured.

Use a separate checkout of the recorded commit and fresh run IDs. Match the
recorded Python/SQLite/fastembed/sqlite-vec versions and the CLI version for each
track. Set `ORMAH_BENCH_CODEX` to that CLI's executable **before** starting Python.
Use the recorded dataset bytes: expected SHA-256 values are in both manifests.
Do not silently substitute a newer dataset release if its checksum differs.

```sh
git clone https://github.com/r-spade/ormah.git ormah-baseline
cd ormah-baseline
git checkout 16d64577fb251fabcb0e53e60956a083aac83dc9
uv sync
uv run ormah eval bench download --dataset all
# Verify downloaded files match dataset_sha256 in the published manifests.
# Set ORMAH_BENCH_CODEX to the matching installed, subscription-authenticated CLI.
unset OPENAI_API_KEY ANTHROPIC_API_KEY

uv run ormah eval bench run longmemeval --mode raw --k 30 --phase all \
  --answer-provider codex --answer-model gpt-5.6-terra \
  --judge-provider codex --judge-model gpt-5.6-terra \
  --workers 4 --run-id reproduce-lme-raw-baseline

uv run ormah eval bench run locomo --mode extract --k 30 --phase all \
  --extract-provider codex --extract-model gpt-5.6-terra \
  --answer-provider codex --answer-model gpt-5.6-terra \
  --judge-provider codex --judge-model gpt-5.6-terra \
  --workers 4 --run-id reproduce-locomo-extract-baseline
```

These are equivalent fresh-run commands, not a verbatim history of the resumed
invocations. `uv sync` alone does not recreate every recorded dependency version;
match the manifest runtime before inference. Fresh extraction is stochastic;
use the published LoCoMo corpus when inspecting the historical result.
The historical run IDs end in `20260920`, while manifests record starts on
2026-09-19. IDs are labels, not authoritative timestamps.

## Known limitations and later whisper work

Thirteen LongMemEval haystacks contained repeated session IDs with identical text
and different dates. Historical seeding collapsed 143 duplicate turn rows by ID,
retaining the last date. None belonged to gold evidence; two affected turns appeared
in one answer context. The effect of preserving both dated copies was not measured.

This is a single run per track with model judgments, not a multi-seed estimate.
LoCoMo has only ten conversation clusters. No independent human adjudication or
matched no-memory/competitor baseline is included. LoCoMo images are not fetched
or vision-encoded. Its extracted memories lack exact supporting-turn provenance;
an abstaining answer alone does not prove extraction lost the needed fact.

Later [whisper diagnostics and experiments](https://github.com/r-spade/ormah/pull/328)
measure a different operating point: much smaller, automatically gated context.
Those development/held-out samples are explicitly diagnostic; no full-data
whisper score is claimed here. Neither tested experimental setting earned a
production-default change. Improving the score is separate from publishing this
baseline honestly.

## Attribution and artifact licenses

LongMemEval: Di Wu et al., [LongMemEval](https://github.com/xiaowu0162/LongMemEval),
with dataset from [xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval).
The dataset card identifies MIT licensing; the accompanying
[MIT notice](LONGMEMEVAL-LICENSE.txt) is retained.

LoCoMo: Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco
Barbieri and Yuwei Fang, *Evaluating Very Long-Term Conversational Memory of LLM
Agents* (2024), [source dataset](https://github.com/snap-research/locomo).
LoCoMo-derived evidence and extracted content retain the upstream
[CC BY-NC 4.0 terms and warranty disclaimer](LOCOMO-LICENSE.txt); they are not
relicensed under Ormah's MIT code license. Changes consist of memory extraction,
retrieval selection, generated answers/judgments, and packaging into final records.
The authors do not endorse these results. Source questions and memory content
remain attributable to their respective benchmark datasets.
