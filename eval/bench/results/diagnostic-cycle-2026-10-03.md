# Stratified diagnostic cycle — 2026-10-03

These are paired diagnostic samples, not leaderboard claims. Selection used the
locked development splits only. The held-out splits were evaluated once after
locking `lower-gate` as the sole candidate. Answer, judge, and fresh extraction
used Codex CLI 0.154.0 with `gpt-5.6-terra`, subscription inference, isolated
configuration, no tools, and $0 API spend.

## Baselines

| Split / track | Recall primary | Whisper primary | Recall abstention | Whisper abstention | Recall / whisper mean context chars |
|---|---:|---:|---:|---:|---:|
| Dev LongMemEval raw | 51/60 | 34/60 | 10/12 | 10/12 | 11,735 / 675 |
| Dev LoCoMo raw | 25/32 | 22/32 | 7/8 | 8/8 | 7,263 / 788 |
| Dev LoCoMo extract | 21/32 | 18/32 | 7/8 | 7/8 | 8,945 / 844 |
| Held LongMemEval raw | 53/60 | 32/60 | 12/12 | 12/12 | 10,441 / 652 |
| Held LoCoMo raw | 28/32 | 20/32 | 7/8 | 7/8 | 7,198 / 822 |
| Held LoCoMo extract | 26/32 | 25/32 | 6/8 | 8/8 | 8,940 / 870 |

LongMemEval supporting-turn retrieval/full-exposure coverage was 92.6%/92.6%
for development recall versus 58.5%/41.5% for whisper, and 92.5%/92.5% for
held-out recall versus 55.7%/38.4% for whisper. LoCoMo raw development was
69.4%/66.3% for recall versus 48.8%/42.4% for whisper; held-out was 60.7%/60.7%
versus 37.8%/32.7%. Extracted mode has no exact turn-level denominator because
session provenance cannot prove that an individual fact survived extraction.

## Controlled development changes

`balanced-preview` changes two 600-character previews to six 200-character
previews, holding the theoretical content allowance at 1,200 characters. It
regressed all three primary samples: LongMemEval 29/60 (1 paired win, 6 losses),
LoCoMo raw 21/32 (0 wins, 1 loss), and LoCoMo extract 16/32 (0 wins, 2 losses).
Realized LoCoMo context increased substantially, so it was rejected.

`lower-gate` changes only the absolute post-reranker injection gate from 0.45 to
0.40. Development was mixed: LongMemEval 35/60 (1 win, 0 losses), LoCoMo raw
21/32 (0 wins, 1 loss) with one abstention regression, and LoCoMo extract 19/32
(2 wins, 1 loss) with one abstention improvement. The heterogeneous primary
aggregate was only +1 question. This was not convincing, but it was locked as
the less harmful candidate for one held-out check; no other variant was tried.

## One-time held-out comparison

| Track | Baseline | Lower gate | Paired wins / losses | 95% bootstrap interval for accuracy delta | Mean context chars, baseline → candidate |
|---|---:|---:|---:|---:|---:|
| LongMemEval raw | 32/60 | 32/60 | 1 / 1 | [-5.0%, +5.0%] question bootstrap | 652 → 698 |
| LoCoMo raw | 20/32 | 18/32 | 0 / 2 | [-14.7%, 0.0%] conversation-cluster bootstrap | 822 → 864 |
| LoCoMo extract | 25/32 | 23/32 | 0 / 2 | [-15.4%, 0.0%] conversation-cluster bootstrap | 870 → 881 |

LongMemEval abstentions tied 12/12. LoCoMo raw abstentions improved 7/8 to 8/8,
but extract abstentions regressed 8/8 to 7/8. The candidate is rejected: it did
not improve LongMemEval, regressed both LoCoMo primary samples, and consumed more
context. Production defaults remain unchanged.

## Reproduction and limitations

The exact IDs, dataset checksums, seed, and strata are in
`eval/bench/splits/diagnostic-v1.json`. Raw journals, frozen haystacks, caches,
and per-question diagnostics remain ignored under `eval/bench/artifacts/`.
Generate paired statistics with:

```console
uv run python -m eval.bench.compare BASELINE_RUN CANDIDATE_RUN
```

Every candidate used the paired recall run through `--haystack-source-run`.
LoCoMo held-out runs reused the same public conversations' development haystack
JSON, then froze new held-out baseline/candidate copies and hashes. LongMemEval
held-out recall seeded new question-specific haystacks; both whisper tracks used
those exact files. Timings were collected on a shared host with varying load and
must not be interpreted as clean speed comparisons. Character/4 token counts are
explicit estimates; provider token usage is measured separately in the job
handoff. Static benchmark extraction still omits production `about_self`,
identity-tier, and link handling.
