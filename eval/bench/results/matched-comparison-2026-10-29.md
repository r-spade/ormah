# Matched comparison: implemented, inference blocked

Ormah's gain over simple retrieval is **not measured by this job**. The reference
methods and frozen diagnostic selection are implemented, but the inference
isolation preflight did not pass. No public score promotion is supported.

| Method | LME planned/scored | LoCoMo planned/scored | Paired score delta |
|---|---:|---:|---|
| No memory | 120 / 0 | 200 / 0 | Unmeasured |
| Raw-turn dense cosine | 120 / 0 | 200 / 0 | Unmeasured |
| Ormah raw hybrid recall | 120 / 0 | 200 / 0 | Unmeasured |
| Ormah extracted hybrid recall | Out of scope | 200 / 0 | Unmeasured |

LoCoMo's 200 planned questions include 160 primary and 40 category-5 adversarial
questions. LME includes 114 answerable and six abstentions. All six LME types,
all five LoCoMo categories and all ten conversations have coverage. The previously
inspected diagnostic-v1 IDs are excluded; these datasets still have historical
scored runs, so this is diagnostic coverage, not a fresh heldout claim.

The common configuration is a 12,000-character dated memory prefix, at most
30 memories, identical question/date/answer prompt/judge per matched question,
BAAI/bge-base-en-v1.5 embeddings, no reranker or automatic temporal filtering.
No-memory deliberately has zero context. The exact #338 extracted LoCoMo corpus
is reused, with no new extraction. LME extraction is out of scope.

The implementation starts from historical stacked harness
`8fadd3f03830c475c9d92a5619878e07b8cf21ac`, not current production. Fetched main
was `af85c14d7b0a835248ba8f11f16e2f5f39541c4b`; it includes CPU inference,
reference-resolution and temporal changes not incorporated in this experiment.
Evidence came read-only from #338 at
`16d4bd58ef8ad9525b82209a5c96999cd11f5ea5`; no publication status was changed.
The committed split contains dataset checksums and exact selected IDs.

## Concrete inference blocker

Codex CLI 0.165.0 returned an arithmetic smoke answer while requesting
`gpt-5.6-terra`; that records an accepted requested alias, not a verified backend
identity. Its event stream showed no tool call. However, `debug prompt-input`
still exposed functions/clock/collaboration tools and skills despite job-local
empty auth home, disabled hooks/shell/apps/plugins/browser, ignored config/rules
on exec and scoped outer execution. Debug and exec are distinct paths, so this
is a failure to establish isolation, not proof that a scored answer used tools.
The model's self-report that tools were unavailable is insufficient.

Claude Code 2.1.280 subscription safe-mode returned status 429, session limit,
with reset reported at 13:30 UTC. Inference stopped without retries, paid APIs,
model switching during a run, global changes or relaxed safety. Both inference
children and launcher removed all three API keys. Resuming requires a verified
isolated subscription provider and an explicit pre-scoring model/CLI amendment.

## Separate historical agent adjudication

A predeclared stratified sample reviewed **40 historical model-judged failures
and 10 historical passes**: 18 LME raw-hybrid cases and 32 LoCoMo extracted-hybrid
cases, across all types/categories and ten conversations. These are not failures
of the unexecuted matched methods. Human review has not happened.

| Primary agent-assigned cause among 40 failures | Cases |
|---|---:|
| Extraction/storage detail loss | 15 |
| Answer misuse of exposed evidence | 10 |
| Retrieval miss | 7 |
| Source/annotation/modality ambiguity | 5 |
| Judge error or uncertainty | 3 |
| Observed exposure truncation | 0 |

Causes can overlap: 17/40 have some extraction/storage concern, 14/40 some
source/annotation ambiguity, 13/40 some answer misuse, and 4/40 some judge concern.
These are agent assessments with confidence and mixed-cause notes, not population
estimates. Negative corpus checks used source-session review plus corpus semantic
candidates; they do not prove absence of every possible unannotated paraphrase.
All 1,500 retrieved contents in these 50 cases matched frozen stored contents;
this establishes no observed content clipping, not semantic sufficiency.
Nine reviewed passes were supported; one illustrates a permissive-judge problem.

Representative cases:

- LME `gpt4_7a0daae1`: purchase on March 10 was stored but not retrieved;
  receipt on March 17 was exposed. The interval was unavailable to the answerer.
- LoCoMo `locomo:8:151`: source names *The Godfather* as motivation; extraction
  retains general fitness goals and loses the film detail.
- LME `b46e15ee`: March 19 Walk for Hunger was fully exposed; answer selected a
  February book drive for an April question asking about one month earlier.
- LME `9ee3ecd6`: answer says “100 more points,” matching gold `100`, but judge
  returns `no`. LoCoMo `locomo:2:88` similarly rejects specific policy topics
  that the exposed memory explicitly identifies as the blog's subject.
- LoCoMo `locomo:3:34` is marked correct despite answering no book recommendation
  and explicitly identifying *Little Women* as a film; partial title matching
  masks the denial. The other recommendation depends on image metadata.
- LoCoMo `locomo:0:135`: question asks October; source says pottery injury last
  month on October 13 (September). The answer gives another genuine October
  setback. This is not a clean extraction failure.

The local job retains `historical-adjudications.jsonl`, `HISTORICAL_REVIEW.md`,
checksummed packets with full source/stored/retrieved data and exact exposure,
and the deterministic selection rule/IDs. Raw datasets and case artifacts are
not committed. Same-model judging, the permissive LoCoMo rubric, unencoded images,
source precision and historical duplicate LME IDs remain limitations. Historical
seeding collapsed 143 same-ID rows in 13 haystacks; none are gold evidence and two
appeared in one context. The effect of retaining both dates is unmeasured.

## Next measurement and validation

The sole recommendation is to **complete the frozen matched measurement after
provider isolation is established**. Synthetic development tests already check
cosine normalization/ties, empty context, exact Unicode budgeting, partial
exposure, actual answer payloads and API-key stripping. Do not tune retrieval or
extraction from these reviewed cases. A later improvement needs separate
synthetic/development cases and an uninspected, prospectively locked holdout;
for LoCoMo, use genuinely new conversations because all ten existing clusters
have now been inspected. Retain conversation-cluster uncertainty for LoCoMo
and question-unit uncertainty for LME; no paired intervals are available yet.

Validation: 110 focused/existing tests passed, including benchmark, raw-cosine
and recall-concurrency tests. Test child peak RSS was 730,816 KiB and wall 33.76s;
that is a child-process peak, not a whole-job benchmark memory measurement.
No benchmark inference latency/usage or matched wins/losses/abstention results
are available. Only non-scored smoke calls consumed model usage. A local launch
recipe preserves two-worker bounds and checkpoint/resume commands; it must not
be used for inference before the provider blocker is resolved.
