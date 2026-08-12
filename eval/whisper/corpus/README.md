# Whisper eval corpus

Private golden (`golden/golden.jsonl`) and mined (`local/`) cases are
**local-only and gitignored** because they may contain real project and personal
data. Public-safe regression fixtures live under `public/` and are tracked.

## Layout

- `golden/golden.jsonl` — hand-authored golden cases (JSONL, one case per line)
- `local/` — cases mined from the live DB via `ormah eval whisper mine`
- `public/contract-smoke-v2.jsonl` — synthetic schema/metric contract smoke fixture;
  it proves the harness, not product quality

## Measurement contract v2

New reviewed corpora use `schema_version: "2.0"`. Each case declares a dataset
version and partition (`public_regression`, `adversarial`, `replay`, or `holdout`).
Each evaluated turn records:

- `target_decision`: `inject`, `abstain`, or `ask_qualify`;
- eligible, required, acceptable, and forbidden node IDs;
- a rationale and risk/slice labels;
- adjudication status and reviewer count (not reviewer identity).

The runner reports micro node precision and recall, inject-required-turn useful recall,
abstain-turn false-positive rate, forbidden-node suppression accuracy, sample
denominators, and descriptive Wilson intervals. Node and turn observations can be
clustered within cases and sessions, so these intervals are diagnostic and are not
release-grade confidence claims. `ask_qualify` is accepted by the schema, but true
turn precision remains unavailable until material helpfulness and serialized
uncertainty can actually be judged. Exact node-label success is reported separately
as `strict_label_turn_precision`.
Repeat irrelevance, confirmed delivery, and downstream lift are reported as
unavailable rather than guessed from silence or agent behavior.

Metric values from contract v2 are not trend-comparable with the old report. Legacy
corpora retain their original metrics and `make eval` thresholds unchanged. The
old evaluator macro-averaged per-prompt node scores and divided forbidden-node hits
by all prompts; v2 uses the denominators stated above.

`--fail-below` remains a point-estimate regression check. It cannot substantiate a
C08 release claim without predeclared minimum samples and cluster-aware uncertainty.

Holdout files must be kept in access-restricted storage. The CLI permits only
aggregate holdout runs and refuses category filters, `--show-failures`, and
provisional labels so it cannot print per-case answers through normal options.

## Mined cases are provisional until reviewed

`ormah eval whisper mine` reads the live `whisper_log` **read-only** and drafts
labels from what whisper actually did — circular evidence, not ground truth.
Every mined case carries `"provisional": true` and is **skipped by
`ormah eval whisper run`** (use `--include-provisional` for a smoke run only).
Review the drafts in `local/mined-review.md`, correct wrong labels in
`local/mined.jsonl`, then run `ormah eval whisper import-labels` to clear the
flags and let the cases bind. The miner mines only prompts backed by a
`whisper_decisions` row, so deliberate-recall exposures (which also land in
`whisper_log`) are excluded.

## Case schema

Legacy cases use `id`, `space`, `memories[]`, `prompts[]`, and optionally
`simulate_session`, `session_id`, `preserve_self`.

Memories support: `node_id`, `title`, `content`, `type`, `tier`, `tags`,
`space`, `confidence`, `importance`, `stability`, `access_count`,
`connections` (typed edges to other in-case nodes), and timestamps as
`created`/`updated`/`last_accessed` ISO strings or `*_days_ago`/`*_hours_ago`
numbers.

Legacy prompt expectations: `should_inject` (a.k.a. `must_include`), `may_include`,
`should_not_inject` (a.k.a. `must_not_include`), `should_suppress`
(a.k.a. `must_be_silent`). Prompts may carry `session_id` and
`recent_prompts` for session-context probes.

Schema-v2 files cannot use legacy aliases. Every case and prompt has a stable unique
`id`; all cases in one file share one `dataset_version` and partition. Each prompt
uses `target_decision`, `eligible_node_ids`, `must_include`, `may_include`,
`must_not_include`, `rationale`, `labels`, and `adjudication`. `draft` labels are
skipped by default; `reviewed` and `adjudicated` labels bind. Reviewer counts are
asserted provenance supplied by corpus maintainers, not cryptographic proof.

## Case-design rules

1. **Labels are set from ground-truth judgment BEFORE the first eval run of a
   batch.** Labels describe what a good memory system *should* do, not what
   the current pipeline does. Per-batch eval runs exist to catch mechanical
   bugs (typo'd node IDs, bad seeding) — never to drive relabeling. A
   well-labeled failing case stays failing; that is the measurement. A label
   may only be edited when the label itself was wrong, with the reason
   recorded.
2. **≥6 memories per retrieval case.** Two-memory cases understate distractor
   pressure; new cases seed a realistic neighborhood.
3. **Distractors are named.** Plausible-but-wrong memories go in
   `should_not_inject` so precision failures are attributable.
4. **Debatable labels get `may_include`** — or the case is dropped. A
   mislabeled case is negative value.
5. **Prompts are phrased as users actually type** — short, contextual,
   often with zero token overlap with the target memory.
6. **Differentiator cases say what they guard** in `notes` (e.g. "guards the
   Wave-3 served-topic memory").
7. **Category is non-restrictive** (see `VALID_CATEGORIES` in `corpus.py` for
   the report ordering set).

## Baselines (honest numbers, --show-failures documents the misses)

| date | prompts | overall F1 | preference | identity | noise acc | notes |
|------|---------|-----------|------------|----------|-----------|-------|
| 2026-07-03 | 53 | 0.89 | 0.60 | 0.86 | 1.00 | pre-expansion |
| 2026-07-03 | 100 | 0.69 | 0.42 | 0.58 | 0.95 | post-expansion re-baseline; 30 honest failures: implicit-trigger misses, identity sibling over-injection, paraphrase-decision misses |
