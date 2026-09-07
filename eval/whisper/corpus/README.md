# Whisper eval corpus

The golden corpus (`golden/golden.jsonl`) and mined cases (`local/`) are
**local-only and gitignored** — they may contain real project and personal
data and are never pushed to GitHub. Only this README and the eval code are
tracked.

## Layout

- `golden/golden.jsonl` — hand-authored golden cases (JSONL, one case per line)
- `local/` — cases mined from the live DB via `ormah eval whisper mine`

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

Each case: `id`, `space`, `memories[]`, `prompts[]`, and optionally
`simulate_session`, `session_id`, `preserve_self`.

Memories support: `node_id`, `title`, `content`, `type`, `tier`, `tags`,
`space`, `confidence`, `importance`, `stability`, `access_count`,
`connections` (typed edges to other in-case nodes), and timestamps as
`created`/`updated`/`last_accessed` ISO strings or `*_days_ago`/`*_hours_ago`
numbers.

Prompt expectations: `should_inject` (a.k.a. `must_include`), `may_include`,
`should_not_inject` (a.k.a. `must_not_include`), `should_suppress`
(a.k.a. `must_be_silent`). Prompts may carry `session_id` and
`recent_prompts` for session-context probes.

## Isolation between cases

The runner reuses one isolated eval database, but each case resets the prior
fixture's nodes plus its retrieval, decision, and feedback diagnostics. This
keeps per-case evidence attributable to the current fixture while preserving
intentional startup metadata and the case's `preserve_self` behavior.

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
