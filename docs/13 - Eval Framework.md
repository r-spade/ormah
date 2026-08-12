# Eval Framework

Verified against the current repository state on 2026-04-07.

Ormah includes an in-repo eval harness for the whisper system under `eval/whisper/`.

## Purpose

The whisper eval answers two questions:

1. are the right memories injected?
2. are irrelevant memories suppressed?

## Architecture

Current eval flow is in-process, not server-HTTP based.

```mermaid
flowchart LR
    CORPUS[golden corpus] --> SEED[seed eval DB]
    SEED --> ENGINE[isolated MemoryEngine]
    ENGINE --> RUN[run get_whisper_context]
    RUN --> METRICS[compute prompt metrics]
    METRICS --> REPORT[aggregate + report/json]
```

## Main Components

- `corpus/` - test definitions
- `seeder.py` - seeds an isolated eval memory graph
- `runner.py` - runs prompt cases against `engine.get_whisper_context(..., _return_debug=True)`
- `metrics.py` - computes precision/recall style metrics
- `report.py` - renders text reports
- `cli.py` - CLI entry

## Important Correction

The current eval runner does not need a running Ormah HTTP server for its main path. It creates an isolated engine and invokes whisper directly in-process.

Older docs describing the runner as `POST /agent/whisper` for each case are stale.

## CLI

```bash
ormah eval whisper run
ormah eval whisper run --category preference
ormah eval whisper run --show-failures
ormah eval whisper run --simulate-session
ormah eval whisper run --preserve-self
ormah eval whisper run --json
```

## Metrics

Aggregate metrics currently include:

- `injection_recall`
- `injection_precision`
- `f1`
- `top2_recall`
- `false_positive_rate`
- `suppression_accuracy` for noise/silent cases

## Current Snapshot

On 2026-04-07, a local run of:

```bash
uv run python3 -m ormah.cli eval whisper run --json
```

produced:

- `total_prompts = 50`
- `injection_recall = 0.9069767441860465`
- `injection_precision = 0.8953488372093024`
- `f1 = 0.8837209302325582`
- `top2_recall = 0.9069767441860465`
- `false_positive_rate = 0.12`
- `suppression_accuracy = 0.8571428571428571`

These numbers are more current than older notes that cited larger corpus totals such as `158` and `90`.

## Session Simulation

The runner can simulate multi-turn whisper behavior by maintaining `recent_prompts` and `session_id`, mirroring the logic used by `/agent/whisper`.

That makes it useful for:

- continuation prompts
- session-aware search-query enhancement
- first-turn vs follow-up behavior

## Walkthrough Example

For a case about the whisper eval pipeline:

1. seed the case-specific memories into the isolated eval DB
2. call `engine.get_whisper_context(..., _return_debug=True)`
3. capture returned injected node ids
4. compare them against:
   - `should_inject`
   - `should_not_inject`
   - silence/suppression expectations
5. aggregate those prompt-level results across the corpus

Seeding clears both candidate-level and prompt-level Whisper diagnostics before each case,
so one case cannot contaminate another case's measurements.

## Code Anchors

- `eval/whisper/cli.py`
- `eval/whisper/runner.py`
- `eval/whisper/metrics.py`
- `eval/whisper/seeder.py`
