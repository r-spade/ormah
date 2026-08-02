"""Batch judgment of candidate pairs — shared by the pairwise maintenance jobs (#87).

K=1 (the default) never builds a batch prompt: callers' existing single-pair
functions run unchanged, so upstream behavior is byte-identical out of the box.
At K>1, an LLM-unavailable chunk aborts the remaining chunks (outage guard).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from ormah.background.llm_client import extract_json, llm_generate

logger = logging.getLogger(__name__)

_BATCH_PREAMBLE = """\
You will judge {n} independent pairs below. Judge each pair strictly on its own; \
do not let any other pair influence a verdict.

Return ONE JSON object: {{"verdicts": [<one verdict object per pair, with the fields \
specified above PLUS a "pair_id" integer matching the pair's number>]}}. \
Include every pair_id exactly once."""


def build_batch_prompt(instruction_block: str, rendered_pairs: list[str]) -> str:
    parts = [instruction_block, _BATCH_PREAMBLE.format(n=len(rendered_pairs))]
    parts += [f"### Pair {i}\n{rp}" for i, rp in enumerate(rendered_pairs)]
    return "\n\n".join(parts)


def parse_batch_verdicts(raw: str, valid_ids: set[int]) -> dict[int, dict] | None:
    """Verdicts keyed by pair_id, or None when the payload is unusable."""
    try:
        data = json.loads(extract_json(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list):
        return None
    out: dict[int, dict] = {}
    for item in verdicts:
        if isinstance(item, dict) and item.get("pair_id") in valid_ids:
            out.setdefault(item["pair_id"], item)
    return out


def judge_pairs(
    settings,
    instruction_block: str,
    pairs: list[Any],
    render_pair: Callable[[Any], str],
    judge_single: Callable[[Any], dict | None],
    k: int | None = None,
) -> list[dict | None]:
    """Judge every pair; result aligned by index. None = no verdict this run.

    *k* overrides settings.maintenance_pairs_per_call (jobs pass their per-job
    K resolved as `job_pairs_per_call or maintenance_pairs_per_call`).
    """
    k = settings.maintenance_pairs_per_call if k is None else k
    if k <= 1:
        return [judge_single(p) for p in pairs]   # legacy flow, unchanged
    results: list[dict | None] = [None] * len(pairs)
    for start in range(0, len(pairs), k):
        idx = list(range(start, min(start + k, len(pairs))))
        if not _judge_chunk(settings, instruction_block, pairs, render_pair,
                            judge_single, idx, results):
            logger.warning("LLM unavailable; leaving %d of %d pairs unjudged this run",
                           len(pairs) - start, len(pairs))
            break
    return results


def _judge_chunk(settings, instruction_block, pairs, render_pair, judge_single,
                 idx, results) -> bool:
    """Judge one chunk. Returns False when the LLM looks unavailable (abort signal)."""
    if len(idx) == 1:
        verdict = judge_single(pairs[idx[0]])
        results[idx[0]] = verdict
        return verdict is not None   # single-path None = unavailable/unusable -> abort
    rendered = [render_pair(pairs[i]) for i in idx]
    prompt = build_batch_prompt(instruction_block, rendered)
    hint = settings.llm_timeout_seconds + settings.maintenance_timeout_per_pair_seconds * len(idx)
    raw = llm_generate(settings, prompt, json_mode=True, timeout_hint_seconds=hint)
    if raw is None:
        return False   # transient outage — no bisect, abort remaining work
    verdicts = parse_batch_verdicts(raw, set(range(len(idx))))
    if verdicts is None:
        logger.warning("batch of %d pairs returned unparseable output; bisecting", len(idx))
        mid = len(idx) // 2
        if not _judge_chunk(settings, instruction_block, pairs, render_pair,
                            judge_single, idx[:mid], results):
            return False
        return _judge_chunk(settings, instruction_block, pairs, render_pair,
                            judge_single, idx[mid:], results)
    for local_id, item in verdicts.items():
        results[idx[local_id]] = item
    return True
