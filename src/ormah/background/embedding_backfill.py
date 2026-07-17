"""Vector-store reconciliation job: backfill missing embeddings (#32)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_embedding_backfill(engine, stop_event=None) -> None:
    """Reconcile the vector store. Raises if the store is left incomplete.

    Unlike the other background jobs, this one does NOT swallow an incomplete
    result: when the run ends with ``missing > 0`` (embeddable nodes still lack a
    vector) it raises so ``tracked()`` records a job failure and ``/admin/health``
    reflects the degradation -- the intended health signal. A permanently-failing
    ("poison") node therefore stays visibly degraded instead of being masked.

    If ``stop_event`` is provided and becomes set, the backfill exits early.
    An interrupted run still raises if ``missing > 0`` — the incomplete state
    is reported honestly so the caller knows to reschedule.
    """
    result = engine.backfill_embeddings(stop_event=stop_event)
    if result.get("embedded") or result.get("missing"):
        logger.info(
            "Embedding backfill (%s): embedded=%d failed=%d missing=%d vec=%d/%d",
            result["mode"], result["embedded"], result["failed"], result["missing"],
            result["vec_count"], result["node_count"],
        )
    if result.get("missing", 0) > 0:
        raise RuntimeError(
            f"Embedding backfill incomplete: {result['missing']} embeddable nodes "
            f"still missing vectors (failed={result['failed']})"
        )
