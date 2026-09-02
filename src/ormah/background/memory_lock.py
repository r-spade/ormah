"""Restore-awareness for jobs that mutate the live memory graph.

``L_mem`` buys background jobs exactly one thing: a mutation must not interleave
with a full graph restore. It does *not* provide per-operation atomicity — every
primitive a job mutates through already locks at its own granularity
(``engine.remember``/``connect``/``update_node``/``execute_merge`` and
``engine.file_store.save`` take ``L_mem`` per call; ``engine.db.transaction()``
takes ``L_db`` with ``BEGIN IMMEDIATE``).

So a job reads the restore epoch once, on entry, and takes ``L_mem`` only around
each apply step, via ``engine.memory_operation_at(epoch)``. If a restore lands
mid-run the job's whole snapshot is stale, not one row of it: the next apply step
raises :class:`RestoredUnderfoot`, the run ends, and the job returns at its next
interval.
"""

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)


class RestoredUnderfoot(Exception):
    """A full graph restore landed while this run was computing its snapshot."""


def restore_aware_job(job):
    """Supply the entry-time restore epoch; end the run cleanly if it moves.

    The wrapped job body takes ``(engine, epoch, *args, **kwargs)``. Callers keep
    calling ``run_x(engine)``.
    """

    @wraps(job)
    def wrapper(engine, *args, **kwargs):
        epoch = engine.restore_epoch
        try:
            return job(engine, epoch, *args, **kwargs)
        except RestoredUnderfoot:
            logger.info(
                "%s aborted: a graph restore landed mid-run; it will retry next interval",
                job.__name__,
            )
            return None

    return wrapper

