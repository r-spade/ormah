"""Retry confirmed-use reinforcements whose claim committed but never landed (#272).

_claim_confirmed_use takes a monotonic latch inside the caller's transaction, and
_record_confirmed_use runs after that transaction commits with its exception isolated.
Before #272 a transient failure there was permanent: the claim was taken, so nothing
retried. The claim now carries a state, and this job sweeps the rows still 'pending'.

'legacy_unknown' (written before the state column existed, or by a binary that predates
it) and 'orphaned' (the node is gone) are terminal and never swept.

Each attempt stamps last_attempt_at, and an attempted row is ineligible until the
backoff expires. That is what stops a wall of permanently-failing claims from filling
every batch and starving every newer claim.

Not LLM-gated, so it keeps working under ORMAH_LLM_PROVIDER=none.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# A claim taken seconds ago may still have its reinforcement in flight — the mutator
# runs after the claiming transaction commits. Matches the interval
# session_watcher_reconcile_interval_minutes uses for the equivalent reason.
_GRACE_MINUTES = 5

# Deliberately smaller than whisper_log_cleanup_batch_size (1000): every row here does
# file I/O under the memory lock, while that job is pure SQL.
_BATCH_SIZE = 200

# How long an attempted-and-still-pending claim steps out of the eligible set.
#
# Without this the job starves: `ORDER BY claimed_at ASC LIMIT _BATCH_SIZE` with no
# record of attempts lets _BATCH_SIZE permanently-failing claims fill every batch
# forever, so a claim taken today is never tried once. Per-row exception isolation
# does not help — it saves the rest of THIS batch, and the next run selects the same
# rows. Deliberately longer than the hourly interval so a broken row is retried at
# most once per couple of runs rather than every run.
_RETRY_BACKOFF_MINUTES = 180


def run_reinforcement_retry(engine) -> None:
    """Re-apply reinforcements for claims left unapplied."""
    try:
        rows = engine.db.conn.execute(
            """
            SELECT whisper_log_id, node_id
            FROM confirmed_use_claims
            WHERE state = 'pending'
              AND claimed_at < datetime('now', ?)
              AND (last_attempt_at IS NULL OR last_attempt_at < datetime('now', ?))
            ORDER BY COALESCE(last_attempt_at, claimed_at) ASC
            LIMIT ?
            """,
            (
                f"-{_GRACE_MINUTES} minutes",
                f"-{_RETRY_BACKOFF_MINUTES} minutes",
                _BATCH_SIZE,
            ),
        ).fetchall()

        if not rows:
            return

        # Stamped BEFORE the loop, in its own committed transaction, for two reasons.
        # A row whose reinforcement raises would otherwise never record the attempt —
        # the mutator rolls its own transaction back — and a process killed mid-batch
        # would leave no trace either. Both cases would reopen the starvation this
        # column exists to close. Marking a row that then succeeds costs nothing: it
        # leaves 'pending' only if it failed, and success makes it terminal anyway.
        with engine.db.transaction() as conn:
            conn.executemany(
                "UPDATE confirmed_use_claims SET last_attempt_at = datetime('now') "
                "WHERE whisper_log_id = ? AND node_id = ?",
                [(r["whisper_log_id"], r["node_id"]) for r in rows],
            )

        repaired = 0
        for row in rows:
            # Isolated per row: one unreadable node must not abandon the rest of the
            # batch, exactly as the call sites isolate their own reinforcement.
            try:
                engine._record_confirmed_use(
                    row["node_id"], whisper_log_id=row["whisper_log_id"]
                )
                repaired += 1
            except Exception:
                logger.exception(
                    "reinforcement retry failed for node %s (whisper_log %s)",
                    row["node_id"],
                    row["whisper_log_id"],
                )

        logger.info(
            "Reinforcement retry: %d/%d claims repaired", repaired, len(rows)
        )
    except Exception:
        logger.exception("Reinforcement retry job failed")
