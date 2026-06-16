"""Bounded forgetting (#28): delete dead-weight archival nodes via conjunction gates.

Two phases per run, both behind the master switch ``deletion_enabled`` (default OFF):
  A. apply §1 gates → soft-delete eligible archival nodes (+ §3 cap backstop, task 06);
  B. hard-purge tombstones past the retention window (task 07).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from ormah.models.node import Tier

logger = logging.getLogger(__name__)


def run_forgetting(engine) -> None:
    """Soft-delete dead-weight archival nodes, then purge expired tombstones."""
    if not engine.settings.deletion_enabled:
        return  # opt-in; the graveyard is untouched until explicitly armed
    try:
        now = datetime.now(timezone.utc)
        _backfill_legacy_archived_at(engine)
        _run_gate_phase(engine, now)
        _run_cap_backstop(engine, now)
        _run_purge(engine, now)
    except Exception:
        logger.exception("Forgetting manager failed")  # stack trace: this job deletes data


def _run_gate_phase(engine, now: datetime) -> int:
    """Phase A: soft-delete archival nodes that are not protected AND are stale."""
    s = engine.settings
    candidates = [
        row["id"] for row in _archival_rows(engine)
        if not _is_protected(engine, row, now) and _is_stale_eligible(s, row, now)
    ]
    deleted = 0
    for node_id in candidates:
        if engine.delete_node_guarded(node_id, _eligibility_guard(engine, node_id, now)):
            deleted += 1
    if deleted:
        logger.info("Forgetting soft-deleted %d archival nodes", deleted)
    return deleted


def _hybrid_row(engine, node_id: str, conn):
    """A fresh gate row reading volatile fields from the SOURCE FILE, not the lagging index.

    Council R3 C5: mutators (`update_node`, `_touch_access`) write the markdown file BEFORE the
    index, so a guard reading only the index can see stale tier/last_accessed and delete a node
    mid-promotion. The file is authoritative for tier / last_accessed / archived_at / FSRS fields.
    `importance` stays index-authoritative (importance_scorer writes the index directly), and
    affinity/edges are read via conn (serialized by BEGIN IMMEDIATE). Returns None if the file is
    gone or no longer archival.
    """
    node = engine.file_store.load(node_id)
    if node is None or node.tier != Tier.archival:
        return None
    irow = conn.execute("SELECT importance FROM nodes WHERE id = ?", (node_id,)).fetchone()
    importance = irow["importance"] if irow and irow["importance"] is not None else node.importance
    return {
        "id": node.id,
        "importance": importance,
        "stability": node.stability,
        "last_review": node.last_review.isoformat() if node.last_review else None,
        "last_accessed": node.last_accessed.isoformat(),
        "archived_at": node.archived_at.isoformat() if node.archived_at else None,
    }


def _eligibility_guard(engine, node_id: str, now: datetime):
    """Build a guard(conn) that re-validates the gates from the source file inside the txn."""
    s = engine.settings

    def guard(conn) -> bool:
        row = _hybrid_row(engine, node_id, conn)
        if row is None:
            return False  # promoted / recalled-out / gone since selection
        return not _is_protected(engine, row, now) and _is_stale_eligible(s, row, now)

    return guard


# --- shared gate predicates -------------------------------------------------

_ROW_COLS = "id, importance, stability, last_review, last_accessed, archived_at"


def _archival_rows(engine):
    return engine.db.conn.execute(
        f"SELECT {_ROW_COLS} FROM nodes WHERE tier = 'archival'"
    ).fetchall()


def _evaluate_protection(engine, row, now: datetime) -> tuple[bool, int]:
    """§1 hard protections — single source of truth, computing connectivity at most once.

    Returns ``(protected, degree)``. ``protected`` True means NEVER delete (Phase A and the cap
    both honor it). ``degree`` is returned so the cap's forget-score never recomputes it (H5).
    Cheap protections short-circuit before the edge query, so Phase A stays cheap for the common
    high-importance / feedback cases.
    """
    s = engine.settings
    if row["id"] == getattr(engine, "user_node_id", None):
        return True, 0                                # gate #7: self node
    if row["archived_at"] is None:
        return True, 0                                # un-aged (e.g. remember(tier=archival))
    importance = row["importance"] if row["importance"] is not None else 0.5
    if importance >= s.decay_importance_threshold:
        return True, 0                                # gate #4: high importance
    if _has_positive_feedback(engine, row["id"]):
        return True, 0                                # gate #5: ever positively useful
    degree, max_weight = _connectivity(engine, row["id"])
    if degree > s.deletion_max_degree or max_weight >= s.deletion_strong_edge_weight:
        return True, degree                           # gate #6: hub / strong edge
    return False, degree


def _is_protected(engine, row, now: datetime) -> bool:
    return _evaluate_protection(engine, row, now)[0]


def _is_stale_eligible(s, row, now: datetime) -> bool:
    """§1 staleness signals — sustained dead weight. NOT protections (the cap skips these)."""
    cutoff = now - timedelta(days=s.deletion_min_archival_days)
    if _parse_dt(row["archived_at"]) > cutoff:
        return False                                  # gate #2: in graveyard long enough
    if _parse_dt(row["last_accessed"]) > cutoff:
        return False                                  # gate #2: not re-accessed
    if _retrievability(row, now) >= s.deletion_retrievability_floor:
        return False                                  # gate #3: R below the floor
    return True


def _retrievability(row, now: datetime) -> float:
    """FSRS retrievability R = exp(-days_since_anchor / stability). 1.0 if uncomputable."""
    stability = row["stability"] if row["stability"] else 1.0
    anchor_str = row["last_review"] or row["last_accessed"]
    try:
        anchor = datetime.fromisoformat(anchor_str)
    except (ValueError, TypeError):
        return 1.0
    days_since = max((now - _aware(anchor)).total_seconds() / 86400, 0.001)
    return math.exp(-days_since / stability)


def _has_positive_feedback(engine, node_id: str) -> bool:
    row = engine.db.conn.execute(
        "SELECT 1 FROM affinity WHERE node_id = ? AND signal > 0 LIMIT 1", (node_id,)
    ).fetchone()
    return row is not None


def _connectivity(engine, node_id: str) -> tuple[int, float]:
    row = engine.db.conn.execute(
        "SELECT COUNT(*) AS degree, COALESCE(MAX(weight), 0) AS max_w "
        "FROM edges WHERE source_id = ? OR target_id = ?",
        (node_id, node_id),
    ).fetchone()
    return row["degree"], row["max_w"]


def _parse_dt(value: str) -> datetime:
    return _aware(datetime.fromisoformat(value))


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _run_cap_backstop(engine, now: datetime) -> int:
    """Evict worst-first by forget-score down to archival_soft_cap, respecting protections."""
    s = engine.settings
    if s.archival_soft_cap <= 0:
        return 0
    rows = _archival_rows(engine)
    overflow = len(rows) - s.archival_soft_cap
    if overflow <= 0:
        return 0

    scored: list[tuple[float, str]] = []
    for row in rows:
        protected, degree = _evaluate_protection(engine, row, now)  # connectivity once (H5)
        if protected:
            continue  # same hard protections as Phase A — accept overflow over deleting these
        scored.append((_forget_score(row, now, degree), row["id"]))

    scored.sort(reverse=True)  # highest forget-score (worst) first
    evicted = 0
    for _score, node_id in scored[:overflow]:
        # atomic delete-if-still-unprotected (council R2 C3) — staleness not required for the cap
        if engine.delete_node_guarded(node_id, _cap_guard(engine, node_id, now)):
            evicted += 1
    if evicted:
        logger.info("Forgetting cap backstop evicted %d archival nodes", evicted)
    return evicted


def _cap_guard(engine, node_id: str, now: datetime):
    def guard(conn) -> bool:
        row = _hybrid_row(engine, node_id, conn)  # source-of-truth recheck (council R3 C5)
        return row is not None and not _is_protected(engine, row, now)

    return guard


def _forget_score(row, now: datetime, degree: int) -> float:
    """Composite worst-first score: low R × low importance × age × low connectivity.

    Candidates already exclude protected nodes, so positive feedback never reaches here.
    """
    r = _retrievability(row, now)
    importance = row["importance"] if row["importance"] is not None else 0.5
    # archived_at is guaranteed non-null (NULL ⇒ protected), so age is well defined.
    age_days = max((now - _parse_dt(row["archived_at"])).total_seconds() / 86400, 0.0)
    return (1.0 - r) * (1.0 - importance) * age_days * (1.0 / (1 + degree))


def _run_purge(engine, now: datetime) -> int:
    """Hard-purge tombstones whose deleted_at is past the retention window."""
    s = engine.settings
    cutoff = now - timedelta(days=s.deletion_retention_days)
    purged = 0
    for node_id, deleted_at, path in engine.file_store.list_deleted():
        if not deleted_at:
            continue  # no clock → keep (fail-safe)
        try:
            ts = datetime.fromisoformat(deleted_at)
        except (ValueError, TypeError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts > cutoff:
            continue  # still inside the reversibility window
        if engine.file_store.purge(node_id, path=path):  # reuse the known path (no re-glob)
            engine._write_audit_log(operation="purge", node_id=node_id)
            purged += 1
    if purged:
        logger.info("Forgetting hard-purged %d expired tombstones", purged)
    return purged


_BACKFILL_META_KEY = "archived_at_legacy_backfill_done"


def _backfill_legacy_archived_at(engine) -> int:
    """One-time: stamp archived_at into legacy archival files lacking it (atomic via file_store).

    Runs once (guarded by a meta flag), outside any migration transaction. Uses the atomic
    file_store.save so a write failure can never truncate a source memory; the done-flag is set
    only on a fully clean pass, so transient failures retry next run.
    """
    done = engine.db.conn.execute(
        "SELECT 1 FROM meta WHERE key = ?", (_BACKFILL_META_KEY,)
    ).fetchone()
    if done:
        return 0

    stamped, skipped = 0, 0
    # Enumerate the SOURCE FILES on disk, not the derived index (council R3 H8): an archival
    # file absent from the index would otherwise be missed forever once `done` is set. We only
    # touch files whose own archived_at is None (true legacy markers) — a file that already
    # carries archived_at is authoritative and must not be re-indexed here, or we would clobber
    # a deliberately index-lagged archived_at=NULL (the mid-promotion state _hybrid_row defends).
    for node in engine.file_store.list_all():
        if node.tier != Tier.archival or node.archived_at is not None:
            continue
        try:
            node.archived_at = node.updated  # best proxy for the legacy demotion time
            path = engine.file_store.save(node)   # atomic tmp + fsync + os.replace
            engine.builder.index_single(path)     # stamp the index
            stamped += 1
        except Exception:
            skipped += 1
            logger.warning("archived_at backfill skipped node %s", node.id)

    # Only mark done on a fully clean pass — a save-ok/index-fail node is re-checked next run
    # (its file has archived_at but the index does not, so the loop re-indexes it).
    if skipped == 0:
        with engine.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')",
                (_BACKFILL_META_KEY,),
            )
    if stamped or skipped:
        logger.info(
            "Forgetting legacy backfill: stamped %d, skipped %d", stamped, skipped
        )
    return stamped
