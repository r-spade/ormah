"""Per-day cooldown on numeric stability updates (#221)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ormah.models.node import CreateNodeRequest, NodeType, Tier
from tests.test_engine.test_confirmed_use_contract import _claim_fresh_event, _reinforce


def _row(engine, node_id: str):
    return engine.db.conn.execute(
        "SELECT access_count, last_accessed, stability, last_review FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()


def _make_node(engine) -> str:
    node_id, _ = engine.remember(CreateNodeRequest(
        content="A node whose reinforcement we are measuring",
        type=NodeType.fact,
        tier=Tier.working,
        title="Cooldown subject",
    ))
    return node_id


def _backdate_review(engine, node_id: str, days: float) -> None:
    """Move both timestamps back so the next touch is off cooldown."""
    when = datetime.now(timezone.utc) - timedelta(days=days)
    node = engine.file_store.load(node_id)
    node.last_review = when
    node.last_accessed = when
    engine.file_store.save(node)
    engine.db.conn.execute(
        "UPDATE nodes SET last_review = ?, last_accessed = ? WHERE id = ?",
        (when.isoformat(), when.isoformat(), node_id),
    )
    engine.db.conn.commit()


def test_ten_touches_in_one_day_produce_one_stability_update(engine):
    """AC4: ten uses, one numeric update, and the latest use time is recorded."""
    node_id = _make_node(engine)
    _reinforce(engine, node_id)

    after_first = _row(engine, node_id)
    for _ in range(9):
        _reinforce(engine, node_id)
    after_ten = _row(engine, node_id)

    assert after_ten["stability"] == after_first["stability"]
    assert after_ten["last_review"] == after_first["last_review"]
    assert after_ten["access_count"] == after_first["access_count"] + 9
    assert after_ten["last_accessed"] > after_first["last_accessed"]


def test_a_touch_after_the_cooldown_moves_stability_again(engine):
    node_id = _make_node(engine)
    _reinforce(engine, node_id)
    before = _row(engine, node_id)

    _backdate_review(engine, node_id, days=1.0)
    _reinforce(engine, node_id)
    after = _row(engine, node_id)

    assert after["stability"] > before["stability"]
    assert after["last_review"] > before["last_review"]


def test_reinforcement_anchors_on_last_accessed_not_a_lagging_last_review(engine):
    """A confirmed use inside the cooldown must not be invisible to the spacing calc.

    last_review gates *whether* the numeric update runs; last_accessed is the last
    confirmed use. When they diverge (a use landed inside the cooldown, so it moved
    last_accessed but not last_review), the growth calculation must measure the gap
    since that last use, not since the last numeric write (PR #239 review comment).
    """
    node_id = _make_node(engine)
    node = engine.file_store.load(node_id)
    node.stability = 1.0
    now = datetime.now(timezone.utc)
    node.last_review = now - timedelta(hours=25)
    node.last_accessed = now - timedelta(hours=1)
    engine.file_store.save(node)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, last_review = ?, last_accessed = ? WHERE id = ?",
        (node.last_review.isoformat(), node.last_accessed.isoformat(), node_id),
    )
    engine.db.conn.commit()

    _reinforce(engine, node_id)

    assert _row(engine, node_id)["stability"] == pytest.approx(1.50, abs=0.01)


def test_a_thirty_day_old_node_is_bounded_to_double(engine):
    """AC1 end to end: the unbounded formula produced ~202.7 here."""
    node_id = _make_node(engine)
    node = engine.file_store.load(node_id)
    node.stability = 1.0
    engine.file_store.save(node)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0 WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()
    _backdate_review(engine, node_id, days=30.0)

    _reinforce(engine, node_id)

    assert _row(engine, node_id)["stability"] == 2.0


def test_the_cooldown_does_not_freeze_the_decay_anchor(engine):
    """last_accessed must keep moving so decay never sees an active node as stale."""
    node_id = _make_node(engine)
    _reinforce(engine, node_id)
    first = _row(engine, node_id)

    _reinforce(engine, node_id)
    second = _row(engine, node_id)

    assert second["last_accessed"] >= first["last_accessed"]
    assert second["last_review"] == first["last_review"]


def _seed_pending_claim(engine, node_id: str, claimed_at: datetime) -> int:
    """Insert a whisper_log row and a 'pending' claim stamped with claimed_at.

    Mirrors _claim_fresh_event's insert shape but takes claimed_at explicitly
    (rather than SQL datetime('now')) so an out-of-order pair of claims can be
    built: an older one applied AFTER a newer one already moved last_accessed
    forward. strftime truncates to whole seconds, matching what
    _claim_confirmed_use's own datetime('now') produces, so the round trip
    through the DB does not introduce a spurious sub-second mismatch.
    """
    with engine.db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO whisper_log "
            "(session_id, prompt_hash, prompt_vec, node_id, score, was_injected, logged_at) "
            "VALUES ('test-out-of-order', ?, X'00', ?, 1.0, 1, datetime('now'))",
            (uuid.uuid4().hex, node_id),
        )
        whisper_log_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO confirmed_use_claims "
            "(whisper_log_id, node_id, claimed_at, state) VALUES (?, ?, ?, 'pending')",
            (whisper_log_id, node_id, claimed_at.strftime("%Y-%m-%d %H:%M:%S")),
        )
    return whisper_log_id


def test_out_of_order_reinforcement_does_not_regress_last_accessed(engine):
    """Final review I1: a sweep applied out of claimed_at order must not walk
    last_accessed backwards.

    C1 is claimed at T0, on event E1. C2 is claimed at T1 > T0, on event E2
    for the SAME node. C2 is applied first (e.g. C1 failed and stayed
    'pending' while a later use went through cleanly), moving last_accessed
    to T1. A retry sweep later applies C1 with its own clock, T0. T0 < T1,
    so the write must not move last_accessed back to T0.
    """
    node_id = _make_node(engine)
    now = datetime.now(timezone.utc)
    t0 = now - timedelta(hours=2)
    t1 = now - timedelta(hours=1)
    assert t0 < t1, "the two claim clocks must be distinguishable and ordered"

    log_c1 = _seed_pending_claim(engine, node_id, t0)
    log_c2 = _seed_pending_claim(engine, node_id, t1)

    # C2 (the newer claim) lands first.
    engine._record_confirmed_use(node_id, whisper_log_id=log_c2)
    after_c2 = _row(engine, node_id)
    assert after_c2["access_count"] == 1

    # C1 (the older claim) is applied afterwards, as a retry sweep would.
    engine._record_confirmed_use(node_id, whisper_log_id=log_c1)
    after_c1 = _row(engine, node_id)

    assert after_c1["access_count"] == 2, "the older claim's use was not counted"
    assert after_c1["last_accessed"] == after_c2["last_accessed"], (
        "an out-of-order claim regressed last_accessed: "
        f"{after_c2['last_accessed']!r} -> {after_c1['last_accessed']!r}"
    )


def test_reinforcement_survives_a_zero_stability_node(engine):
    """Node.stability is ge=0.0; exp(-t/0) used to raise ZeroDivisionError."""
    node_id = _make_node(engine)
    node = engine.file_store.load(node_id)
    node.stability = 0.0
    engine.file_store.save(node)
    engine.db.conn.execute("UPDATE nodes SET stability = 0.0 WHERE id = ?", (node_id,))
    engine.db.conn.commit()

    _reinforce(engine, node_id)

    assert _row(engine, node_id)["stability"] > 0.0


def test_concurrent_touches_run_reinforcement_once(engine, monkeypatch):
    """AC4 under concurrency (council round 3 I1; test rebuilt after task review).

    The sequential ten-touch test cannot see this: it is the *interleaving* that
    breaks the cooldown. Both threads read last_review before either writes it,
    both conclude they are off cooldown, and both reinforce. The recall paths
    that reach _record_confirmed_use carry no additional serialization, so this is
    the production shape, not a synthetic one.

    TWO construction choices, both load-bearing — read before editing:

    1. A DELAY widens the window; a Barrier would deadlock. Synchronizing the
       threads *inside* the critical section only works while the section is
       unguarded: once @_serialized_memory_operation is in place the second
       thread cannot enter until the first leaves, so a barrier there would time
       out on correct code. A sleep does not synchronize, so it is safe in both
       states — it just makes the unguarded race overwhelmingly likely.

    2. The assertion COUNTS reinforcements; it cannot be the final stability.
       Both threads read S=1.0 and both write 1.5 — the same value. Compounding
       to 2.25 needs read-after-write, which is the serialization the bug
       removes, so `stability == 1.5` holds whether the race fired or not. The
       number of reinforced_stability calls is the only signal that separates
       one bump from two.
    """
    import threading
    import time

    from ormah import lifecycle

    node_id = _make_node(engine)
    node = engine.file_store.load(node_id)
    node.stability = 1.0
    node.last_review = None
    engine.file_store.save(node)
    engine.db.conn.execute(
        "UPDATE nodes SET stability = 1.0, last_review = NULL WHERE id = ?", (node_id,)
    )
    engine.db.conn.commit()

    reinforcements: list[float] = []
    real_due = lifecycle.reinforcement_due
    real_reinforce = lifecycle.reinforced_stability
    lock = threading.Lock()

    def _slow_due(last_review, now, cooldown_days):
        verdict = real_due(last_review, now, cooldown_days)
        time.sleep(0.05)  # widen the check-then-write window
        return verdict

    def _counting_reinforce(*args, **kwargs):
        with lock:
            reinforcements.append(1.0)
        return real_reinforce(*args, **kwargs)

    import ormah.engine.memory_engine as engine_module

    monkeypatch.setattr(engine_module.lifecycle, "reinforcement_due", _slow_due)
    monkeypatch.setattr(engine_module.lifecycle, "reinforced_stability", _counting_reinforce)

    errors: list[BaseException] = []

    # Issue #272: each thread needs its own claimed event up front — claiming
    # must complete (its own transaction) before the mutator race below runs,
    # and the mutator is now at-most-once per (whisper_log_id, node_id).
    log_ids = [_claim_fresh_event(engine, node_id) for _ in range(4)]

    def _touch(whisper_log_id: int) -> None:
        try:
            engine._record_confirmed_use(node_id, whisper_log_id=whisper_log_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced via `errors` below
            errors.append(exc)

    threads = [threading.Thread(target=_touch, args=(log_id,)) for log_id in log_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"a touch thread raised: {errors}"
    assert not [t for t in threads if t.is_alive()], "a touch thread never finished"
    assert len(reinforcements) == 1, (
        f"cooldown ran reinforcement {len(reinforcements)} times under concurrency"
    )
    assert _row(engine, node_id)["access_count"] == 4, "every touch must still be counted"
