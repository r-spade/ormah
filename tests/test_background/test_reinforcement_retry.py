"""#272 D5: the sleep-cycle sweeper that makes the confirmed-use claim durable."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ormah.background.reinforcement_retry import _BATCH_SIZE, run_reinforcement_retry
from ormah.config import Settings
from tests.test_engine.test_confirmed_use_contract import (  # noqa: E402
    _claim_row,
    _make_nodes,
    _seed_whisper_log,
    _snapshot,
)


@pytest.fixture
def seeded_claim(engine):
    """A node and a whisper_log row, with the node's pre-reinforcement snapshot."""
    node_id = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, node_id)
    return log_id, node_id, _snapshot(engine, node_id)


def _stale_claim(engine, log_id, node_id, minutes_ago=30):
    """Insert a claim that was taken but never applied, old enough to be swept.

    `state` is named explicitly. The column DEFAULT is the terminal 'legacy_unknown'
    (Step 3), so an INSERT that omits it produces a row the sweeper correctly ignores
    — which would make every test in this file pass vacuously.
    """
    claimed_at = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO confirmed_use_claims "
            "(whisper_log_id, node_id, claimed_at, state) VALUES (?, ?, ?, 'pending')",
            (log_id, node_id, claimed_at),
        )


def test_sweeper_reinforces_a_pending_claim(engine, seeded_claim):
    """#272 D5-3: a claim the mutator never applied is repaired."""
    log_id, node_id, before = seeded_claim
    _stale_claim(engine, log_id, node_id)

    run_reinforcement_retry(engine)

    row = engine.db.conn.execute(
        "SELECT state, reinforced_at FROM confirmed_use_claims WHERE whisper_log_id = ?",
        (log_id,),
    ).fetchone()
    assert row["state"] == "applied", "the sweeper did not apply the claim"
    assert row["reinforced_at"] is not None
    assert _snapshot(engine, node_id) != before, "the sweeper marked but did not reinforce"


def test_sweeper_never_touches_terminal_claims(engine, seeded_claim):
    """#272 D5-8a: legacy_unknown and orphaned are terminal, not work items."""
    log_id, node_id, before = seeded_claim
    _stale_claim(engine, log_id, node_id)
    with engine.db.transaction() as conn:
        conn.execute(
            "UPDATE confirmed_use_claims SET state = 'legacy_unknown' "
            "WHERE whisper_log_id = ? AND node_id = ?",
            (log_id, node_id),
        )

    run_reinforcement_retry(engine)

    assert _snapshot(engine, node_id) == before, "the sweeper re-reinforced a legacy claim"
    assert _claim_row(engine, log_id, node_id)["state"] == "legacy_unknown"


def test_sweeper_is_at_most_once_across_runs(engine, seeded_claim):
    """#272 D5-4: running the sweeper twice reinforces once."""
    log_id, node_id, _ = seeded_claim
    _stale_claim(engine, log_id, node_id)

    run_reinforcement_retry(engine)
    after_first = _snapshot(engine, node_id)
    run_reinforcement_retry(engine)

    assert _snapshot(engine, node_id) == after_first, "the second sweep reinforced again"


def test_sweeper_skips_claims_inside_the_grace_margin(engine, seeded_claim):
    """#272 D5-5: a claim taken seconds ago may still be in flight — do not race it."""
    log_id, node_id, before = seeded_claim
    _stale_claim(engine, log_id, node_id, minutes_ago=0)

    run_reinforcement_retry(engine)

    assert _snapshot(engine, node_id) == before, "the sweeper raced an in-flight claim"


def test_sweeper_isolates_one_bad_node_from_the_batch(engine, seeded_claim, monkeypatch):
    """#272 D5-6: one RAISING node must not abandon the rest of the batch.

    The failure has to be injected. A claim for a node that merely does not exist
    returns cleanly through the terminal-claim path and would prove nothing about
    isolation. Both claims hang off the same whisper_log row: the primary key is
    (whisper_log_id, node_id) and only whisper_log_id carries a foreign key, so
    this is a legal pair. The bad one is older so ORDER BY claimed_at runs it first.
    """
    log_id, node_id, before = seeded_claim
    _stale_claim(engine, log_id, "ghost-node", minutes_ago=60)
    _stale_claim(engine, log_id, node_id, minutes_ago=30)

    real = engine._record_confirmed_use

    def flaky(target, *, whisper_log_id):
        if target == "ghost-node":
            raise OSError("disk full")
        return real(target, whisper_log_id=whisper_log_id)

    monkeypatch.setattr(engine, "_record_confirmed_use", flaky)

    run_reinforcement_retry(engine)

    assert _snapshot(engine, node_id) != before, "a sibling failure abandoned the batch"


def test_a_wall_of_failing_claims_does_not_starve_the_newest(
    engine, seeded_claim, monkeypatch
):
    """#272 D5-11: permanently-failing claims must not monopolise every run.

    Council round 1 (Codex, MEDIUM, 0.96) found this. Per-row isolation only saves the
    rest of THIS batch; it does nothing about the next run picking the same rows. With
    `ORDER BY claimed_at ASC LIMIT _BATCH_SIZE` and no record of attempts, _BATCH_SIZE
    permanently-broken claims fill every batch forever and a claim taken today is never
    tried at all — the sweeper reports itself healthy while the feature it exists to
    deliver is dead for every new claim.

    last_attempt_at plus the retry backoff is what breaks the tie: an attempted row
    steps out of the eligible set until the backoff expires, so the next run reaches
    past the wall. The wall here is deliberately larger than _BATCH_SIZE and OLDER than
    the victim, so claimed_at ordering alone would never get to it.
    """
    log_id, node_id, before = seeded_claim

    wall = _BATCH_SIZE + 5
    for i in range(wall):
        _stale_claim(engine, log_id, f"broken-{i}", minutes_ago=600 + i)
    _stale_claim(engine, log_id, node_id, minutes_ago=30)

    real = engine._record_confirmed_use

    def flaky(target, *, whisper_log_id):
        if target.startswith("broken-"):
            raise OSError("disk full")
        return real(target, whisper_log_id=whisper_log_id)

    monkeypatch.setattr(engine, "_record_confirmed_use", flaky)

    run_reinforcement_retry(engine)
    assert _snapshot(engine, node_id) == before, (
        "the wall is not large enough — the victim was reached on run 1, so run 2 "
        "proves nothing"
    )

    run_reinforcement_retry(engine)

    assert _snapshot(engine, node_id) != before, (
        "the failing wall starved the newer claim across every run"
    )


def test_scheduler_registers_reinforcement_retry(tmp_path, monkeypatch):
    """Final review I2: pin the sweeper's scheduler registration.

    Every other test in this file calls run_reinforcement_retry(engine) directly.
    Deleting the add_job block in start_scheduler would leave every one of them
    green while the durability this branch exists to add is silently off in
    production. Same shape as test_scheduler_registers_whisper_log_cleanup
    (tests/test_background/test_whisper_log_cleanup.py) and its siblings in
    tests/test_backup.py and tests/test_cloud_jobs.py.
    """
    from ormah.background import scheduler as scheduler_module

    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, func, trigger, **kwargs):
            self.jobs.append({"func": func, "trigger": trigger, **kwargs})

        def start(self):
            pass

        def get_jobs(self):
            return self.jobs

    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeScheduler)
    settings = Settings(
        memory_dir=tmp_path / "memory",
        reinforcement_retry_interval_minutes=45,
    )
    engine = SimpleNamespace(
        settings=settings,
        builder=SimpleNamespace(incremental_update=lambda: (0, 0)),
    )

    fake_scheduler, _tracker = scheduler_module.start_scheduler(engine)

    job = next(job for job in fake_scheduler.jobs if job["id"] == "reinforcement_retry")
    assert job["trigger"] == "interval"
    assert job["minutes"] == 45
