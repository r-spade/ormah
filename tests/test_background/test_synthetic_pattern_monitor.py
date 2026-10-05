"""Rot detection for the synthetic-prompt pattern list (#143)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ormah.background.synthetic_pattern_monitor import (
    BUILTIN,
    OPERATOR,
    find_rotted_patterns,
    live_patterns,
    run_synthetic_pattern_monitor,
)
from ormah.config import Settings

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)
TASK_NOTIFICATION = r"<task-notification>"


def _decision(engine, *, outcome, matched_pattern, logged_at):
    """Insert one whisper_decisions row directly — this is the job's only input."""
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT INTO whisper_decisions (session_id, space, prompt_hash, intent, "
            "outcome, logged_at, matched_pattern) VALUES (?, NULL, 'h', NULL, ?, ?, ?)",
            ("s", outcome, logged_at.isoformat(), matched_pattern),
        )


def test_live_patterns_includes_builtins_and_operator_entries(engine):
    engine.settings.whisper_synthetic_prompt_patterns = [r"BATCH JOB"]
    live = live_patterns(engine.settings)

    assert (TASK_NOTIFICATION, BUILTIN) in live
    assert (r"BATCH JOB", OPERATOR) in live


def test_live_patterns_dedups_an_operator_copy_of_a_builtin(engine):
    """One regex must yield one entry, or it yields two proposals (council I1)."""
    engine.settings.whisper_synthetic_prompt_patterns = [TASK_NOTIFICATION]

    live = live_patterns(engine.settings)

    assert [p for p, _ in live].count(TASK_NOTIFICATION) == 1
    assert (TASK_NOTIFICATION, OPERATOR) in live  # operator wins: it is what the user can remove


def test_pattern_that_never_matched_is_not_rot(engine):
    """Irrelevance, not rot: <scheduled-task> matching zero means this install
    never runs scheduled tasks. Proposing removal would be noise."""
    engine.settings.whisper_pattern_rot_days = 30
    _decision(engine, outcome="injected", matched_pattern=None, logged_at=NOW)

    assert find_rotted_patterns(engine.db.conn, engine.settings, NOW) == []


def test_pattern_still_firing_is_not_rot(engine):
    engine.settings.whisper_pattern_rot_days = 30
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=TASK_NOTIFICATION, logged_at=NOW - timedelta(days=2))

    assert find_rotted_patterns(engine.db.conn, engine.settings, NOW) == []


def _rotted_history(engine, pattern, *, hits=2, age_days=60, opportunity=None):
    """`hits` past matches for `pattern`, then enough later traffic to prove the
    pattern had a real chance to fire again (the opportunity guard).

    hits defaults to 2 because whisper_pattern_rot_min_matches defaults to 2 — a
    single historical match is deliberately not rot (council I4).

    opportunity defaults to the configured minimum: a rotted pattern needs ample
    traffic since it last fired, or "it stopped" cannot be told from "the user was
    away".
    """
    for i in range(hits):
        _decision(engine, outcome="silent_synthetic", matched_pattern=pattern,
                  logged_at=NOW - timedelta(days=age_days + i))
    n = (engine.settings.whisper_pattern_rot_min_opportunity
         if opportunity is None else opportunity)
    for i in range(n):
        _decision(engine, outcome="injected", matched_pattern=None,
                  logged_at=NOW - timedelta(minutes=n - i))


def test_pattern_that_matched_before_and_stopped_is_rot(engine):
    engine.settings.whisper_pattern_rot_days = 30
    _rotted_history(engine, TASK_NOTIFICATION)

    rotted = find_rotted_patterns(engine.db.conn, engine.settings, NOW)

    assert len(rotted) == 1
    assert rotted[0].pattern == TASK_NOTIFICATION
    assert rotted[0].origin == BUILTIN


def test_operator_pattern_rot_carries_the_operator_origin(engine):
    engine.settings.whisper_pattern_rot_days = 30
    engine.settings.whisper_synthetic_prompt_patterns = [r"BATCH JOB"]
    _rotted_history(engine, r"BATCH JOB")

    rotted = find_rotted_patterns(engine.db.conn, engine.settings, NOW)

    assert [(r.pattern, r.origin) for r in rotted] == [(r"BATCH JOB", OPERATOR)]


def test_filter_disabled_proposes_nothing(engine):
    """council C1. With the filter off nothing writes silent_synthetic, so every
    last_seen freezes while human traffic keeps the vacation guard happy — the
    whole pattern list would age into false proposals claiming an upstream rename
    that never happened, and the user might delete a still-valid filter."""
    engine.settings.whisper_pattern_rot_days = 30
    engine.settings.whisper_synthetic_filter_enabled = False
    _rotted_history(engine, TASK_NOTIFICATION)

    assert find_rotted_patterns(engine.db.conn, engine.settings, NOW) == []


def test_single_historical_match_is_not_rot(engine):
    """council I4. One match months ago is not evidence of a live workflow."""
    engine.settings.whisper_pattern_rot_days = 30
    engine.settings.whisper_pattern_rot_min_matches = 2
    _rotted_history(engine, TASK_NOTIFICATION, hits=1)

    assert find_rotted_patterns(engine.db.conn, engine.settings, NOW) == []


def test_no_traffic_since_last_match_proposes_nothing(engine):
    """Rewritten from the old global vacation guard, which the opportunity guard
    replaces (final review, Important): zero traffic since the pattern last fired
    is zero opportunity to prove it stopped, whatever the calendar says."""
    engine.settings.whisper_pattern_rot_days = 30
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=TASK_NOTIFICATION, logged_at=NOW - timedelta(days=60))
    # No row after the match at all — zero opportunity.

    assert find_rotted_patterns(engine.db.conn, engine.settings, NOW) == []


def test_returning_from_vacation_does_not_rot_everything(engine):
    """The old global guard was satisfied by ONE prompt after a month away and
    proposed the entire pattern list as rotted (final review, Important)."""
    engine.settings.whisper_pattern_rot_days = 30
    # Fired right up until the user left, 36 days ago...
    _rotted_history(engine, TASK_NOTIFICATION, age_days=36, opportunity=0)
    # ...then a single prompt on returning.
    _decision(engine, outcome="injected", matched_pattern=None, logged_at=NOW)

    assert find_rotted_patterns(engine.db.conn, engine.settings, NOW) == []


def test_ample_traffic_since_last_match_is_rot(engine):
    """The mirror case: the marker really did stop, and there was plenty of
    traffic in which it could have appeared."""
    engine.settings.whisper_pattern_rot_days = 30
    _rotted_history(engine, TASK_NOTIFICATION)

    rotted = find_rotted_patterns(engine.db.conn, engine.settings, NOW)

    assert [r.pattern for r in rotted] == [TASK_NOTIFICATION]


def test_pattern_removed_from_config_is_ignored(engine):
    """History for a pattern the user already deleted is not actionable."""
    engine.settings.whisper_pattern_rot_days = 30
    engine.settings.whisper_synthetic_prompt_patterns = []
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=r"GONE FROM ENV", logged_at=NOW - timedelta(days=60))
    _decision(engine, outcome="injected", matched_pattern=None, logged_at=NOW)

    rotted = find_rotted_patterns(engine.db.conn, engine.settings, NOW)

    assert all(r.pattern != r"GONE FROM ENV" for r in rotted)


def test_find_rotted_patterns_writes_nothing(engine):
    """Detection is a pure read; only task 4's job writes."""
    engine.settings.whisper_pattern_rot_days = 30
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=TASK_NOTIFICATION, logged_at=NOW - timedelta(days=60))
    _decision(engine, outcome="injected", matched_pattern=None, logged_at=NOW)

    find_rotted_patterns(engine.db.conn, engine.settings, NOW)

    count = engine.db.conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
    assert count == 0


def _rot_one_builtin(engine):
    """A rotted <task-notification> plus live traffic — the standard setup.

    Reuses _rotted_history from task 3, so the 2-match minimum stays in one place.
    """
    engine.settings.whisper_pattern_rot_days = 30
    _rotted_history(engine, TASK_NOTIFICATION)


def test_rotted_pattern_creates_one_pending_proposal(engine):
    _rot_one_builtin(engine)

    result = run_synthetic_pattern_monitor(engine, now=NOW)

    assert result == {"rotted": 1, "proposals_created": 1, "proposals_refreshed": 0}
    row = engine.db.conn.execute(
        "SELECT type, status, source_nodes, proposed_action, reason FROM proposals"
    ).fetchone()
    assert row["type"] == "pattern"
    assert row["status"] == "pending"
    assert row["source_nodes"] == "[]"
    assert TASK_NOTIFICATION in row["proposed_action"]


def test_running_twice_does_not_duplicate(engine):
    """The job runs daily and the pattern stays rotted daily."""
    _rot_one_builtin(engine)

    run_synthetic_pattern_monitor(engine, now=NOW)
    second = run_synthetic_pattern_monitor(engine, now=NOW + timedelta(days=1))

    assert second["proposals_created"] == 0
    count = engine.db.conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
    assert count == 1


def test_proposed_action_is_stable_across_days(engine):
    """proposed_action IS the dedup key: a date or count in it would change every
    run, the dedup would never hit, and this would file one proposal per day."""
    _rot_one_builtin(engine)

    run_synthetic_pattern_monitor(engine, now=NOW)
    first = engine.db.conn.execute("SELECT proposed_action FROM proposals").fetchone()[0]
    engine.db.conn.execute("DELETE FROM proposals")
    run_synthetic_pattern_monitor(engine, now=NOW + timedelta(days=9))
    later = engine.db.conn.execute("SELECT proposed_action FROM proposals").fetchone()[0]

    assert first == later


def test_rejected_proposal_is_not_re_proposed(engine):
    """Rejecting means "I know, leave it" — it must not come back tomorrow."""
    _rot_one_builtin(engine)
    run_synthetic_pattern_monitor(engine, now=NOW)
    engine.db.conn.execute("UPDATE proposals SET status = 'rejected'")
    engine.db.conn.commit()

    result = run_synthetic_pattern_monitor(engine, now=NOW + timedelta(days=1))

    assert result["proposals_created"] == 0


def test_builtin_and_operator_get_different_actions(engine):
    """Telling the user to remove from .env a pattern that is not in their .env
    is an instruction impossible to follow."""
    engine.settings.whisper_pattern_rot_days = 30
    engine.settings.whisper_synthetic_prompt_patterns = [r"BATCH JOB"]
    _rotted_history(engine, TASK_NOTIFICATION)
    _rotted_history(engine, r"BATCH JOB")

    run_synthetic_pattern_monitor(engine, now=NOW)

    actions = {
        r["proposed_action"]
        for r in engine.db.conn.execute("SELECT proposed_action FROM proposals").fetchall()
    }
    operator_action = next(a for a in actions if r"BATCH JOB" in a)
    builtin_action = next(a for a in actions if TASK_NOTIFICATION in a)
    assert "ORMAH_WHISPER_SYNTHETIC_PROMPT_PATTERNS" in operator_action
    assert "ORMAH_WHISPER_SYNTHETIC_PROMPT_PATTERNS" not in builtin_action


def test_reason_carries_the_variable_evidence(engine):
    _rot_one_builtin(engine)

    run_synthetic_pattern_monitor(engine, now=NOW)

    reason = engine.db.conn.execute("SELECT reason FROM proposals").fetchone()[0]
    assert (NOW - timedelta(days=60)).isoformat() in reason


def test_a_second_rot_episode_gets_a_fresh_proposal(engine):
    """council I2. Pattern rots, is repaired, resumes matching, rots AGAIN.

    Without `created > last_seen` in the dedup, the historical row would block
    the new episode forever and the second regression would go unreported.
    """
    _rot_one_builtin(engine)
    run_synthetic_pattern_monitor(engine, now=NOW)
    engine.db.conn.execute("UPDATE proposals SET status = 'approved'")
    engine.db.conn.commit()

    # The marker comes back (repaired), fires twice, then goes quiet again.
    later = NOW + timedelta(days=100)
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=TASK_NOTIFICATION, logged_at=later)
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=TASK_NOTIFICATION, logged_at=later + timedelta(days=1))
    much_later = later + timedelta(days=60)
    n = engine.settings.whisper_pattern_rot_min_opportunity
    for i in range(n):
        _decision(engine, outcome="injected", matched_pattern=None,
                  logged_at=much_later - timedelta(minutes=n - i))

    result = run_synthetic_pattern_monitor(engine, now=much_later)

    assert result["proposals_created"] == 1
    count = engine.db.conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
    assert count == 2


def test_abandoned_pending_proposal_is_not_duplicated(engine):
    """An unresolved pending proposal must not be filed twice.

    The pattern rots, a proposal is filed and left pending, the marker comes back,
    then rots again. `created > last_seen` alone stops blocking at that point, so
    without the pending rule the queue would show two identical proposals with
    contradictory reasons (council-pr A1).
    """
    _rot_one_builtin(engine)
    first = run_synthetic_pattern_monitor(engine, now=NOW)
    assert first["proposals_created"] == 1  # left pending on purpose

    # The marker comes back, fires twice, then goes quiet again.
    later = NOW + timedelta(days=100)
    for i in range(2):
        _decision(engine, outcome="silent_synthetic",
                  matched_pattern=TASK_NOTIFICATION, logged_at=later + timedelta(days=i))
    much_later = later + timedelta(days=60)
    for i in range(engine.settings.whisper_pattern_rot_min_opportunity):
        _decision(engine, outcome="injected", matched_pattern=None,
                  logged_at=much_later - timedelta(minutes=i + 1))

    before = engine.db.conn.execute(
        "SELECT reason FROM proposals WHERE type = 'pattern'"
    ).fetchone()["reason"]

    second = run_synthetic_pattern_monitor(engine, now=much_later)

    assert second["proposals_created"] == 0
    total = engine.db.conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE type = 'pattern' AND status = 'pending'"
    ).fetchone()[0]
    assert total == 1

    # Not duplicating is only half of it: the surviving row must describe the
    # SECOND episode, not keep quoting the first (council-pr B1).
    assert second["proposals_refreshed"] == 1
    row = engine.db.conn.execute(
        "SELECT reason, created FROM proposals WHERE type = 'pattern'"
    ).fetchone()
    assert row["reason"] != before
    assert (later + timedelta(days=1)).isoformat() in row["reason"]
    assert row["created"] == much_later.isoformat()


def test_proposed_action_says_the_action_is_manual(engine):
    """council I3. Approving executes nothing, yet the shared proposals surface
    reports success and drops the item — the text must not let the user believe
    the repair happened."""
    _rot_one_builtin(engine)

    run_synthetic_pattern_monitor(engine, now=NOW)

    action = engine.db.conn.execute("SELECT proposed_action FROM proposals").fetchone()[0]
    assert action.startswith("MANUAL ACTION REQUIRED")


def test_no_rot_creates_nothing(engine):
    engine.settings.whisper_pattern_rot_days = 30
    _decision(engine, outcome="silent_synthetic",
              matched_pattern=TASK_NOTIFICATION, logged_at=NOW - timedelta(days=1))

    assert run_synthetic_pattern_monitor(engine, now=NOW) == {
        "rotted": 0, "proposals_created": 0, "proposals_refreshed": 0,
    }


def test_decay_manager_does_not_eat_pattern_proposals(engine):
    """decay_manager.py:20-24 deletes type='decay' proposals on EVERY run,
    unguarded. This pins that 'pattern' is not caught by that DELETE."""
    from ormah.background.decay_manager import run_decay

    _rot_one_builtin(engine)
    run_synthetic_pattern_monitor(engine, now=NOW)

    run_decay(engine)

    count = engine.db.conn.execute(
        "SELECT COUNT(*) FROM proposals WHERE type = 'pattern'"
    ).fetchone()[0]
    assert count == 1


def _settings(**overrides):
    """Settings isolated from the user's global .env (which carries an
    llm_provider this branch rejects — an unrelated ValidationError would make
    these tests pass for the wrong reason)."""
    base = dict(_env_file=None, llm_provider="none", ingest_llm_provider="none")
    base.update(overrides)
    return Settings(**base)


def test_zero_monitor_interval_is_rejected():
    """0 makes APScheduler fire every second, forever, and tracked() logs each
    run as a success — an invisible hot loop (final review, Important)."""
    with pytest.raises(ValidationError, match="whisper pattern monitor settings must be >= 1"):
        _settings(whisper_pattern_monitor_interval_minutes=0)


def test_zero_rot_days_is_rejected():
    """0 used to silently disable the job (rot_days doubled as the guard window)."""
    with pytest.raises(ValidationError, match="whisper pattern monitor settings must be >= 1"):
        _settings(whisper_pattern_rot_days=0)


def test_defaults_are_valid():
    s = _settings()
    assert s.whisper_pattern_rot_days == 30
    assert s.whisper_pattern_monitor_interval_minutes == 1440
    assert s.whisper_pattern_rot_min_matches == 2
    assert s.whisper_pattern_rot_min_opportunity == 50
