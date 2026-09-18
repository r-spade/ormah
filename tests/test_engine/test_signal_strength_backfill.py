"""The #218 backfill recomputes historical strength exactly, and only once."""

import json

import pytest

from ormah import signal_strength as ss


def _seed(engine, *, source, polarity, evidence, strength=0.85):
    """Insert a signals row with a stale strength.

    whisper_log_id stays NULL: the unique index on
    (whisper_log_id, signal_type, source) is partial on whisper_log_id IS NOT NULL,
    so NULL rows never collide however many are seeded.
    """
    cursor = engine.db.conn.execute(
        "INSERT INTO signals "
        "(whisper_log_id, node_id, signal_type, polarity, strength, source, evidence, created) "
        "VALUES (NULL, 'seed-node', 'seeded', ?, ?, ?, ?, datetime('now'))",
        (polarity, strength, source, json.dumps(evidence) if evidence is not None else None),
    )
    engine.db.conn.commit()
    return cursor.lastrowid


def _rerun(engine):
    """Clear both markers so the migration makes a full pass over the table."""
    engine.db.conn.execute(
        "DELETE FROM meta WHERE key IN "
        "('signal_strength_ladder_version', 'signal_strength_ladder_cutoff')"
    )
    engine.db.conn.commit()
    engine._migrate_signal_strength()


def _row(engine, signal_id):
    return engine.db.conn.execute(
        "SELECT strength, polarity, evidence FROM signals WHERE id = ?", (signal_id,)
    ).fetchone()


def test_backfill_recomputes_each_channel_exactly(engine):
    verbatim = _seed(
        engine, source=ss.HEURISTIC_SOURCE, polarity=1, evidence={"match": "node_id"}
    )
    overlap = _seed(
        engine,
        source=ss.HEURISTIC_SOURCE,
        polarity=1,
        evidence={"match": "token_overlap", "overlap_ratio": 1.167},
    )
    judged = _seed(
        engine,
        source=ss.LLM_JUDGE_SOURCE,
        polarity=1,
        evidence={"confidence": 0.88, "min_confidence": 0.75},
    )
    implicit = _seed(engine, source="implicit", polarity=1, evidence={"source": "implicit"},
                     strength=1.0)

    _rerun(engine)

    assert _row(engine, verbatim)["strength"] == ss.VERBATIM_NODE_ID
    assert _row(engine, overlap)["strength"] == pytest.approx(ss.token_overlap_strength(1.167))
    assert _row(engine, judged)["strength"] == pytest.approx(ss.judge_strength(0.88, 0.75, 1))
    assert _row(engine, implicit)["strength"] == ss.IMPLICIT


def test_backfill_uses_the_rows_own_min_confidence(engine):
    """Not today's setting — the judge stamped it on the row when it wrote it."""
    lenient = _seed(
        engine, source=ss.LLM_JUDGE_SOURCE, polarity=1,
        evidence={"confidence": 0.80, "min_confidence": 0.75},
    )
    strict = _seed(
        engine, source=ss.LLM_JUDGE_SOURCE, polarity=1,
        evidence={"confidence": 0.80, "min_confidence": 0.80},
    )

    _rerun(engine)

    assert _row(engine, lenient)["strength"] > _row(engine, strict)["strength"]


def test_backfill_zeroes_rows_that_assert_nothing(engine):
    uncertain = _seed(
        engine, source=ss.LLM_JUDGE_SOURCE, polarity=0,
        evidence={"confidence": 0.35, "min_confidence": 0.75},
    )

    _rerun(engine)

    assert _row(engine, uncertain)["strength"] == 0.0


def test_backfill_survives_missing_evidence(engine):
    orphan = _seed(engine, source=ss.HEURISTIC_SOURCE, polarity=1, evidence=None)

    _rerun(engine)

    assert _row(engine, orphan)["strength"] == ss.UNKNOWN


def test_backfill_leaves_evidence_and_polarity_untouched(engine):
    evidence = {"match": "token_overlap", "overlap_ratio": 1.167}
    signal_id = _seed(engine, source=ss.HEURISTIC_SOURCE, polarity=1, evidence=evidence)
    before = _row(engine, signal_id)

    _rerun(engine)

    after = _row(engine, signal_id)
    assert after["evidence"] == before["evidence"]
    assert after["polarity"] == before["polarity"]
    assert after["strength"] != before["strength"]


def test_backfill_is_idempotent(engine):
    signal_id = _seed(
        engine,
        source=ss.HEURISTIC_SOURCE,
        polarity=1,
        evidence={"match": "token_overlap", "overlap_ratio": 1.167},
    )

    _rerun(engine)
    once = _row(engine, signal_id)["strength"]
    _rerun(engine)
    twice = _row(engine, signal_id)["strength"]

    assert once == twice


def test_rows_at_or_below_the_cutoff_are_not_rescanned(engine):
    """The cutoff, not the recompute, is what keeps a later start cheap."""
    signal_id = _seed(
        engine, source=ss.HEURISTIC_SOURCE, polarity=1, evidence={"match": "node_id"}
    )
    _rerun(engine)

    engine.db.conn.execute("UPDATE signals SET strength = 0.123 WHERE id = ?", (signal_id,))
    engine.db.conn.commit()
    engine._migrate_signal_strength()

    assert _row(engine, signal_id)["strength"] == 0.123


def test_rescan_repairs_a_legacy_row_written_after_the_stamp(engine):
    """A one-time stamp cannot repair what an old binary writes after it.

    A rollback then re-upgrade, or the second unmanaged server process this project
    already knows about (#238), inserts pre-ladder values into a table the stamp
    declares migrated. Council peer finding, /council-pr of 2026-08-26.
    """
    _rerun(engine)

    legacy = _seed(
        engine,
        source=ss.HEURISTIC_SOURCE,
        polarity=1,
        evidence={"match": "node_id"},
        strength=1.0,
    )
    engine._migrate_signal_strength()

    assert _row(engine, legacy)["strength"] == ss.VERBATIM_NODE_ID


def test_rescan_leaves_an_already_correct_row_untouched(engine):
    """The repair is a repair only because a current-code row recomputes to itself.

    strength_from_evidence is a pure function of (source, polarity, evidence), and
    every write site stores exactly what it returns for what it recorded. Without
    that, rescanning would drift correct rows instead of confirming them.
    """
    _rerun(engine)

    correct = _seed(
        engine,
        source=ss.HEURISTIC_SOURCE,
        polarity=1,
        evidence={"match": "token_overlap", "overlap_ratio": 0.667},
        strength=ss.token_overlap_strength(0.667),
    )
    before = _row(engine, correct)["strength"]
    engine._migrate_signal_strength()

    assert _row(engine, correct)["strength"] == before


def test_the_cutoff_advances_after_a_repair(engine):
    """Each start covers only what arrived since the last one."""
    _rerun(engine)
    legacy = _seed(
        engine, source="implicit", polarity=1, evidence={"source": "implicit"}, strength=1.0
    )
    engine._migrate_signal_strength()
    assert _row(engine, legacy)["strength"] == ss.IMPLICIT

    engine.db.conn.execute("UPDATE signals SET strength = 0.123 WHERE id = ?", (legacy,))
    engine.db.conn.commit()
    engine._migrate_signal_strength()

    assert _row(engine, legacy)["strength"] == 0.123, "the cutoff did not advance"


def test_a_poisoned_evidence_row_cannot_block_startup(engine):
    """One malformed historical value must not stop the server from ever booting.

    The migration runs at every startup and writes into signals.strength, which is
    REAL NOT NULL. SQLite stores a NaN REAL as NULL, so binding one raises
    IntegrityError — and with an every-boot migration that failure is permanent,
    not a one-time hiccup. Codex peer finding, /council-pr of 2026-08-26.
    """
    poisoned = _seed(
        engine,
        source=ss.HEURISTIC_SOURCE,
        polarity=1,
        evidence={"match": "token_overlap", "overlap_ratio": float("nan")},
    )
    huge = _seed(
        engine,
        source=ss.LLM_JUDGE_SOURCE,
        polarity=1,
        evidence={"confidence": int("9" * 400), "min_confidence": 0.75},
    )

    _rerun(engine)          # must not raise
    engine._migrate_signal_strength()   # and the next boot must not raise either

    for signal_id, lo, hi in (
        (poisoned, ss.OVERLAP_FLOOR, ss.OVERLAP_FLOOR + ss.OVERLAP_SPAN),
        (huge, ss.JUDGE_LO, ss.JUDGE_HI),
    ):
        strength = _row(engine, signal_id)["strength"]
        assert strength is not None
        assert lo <= strength <= hi
