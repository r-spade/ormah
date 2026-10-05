"""submit_feedback records a real strength, not a hardcoded 1.0 (#218)."""

import pytest

from ormah import signal_strength as ss
from ormah.models.node import CreateNodeRequest


def _node_with_whisper(engine, title):
    """Create a node and a whisper_log row submit_feedback can resolve."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content="caching architecture note for the strength ladder",
        title=title,
        type="fact",
        tier="working",
    ))
    engine.recall_search("caching architecture", limit=10)
    row = engine.db.conn.execute(
        "SELECT id FROM whisper_log WHERE node_id = ? ORDER BY id DESC LIMIT 1",
        (node_id,),
    ).fetchone()
    assert row is not None, "no whisper_log row was created — check the surface used"
    return node_id, row["id"]


def _stored_strength(engine, whisper_log_id):
    return engine.db.conn.execute(
        "SELECT strength FROM signals WHERE whisper_log_id = ? "
        "AND signal_type = 'feedback_submitted'",
        (whisper_log_id,),
    ).fetchone()["strength"]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("explicit", ss.EXPLICIT),
        ("implicit", ss.IMPLICIT),
        ("auto_heuristic", ss.UNKNOWN),
    ],
)
def test_each_feedback_source_records_its_own_rung(engine, source, expected):
    node_id, log_id = _node_with_whisper(engine, f"Caching {source}")
    engine.submit_feedback(node_id, signal=1, source=source, whisper_log_id=log_id)
    assert _stored_strength(engine, log_id) == expected


def test_explicit_and_implicit_no_longer_collide(engine):
    """Both used to store exactly 1.0, leaving source as the only discriminator."""
    explicit_id, explicit_log = _node_with_whisper(engine, "Caching E")
    implicit_id, implicit_log = _node_with_whisper(engine, "Caching I")
    engine.submit_feedback(explicit_id, signal=1, source="explicit", whisper_log_id=explicit_log)
    engine.submit_feedback(implicit_id, signal=1, source="implicit", whisper_log_id=implicit_log)
    assert _stored_strength(engine, explicit_log) != _stored_strength(engine, implicit_log)


def test_negative_feedback_keeps_its_channel_rung(engine):
    """strength is the evidence for THIS row's polarity, so -1 is not weaker."""
    node_id, log_id = _node_with_whisper(engine, "Caching N")
    engine.submit_feedback(node_id, signal=-1, source="explicit", whisper_log_id=log_id)
    assert _stored_strength(engine, log_id) == ss.EXPLICIT
