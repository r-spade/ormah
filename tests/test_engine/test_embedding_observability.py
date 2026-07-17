"""stats() must expose the embedding gap + schema version, and recovery must
prove out end-to-end via the registered job (#32, council I3)."""
from __future__ import annotations

from ormah.engine.memory_engine import _EMBEDDING_SCHEMA_VERSION
from ormah.models.node import CreateNodeRequest


def _set_schema_current(engine):
    with engine.db.transaction() as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES "
                     "('embedding_schema_version', ?)", (str(_EMBEDDING_SCHEMA_VERSION),))


def test_stats_exposes_embedding_gap_and_version(engine):
    nid, _ = engine.remember(CreateNodeRequest(title="x", content="y"))
    _set_schema_current(engine)
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid,))
    s = engine.stats()
    assert s["embedding_gap"] >= 1
    assert s["embedding_schema_version"] == _EMBEDDING_SCHEMA_VERSION
    assert "vec_count" in s


def test_e2e_gap_recovers_via_registered_job(engine):
    from ormah.background.scheduler import start_scheduler
    from ormah.background.embedding_backfill import run_embedding_backfill
    nid, _ = engine.remember(CreateNodeRequest(title="recover", content="me"))
    _set_schema_current(engine)
    with engine.db.transaction() as conn:
        conn.execute("DELETE FROM node_vectors WHERE id = ?", (nid,))
    assert engine.stats()["embedding_gap"] >= 1
    scheduler, _t = start_scheduler(engine)
    try:
        assert scheduler.get_job("embedding_backfill") is not None
        run_embedding_backfill(engine)
    finally:
        scheduler.shutdown(wait=False)
    assert engine.stats()["embedding_gap"] == 0
