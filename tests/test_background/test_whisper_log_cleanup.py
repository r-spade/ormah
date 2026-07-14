"""Tests for normalized whisper payload retention."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ormah.background.whisper_log_cleanup import run_whisper_log_cleanup
from ormah.config import Settings


NOW = datetime(2026, 7, 12, tzinfo=timezone.utc)


def _event(engine, suffix: str, *, age_days: int, injected: int) -> tuple[int, int]:
    logged_at = (NOW - timedelta(days=age_days)).isoformat()
    with engine.db.transaction() as conn:
        event_id = engine.db.insert_retrieval_event(
            conn,
            surface="whisper",
            session_id=f"session-{suffix}",
            space="ormah",
            prompt_hash=f"hash-{suffix}",
            prompt_text=f"prompt {suffix}",
            prompt_vec=b"v" * 3072,
            logged_at=logged_at,
        )
        cursor = conn.execute(
            """
            INSERT INTO whisper_log
                (session_id, space, prompt_hash, prompt_text, prompt_vec,
                 node_id, score, was_injected, logged_at, retrieval_event_id)
            VALUES (?, 'ormah', ?, NULL, X'', ?, 0.5, ?, ?, ?)
            """,
            (
                f"session-{suffix}",
                f"hash-{suffix}",
                f"node-{suffix}",
                injected,
                logged_at,
                event_id,
            ),
        )
    return int(cursor.lastrowid), event_id


def test_cleanup_deletes_only_stale_unreferenced_rejections(engine):
    engine.settings.whisper_log_rejected_retention_days = 30
    engine.settings.whisper_log_cleanup_batch_size = 100
    stale_id, stale_event = _event(engine, "stale", age_days=31, injected=0)
    recent_id, _ = _event(engine, "recent", age_days=29, injected=0)
    injected_id, _ = _event(engine, "injected", age_days=90, injected=1)
    referenced_id, _ = _event(engine, "referenced", age_days=90, injected=0)
    signaled_id, _ = _event(engine, "signaled", age_days=90, injected=0)
    with engine.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO affinity
                (prompt_vec, prompt_text, node_id, signal, source,
                 confirmed_at, session_id, whisper_log_id)
            VALUES (X'', 'prompt', 'node-referenced', 1, 'explicit', ?,
                    'session-referenced', ?)
            """,
            (NOW.isoformat(), referenced_id),
        )
        conn.execute(
            """
            INSERT INTO signals
                (whisper_log_id, node_id, signal_type, polarity, strength,
                 source, session_id, created)
            VALUES (?, 'node-signaled', 'feedback_submitted', -1, 1.0,
                    'implicit', 'session-signaled', ?)
            """,
            (signaled_id, NOW.isoformat()),
        )

    result = run_whisper_log_cleanup(engine, now=NOW)

    assert result == {"candidate_rows_deleted": 1, "events_deleted": 1}
    remaining = {
        row["id"] for row in engine.db.conn.execute("SELECT id FROM whisper_log").fetchall()
    }
    assert stale_id not in remaining
    assert recent_id in remaining
    assert injected_id in remaining
    assert referenced_id in remaining
    assert signaled_id in remaining
    assert engine.db.conn.execute(
        "SELECT 1 FROM retrieval_events WHERE id = ?", (stale_event,)
    ).fetchone() is None


def test_cleanup_is_bounded_and_idempotent(engine):
    engine.settings.whisper_log_rejected_retention_days = 30
    engine.settings.whisper_log_cleanup_batch_size = 2
    for i in range(3):
        _event(engine, f"old-{i}", age_days=60, injected=0)

    first = run_whisper_log_cleanup(engine, now=NOW)
    second = run_whisper_log_cleanup(engine, now=NOW)
    third = run_whisper_log_cleanup(engine, now=NOW)

    assert first["candidate_rows_deleted"] == 2
    assert second["candidate_rows_deleted"] == 1
    assert third == {"candidate_rows_deleted": 0, "events_deleted": 0}


def test_normalized_event_stores_one_vector_for_many_candidates(engine):
    payload = b"v" * 3072
    with engine.db.transaction() as conn:
        event_id = engine.db.insert_retrieval_event(
            conn,
            surface="whisper",
            session_id="storage-session",
            space="ormah",
            prompt_hash="storage-hash",
            prompt_text="storage prompt",
            prompt_vec=payload,
            logged_at=NOW.isoformat(),
        )
        for i in range(40):
            conn.execute(
                """
                INSERT INTO whisper_log
                    (session_id, prompt_hash, prompt_vec, node_id, score,
                     was_injected, logged_at, retrieval_event_id)
                VALUES ('storage-session', 'storage-hash', X'', ?, 0.5, 0, ?, ?)
                """,
                (f"node-{i}", NOW.isoformat(), event_id),
            )

    candidate_bytes = engine.db.conn.execute(
        "SELECT SUM(length(prompt_vec)) FROM whisper_log WHERE retrieval_event_id = ?",
        (event_id,),
    ).fetchone()[0]
    event_bytes = engine.db.conn.execute(
        "SELECT length(prompt_vec) FROM retrieval_events WHERE id = ?",
        (event_id,),
    ).fetchone()[0]
    assert candidate_bytes == 0
    assert event_bytes == 3072

    plan = engine.db.conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM whisper_log "
        "WHERE was_injected = 0 AND logged_at < ? ORDER BY logged_at LIMIT 1000",
        ((NOW - timedelta(days=30)).isoformat(),),
    ).fetchall()
    assert any("idx_whisper_log_retention" in row["detail"] for row in plan)

    session_plan = engine.db.conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM whisper_log "
        "WHERE session_id = ? AND was_injected = 1",
        ("storage-session",),
    ).fetchall()
    assert any("idx_whisper_log_session_injected" in row["detail"] for row in session_plan)

    node_plan = engine.db.conn.execute(
        "EXPLAIN QUERY PLAN SELECT 1 FROM whisper_log "
        "WHERE node_id = ? AND was_injected = 1 AND logged_at > ?",
        ("node-1", (NOW - timedelta(days=7)).isoformat()),
    ).fetchall()
    assert any("idx_whisper_log_node_injected_logged" in row["detail"] for row in node_plan)


def test_scheduler_registers_whisper_log_cleanup(tmp_path, monkeypatch):
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
        whisper_log_cleanup_interval_hours=12,
    )
    engine = SimpleNamespace(
        settings=settings,
        builder=SimpleNamespace(incremental_update=lambda: (0, 0)),
    )

    scheduler, _tracker = scheduler_module.start_scheduler(engine)

    job = next(job for job in scheduler.jobs if job["id"] == "whisper_log_cleanup")
    assert job["trigger"] == "interval"
    assert job["hours"] == 12
