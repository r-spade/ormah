import sqlite3
from datetime import datetime, timedelta, timezone

from ormah.engine.whisper_health import compute_whisper_health

NOW = datetime(2026, 6, 24, tzinfo=timezone.utc)
ISO = NOW.isoformat()


def _db() -> sqlite3.Connection:
    # isolation_level=None mirrors the production Database (autocommit; the
    # whisper-health read opens its own BEGIN DEFERRED snapshot explicitly).
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE whisper_log "
        "(id INTEGER PRIMARY KEY, was_injected INTEGER, logged_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE affinity "
        "(whisper_log_id INTEGER, signal INTEGER, confirmed_at TEXT)"
    )
    return conn


def _inject(conn, wid, when=ISO, injected=1):
    conn.execute(
        "INSERT INTO whisper_log (id, was_injected, logged_at) VALUES (?, ?, ?)",
        (wid, injected, when),
    )


def _feedback(conn, wid, signal, when=ISO):
    conn.execute(
        "INSERT INTO affinity (whisper_log_id, signal, confirmed_at) VALUES (?, ?, ?)",
        (wid, signal, when),
    )


def test_empty_store_ratios_none():
    out = compute_whisper_health(_db(), NOW)
    for window in ("all_time", "last_7d"):
        assert out[window]["coverage"] is None
        assert out[window]["precision"] is None
        assert out[window]["injected"] == 0
    assert out["all_time"]["unlinked_feedback_rows"] == 0


def test_injection_without_feedback():
    conn = _db()
    _inject(conn, 1)
    _inject(conn, 2)
    out = compute_whisper_health(conn, NOW)["all_time"]
    assert out["injected"] == 2
    assert out["coverage"] == 0.0
    assert out["precision"] is None


def test_mixed_signals_precision():
    conn = _db()
    for wid in (1, 2, 3, 4):
        _inject(conn, wid)
    _feedback(conn, 1, 1)
    _feedback(conn, 2, 1)
    _feedback(conn, 3, 1)
    _feedback(conn, 4, -1)
    out = compute_whisper_health(conn, NOW)["all_time"]
    assert out["precision"] == 0.75
    assert out["coverage"] == 1.0


def test_distinct_guards_against_double_count():
    # In production idx_affinity_node_whisper_log_unique (schema.sql:126) forbids two
    # affinity rows on one whisper_log_id, so this two-row shape can't occur for
    # real. The minimal schema here omits that index on purpose, to assert the
    # DISTINCT clause is a defensive guard that keeps coverage <= 1.0 regardless.
    conn = _db()
    _inject(conn, 1)
    _feedback(conn, 1, 1)
    _feedback(conn, 1, -1)
    out = compute_whisper_health(conn, NOW)["all_time"]
    assert out["feedback_rows"] == 1
    assert out["coverage"] == 1.0  # not 2.0


def test_held_back_candidate_feedback_excluded():
    # C1: feedback on a was_injected=0 candidate must NOT inflate coverage.
    conn = _db()
    _inject(conn, 1, injected=1)
    _feedback(conn, 1, 1)
    _inject(conn, 2, injected=0)  # held-back candidate
    _feedback(conn, 2, 1)         # later converted to affinity
    out = compute_whisper_health(conn, NOW)["all_time"]
    assert out["injected"] == 1
    assert out["feedback_rows"] == 1
    assert out["coverage"] == 1.0  # not 2.0
    assert out["positive"] == 1   # held-back signal excluded from precision too


def test_legacy_null_whisper_log_id_surfaced_not_counted():
    # I1 (council r2): pre-#21 affinity rows carry whisper_log_id = NULL. They are
    # excluded from linked-only coverage/precision but surfaced via
    # unlinked_feedback_rows so the loss is visible, not silent.
    conn = _db()
    _inject(conn, 1)
    _feedback(conn, 1, 1)
    conn.execute(
        "INSERT INTO affinity (whisper_log_id, signal, confirmed_at) "
        "VALUES (NULL, 1, ?)",
        (ISO,),
    )
    out = compute_whisper_health(conn, NOW)["all_time"]
    assert out["coverage"] == 1.0            # linked-only, NULL row ignored
    assert out["positive"] == 1              # NULL row excluded from precision
    assert out["unlinked_feedback_rows"] == 1  # but counted and exposed


def test_last_7d_old_injection_recent_feedback():
    # I1 (r1): recent feedback for an old injection must not push last_7d above 1.0.
    conn = _db()
    old = (NOW - timedelta(days=10)).isoformat()
    _inject(conn, 1, when=old)
    _feedback(conn, 1, 1, when=ISO)  # feedback today
    out = compute_whisper_health(conn, NOW)
    assert out["all_time"]["coverage"] == 1.0
    assert out["last_7d"]["injected"] == 0
    assert out["last_7d"]["feedback_rows"] == 0
    assert out["last_7d"]["coverage"] is None


def test_mixed_confirmed_at_format_still_counted():
    # I2 (r1): confirmed_at in datetime('now') space-format must not be dropped,
    # because the window filters wl.logged_at, never confirmed_at.
    conn = _db()
    _inject(conn, 1, when=ISO)
    _feedback(conn, 1, 1, when="2026-06-24 00:00:00")  # space-format, no TZ
    out = compute_whisper_health(conn, NOW)["last_7d"]
    assert out["feedback_rows"] == 1
    assert out["coverage"] == 1.0


def test_reads_under_snapshot_no_transaction_leak():
    # Council-PR I1: the aggregates run inside one BEGIN DEFERRED snapshot so a
    # concurrent insert can't push coverage above 100%. Verify the snapshot is
    # opened and closed cleanly (no dangling transaction on the connection).
    conn = _db()
    _inject(conn, 1)
    _feedback(conn, 1, 1)
    out = compute_whisper_health(conn, NOW)["all_time"]
    assert out["coverage"] == 1.0
    assert conn.in_transaction is False


def test_seven_day_cutoff():
    conn = _db()
    old = (NOW - timedelta(days=10)).isoformat()
    _inject(conn, 1, when=old)
    _feedback(conn, 1, 1, when=old)
    out = compute_whisper_health(conn, NOW)
    assert out["all_time"]["injected"] == 1
    assert out["all_time"]["coverage"] == 1.0
    assert out["last_7d"]["injected"] == 0
    assert out["last_7d"]["coverage"] is None
    assert out["last_7d"]["precision"] is None


def test_stats_exposes_whisper_health(engine):
    out = engine.stats()
    assert "feedback_health" in out["whisper"]
    wh = out["whisper"]["feedback_health"]
    assert set(wh) == {"all_time", "last_7d"}
    assert set(wh["last_7d"]) == {
        "injected", "feedback_rows", "coverage",
        "positive", "negative", "precision",
    }
    assert set(wh["all_time"]) == {
        "injected", "feedback_rows", "coverage",
        "positive", "negative", "precision", "unlinked_feedback_rows",
    }
    assert wh["all_time"]["injected"] == 0
    assert wh["all_time"]["coverage"] is None


def test_stats_whisper_health_seeded(engine):
    # I2 (council r2): exercise real schema (NOT NULL cols + JOIN), not just empty.
    from datetime import datetime, timezone

    conn = engine.db.conn
    conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, prompt_hash, prompt_vec, node_id, score, was_injected, logged_at) "
        "VALUES ('s1', 'h1', X'00', 'n1', 0.5, 1, ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    wid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO affinity "
        "(prompt_vec, node_id, signal, source, confirmed_at, session_id, whisper_log_id) "
        "VALUES (X'00', 'n1', 1, 'explicit', datetime('now'), 's1', ?)",
        (wid,),
    )
    wh = engine.stats()["whisper"]["feedback_health"]["all_time"]
    assert wh["injected"] == 1
    assert wh["feedback_rows"] == 1
    assert wh["coverage"] == 1.0
    assert wh["precision"] == 1.0


def test_exact_feedback_on_older_injected_event_counts_in_health(engine):
    conn = engine.db.conn
    node_id = "health-node-1"
    conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, prompt_hash, prompt_vec, node_id, score, was_injected, logged_at) "
        "VALUES ('sess-a', 'hash-a', X'00', ?, 0.8, 1, ?)",
        (node_id, datetime.now(timezone.utc).isoformat()),
    )
    injected_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, prompt_hash, prompt_vec, node_id, score, was_injected, logged_at) "
        "VALUES ('sess-b', 'hash-b', X'01', ?, 0.3, 0, ?)",
        (node_id, datetime.now(timezone.utc).isoformat()),
    )

    result = engine.submit_feedback(
        node_id, 1, "explicit", whisper_log_id=injected_id,
    )

    assert "Feedback recorded" in result
    wh = engine.stats()["whisper"]["feedback_health"]["all_time"]
    assert wh["injected"] == 1
    assert wh["feedback_rows"] == 1
    assert wh["coverage"] == 1.0
    assert wh["precision"] == 1.0
