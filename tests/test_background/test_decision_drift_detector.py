"""Tests for decision drift detection background job."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from ormah.background.decision_drift_detector import (
    _commits_since,
    _already_alerted,
    _update_alert,
    _create_drift_observation,
    run_decision_drift_detection,
)
from ormah.models.node import CreateNodeRequest, NodeType, Tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision(engine, title: str, file_tags: list[str], days_ago: int = 10) -> str:
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    node_id, _ = engine.remember(
        CreateNodeRequest(
            content=f"We decided to {title}.",
            type=NodeType.decision,
            tier=Tier.core,
            title=title,
            tags=file_tags,
        ),
        agent_id="test",
    )
    # Backdate created timestamp to simulate an old decision
    with engine.db.transaction() as conn:
        conn.execute("UPDATE nodes SET created = ? WHERE id = ?", (created, node_id))
    return node_id


def _observation_count(engine) -> int:
    return engine.db.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE type = 'observation' AND title LIKE 'Decision drift:%'"
    ).fetchone()[0]


def _alert_row(engine, decision_id: str, file_path: str):
    return engine.db.conn.execute(
        "SELECT * FROM decision_drift_alerts WHERE decision_id = ? AND file_path = ?",
        (decision_id, file_path),
    ).fetchone()


# ---------------------------------------------------------------------------
# Unit: _commits_since
# ---------------------------------------------------------------------------

def _fake_run_commits(n: int):
    def _run(*args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "\n".join(f"abc{i} some change" for i in range(n))
        return m
    return _run


def test_commits_since_counts_correctly():
    with patch("subprocess.run", side_effect=_fake_run_commits(7)):
        count = _commits_since(MagicMock(), "src/foo.py", "2026-01-01T00:00:00")
    assert count == 7


def test_commits_since_returns_zero_on_failure():
    def _fail(*args, **kwargs):
        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        return m

    with patch("subprocess.run", side_effect=_fail):
        count = _commits_since(MagicMock(), "src/foo.py", "2026-01-01T00:00:00")
    assert count == 0


def test_commits_since_returns_zero_on_exception():
    with patch("subprocess.run", side_effect=Exception("boom")):
        count = _commits_since(MagicMock(), "src/foo.py", "2026-01-01T00:00:00")
    assert count == 0


# ---------------------------------------------------------------------------
# Unit: alert tracking
# ---------------------------------------------------------------------------

def test_already_alerted_false_initially(engine):
    engine.settings.contradiction_prestorage_enabled = False
    nid = _make_decision(engine, "Use SQLite for storage", ["file:src/db.py"])
    assert not _already_alerted(engine, nid, "src/db.py")


def test_already_alerted_true_after_update(engine):
    engine.settings.contradiction_prestorage_enabled = False
    nid = _make_decision(engine, "Use Redis for caching", ["file:src/cache.py"])

    # Ensure table exists by running the job once (disabled so no alerts created)
    engine.settings.decision_drift_enabled = False
    run_decision_drift_detection(engine)
    engine.settings.decision_drift_enabled = True

    from ormah.background.decision_drift_detector import _ensure_table
    _ensure_table(engine)

    _update_alert(engine, nid, "src/cache.py", 8, "fake-obs-id")
    assert _already_alerted(engine, nid, "src/cache.py")


# ---------------------------------------------------------------------------
# Unit: _create_drift_observation
# ---------------------------------------------------------------------------

def test_create_drift_observation_returns_id_and_creates_node(engine):
    engine.settings.contradiction_prestorage_enabled = False
    nid = _make_decision(engine, "Use JWT for auth", ["file:src/auth.py"])

    node_dict = {
        "id": nid,
        "title": "Use JWT for auth",
        "created": "2026-01-01T00:00:00",
        "space": None,
    }

    from ormah.background.decision_drift_detector import _ensure_table
    _ensure_table(engine)

    obs_id = _create_drift_observation(engine, node_dict, "src/auth.py", commits=12)
    assert obs_id

    obs = engine.graph.get_node(obs_id)
    assert obs is not None
    assert "drift" in obs["title"].lower() or "stale" in obs["title"].lower()
    assert obs["type"] == "observation"


# ---------------------------------------------------------------------------
# Integration: run_decision_drift_detection
# ---------------------------------------------------------------------------

def _fake_subprocess(repo_commits: int):
    """Returns a subprocess.run side_effect that fakes git rev-parse + git log."""
    def _run(args, **kwargs):
        m = MagicMock()
        if "rev-parse" in args:
            m.returncode = 0
            m.stdout = "/repo\n"
        elif "log" in args:
            m.returncode = 0
            m.stdout = "\n".join(f"abc{i} change" for i in range(repo_commits))
        else:
            m.returncode = 1
            m.stdout = ""
        return m
    return _run


def test_run_creates_observation_when_threshold_exceeded(engine):
    engine.settings.decision_drift_enabled = True
    engine.settings.decision_drift_churn_threshold = 3
    engine.settings.contradiction_prestorage_enabled = False

    _make_decision(engine, "Use PostgreSQL for main DB", ["file:src/models.py"], days_ago=30)

    with patch("subprocess.run", side_effect=_fake_subprocess(repo_commits=5)):
        run_decision_drift_detection(engine)

    assert _observation_count(engine) == 1


def test_run_skips_when_below_threshold(engine):
    engine.settings.decision_drift_enabled = True
    engine.settings.decision_drift_churn_threshold = 10
    engine.settings.contradiction_prestorage_enabled = False

    _make_decision(engine, "Use Celery for task queue", ["file:src/tasks.py"], days_ago=20)

    with patch("subprocess.run", side_effect=_fake_subprocess(repo_commits=3)):
        run_decision_drift_detection(engine)

    assert _observation_count(engine) == 0


def test_run_skips_decisions_without_file_tags(engine):
    engine.settings.decision_drift_enabled = True
    engine.settings.decision_drift_churn_threshold = 2
    engine.settings.contradiction_prestorage_enabled = False

    # Decision with no file: tags — should be ignored
    engine.remember(
        CreateNodeRequest(
            content="We decided to use blue as primary color.",
            type=NodeType.decision,
            title="Use blue color palette",
            tags=["design", "ui"],
        ),
        agent_id="test",
    )

    with patch("subprocess.run", side_effect=_fake_subprocess(repo_commits=50)):
        run_decision_drift_detection(engine)

    assert _observation_count(engine) == 0


def test_run_skipped_when_disabled(engine):
    engine.settings.decision_drift_enabled = False

    _make_decision(engine, "Use gRPC for service comms", ["file:src/grpc_client.py"])

    with patch("subprocess.run") as mock_run:
        run_decision_drift_detection(engine)
        mock_run.assert_not_called()

    assert _observation_count(engine) == 0


def test_run_suppresses_repeat_alerts_within_refire_window(engine):
    engine.settings.decision_drift_enabled = True
    engine.settings.decision_drift_churn_threshold = 2
    engine.settings.decision_drift_refire_days = 30
    engine.settings.contradiction_prestorage_enabled = False

    nid = _make_decision(engine, "Use S3 for file storage", ["file:src/storage.py"], days_ago=60)

    with patch("subprocess.run", side_effect=_fake_subprocess(repo_commits=10)):
        run_decision_drift_detection(engine)  # first run — creates alert

    assert _observation_count(engine) == 1

    with patch("subprocess.run", side_effect=_fake_subprocess(repo_commits=15)):
        run_decision_drift_detection(engine)  # second run — suppressed

    assert _observation_count(engine) == 1  # still just one


def test_run_fires_again_after_refire_window(engine):
    engine.settings.decision_drift_enabled = True
    engine.settings.decision_drift_churn_threshold = 2
    engine.settings.decision_drift_refire_days = 0  # no suppression
    engine.settings.contradiction_prestorage_enabled = False

    _make_decision(engine, "Use Nginx as reverse proxy", ["file:src/nginx.conf"], days_ago=90)

    with patch("subprocess.run", side_effect=_fake_subprocess(repo_commits=10)):
        run_decision_drift_detection(engine)
        run_decision_drift_detection(engine)  # refire_days=0 → fires again

    assert _observation_count(engine) == 2


def test_run_handles_no_git_repo(engine):
    engine.settings.decision_drift_enabled = True
    engine.settings.contradiction_prestorage_enabled = False

    _make_decision(engine, "Use Docker for containers", ["file:src/Dockerfile"])

    def _no_repo(args, **kwargs):
        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        return m

    with patch("subprocess.run", side_effect=_no_repo):
        run_decision_drift_detection(engine)  # should not raise

    assert _observation_count(engine) == 0
