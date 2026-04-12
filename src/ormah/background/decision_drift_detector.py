"""Decision Drift Detection: flag stale [decision] memories when associated code churns.

A decision node is considered stale when the files it references (via ``file:<path>``
tags) have accumulated >= N commits since the decision was created without a
corresponding memory update. Staleness is surfaced as a new ``[observation]`` node
linked to the decision, and whispered on the next session.

Scope constraint: only decisions with explicit ``file:<path>`` tags are checked.
No automatic file association is attempted.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS decision_drift_alerts (
    decision_id     TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    commits_since   INTEGER NOT NULL DEFAULT 0,
    alerted_at      TEXT NOT NULL,
    observation_id  TEXT,
    PRIMARY KEY (decision_id, file_path)
);
"""

_FILE_TAG_PREFIX = "file:"


def _ensure_table(engine) -> None:
    with engine.db.transaction() as conn:
        conn.execute(_TABLE_DDL)


def _git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def _commits_since(repo_root: Path, file_path: str, since_iso: str) -> int:
    """Count commits touching file_path since a given ISO datetime."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                f"--after={since_iso}",
                "--oneline",
                "--follow",
                "--diff-filter=M",
                "--",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return 0
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        return len(lines)
    except Exception as e:
        logger.debug("git log failed for %s: %s", file_path, e)
        return 0


def _already_alerted(engine, decision_id: str, file_path: str) -> bool:
    _ensure_table(engine)
    row = engine.db.conn.execute(
        "SELECT 1 FROM decision_drift_alerts WHERE decision_id = ? AND file_path = ?",
        (decision_id, file_path),
    ).fetchone()
    return row is not None


def _update_alert(engine, decision_id: str, file_path: str, commits: int, obs_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with engine.db.transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO decision_drift_alerts
                (decision_id, file_path, commits_since, alerted_at, observation_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (decision_id, file_path, commits, now, obs_id),
        )


def _create_drift_observation(engine, decision_node: dict, file_path: str, commits: int) -> str:
    """Create an [observation] node flagging the stale decision, linked back to it."""
    from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType, Tier

    title = f"Decision drift: '{decision_node['title']}' may be stale"
    content = (
        f"The decision '{decision_node['title']}' references `{file_path}`, "
        f"which has accumulated {commits} commit(s) since the decision was recorded "
        f"(created: {decision_node.get('created', 'unknown')}). "
        f"Consider reviewing whether this decision still reflects the current state of the code.\n\n"
        f"Decision ID: {decision_node['id']}"
    )

    req = CreateNodeRequest(
        title=title,
        content=content,
        type=NodeType.observation,
        tier=Tier.working,
        tags=["decision-drift", f"file:{file_path}"],
        space=decision_node.get("space"),
    )

    obs_id, _ = engine.remember(req, agent_id="decision_drift_detector")

    # Link observation → decision
    try:
        from ormah.models.node import ConnectRequest
        engine.connect(
            ConnectRequest(
                source_id=obs_id,
                target_id=decision_node["id"],
                edge=EdgeType.supports,
                weight=0.9,
            )
        )
    except Exception as e:
        logger.debug("Failed to link drift observation to decision: %s", e)

    return obs_id


def run_decision_drift_detection(engine) -> None:
    """Main entry point for the decision drift detector background job."""
    settings = engine.settings
    if not getattr(settings, "decision_drift_enabled", True):
        logger.debug("Decision drift detector skipped: disabled")
        return

    churn_threshold: int = getattr(settings, "decision_drift_churn_threshold", 5)
    refire_days: int = getattr(settings, "decision_drift_refire_days", 30)

    _ensure_table(engine)

    # Discover git repo root
    repo_root: Path | None = None
    for candidate in [settings.memory_dir, Path.cwd()]:
        repo_root = _git_root(candidate)
        if repo_root:
            break

    if repo_root is None:
        logger.debug("Decision drift detector: no git repo found — skipping")
        return

    # Find all decision nodes
    decision_rows = engine.db.conn.execute(
        "SELECT id, title, created, space FROM nodes WHERE type = 'decision'"
    ).fetchall()

    if not decision_rows:
        logger.debug("Decision drift detector: no decision nodes found")
        return

    logger.info("Decision drift detector: checking %d decision node(s)", len(decision_rows))

    now = datetime.now(timezone.utc)
    alerts_created = 0

    for row in decision_rows:
        node_id = row["id"]
        created_iso = row["created"] or now.isoformat()

        # Get file tags for this decision
        tag_rows = engine.db.conn.execute(
            "SELECT tag FROM node_tags WHERE node_id = ? AND tag LIKE ?",
            (node_id, f"{_FILE_TAG_PREFIX}%"),
        ).fetchall()

        if not tag_rows:
            continue  # no file association — skip (scope constraint)

        node_dict = dict(row)

        for tag_row in tag_rows:
            file_path = tag_row["tag"][len(_FILE_TAG_PREFIX):]

            # Check if already alerted and within refire window
            existing = engine.db.conn.execute(
                "SELECT alerted_at FROM decision_drift_alerts "
                "WHERE decision_id = ? AND file_path = ?",
                (node_id, file_path),
            ).fetchone()

            if existing:
                last_alerted = datetime.fromisoformat(existing["alerted_at"])
                days_since = (now - last_alerted.replace(tzinfo=timezone.utc)).days
                if days_since < refire_days:
                    continue  # suppress within refire window

            commits = _commits_since(repo_root, file_path, created_iso)

            if commits < churn_threshold:
                continue

            logger.info(
                "Decision drift: '%s' — %s has %d commits since creation",
                row["title"],
                file_path,
                commits,
            )

            obs_id = _create_drift_observation(engine, node_dict, file_path, commits)
            _update_alert(engine, node_id, file_path, commits, obs_id)
            alerts_created += 1

    logger.info("Decision drift detector: created %d alert(s)", alerts_created)
