"""Tests for co-change mapping background job."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from ormah.background.co_change_mapper import (
    _get_co_change_counts,
    _nodes_for_file,
    _already_linked,
    _write_edge,
    run_co_change_mapping,
)
from ormah.models.node import CreateNodeRequest, NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(engine, title: str, file_tag: str) -> str:
    node_id, _ = engine.remember(
        CreateNodeRequest(
            content=f"Content for {title}",
            type=NodeType.fact,
            title=title,
            tags=[f"file:{file_tag}"],
        ),
        agent_id="test",
    )
    return node_id


def _edges_between(engine, id_a: str, id_b: str):
    return engine.db.conn.execute(
        "SELECT edge_type FROM edges WHERE "
        "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
        (id_a, id_b, id_b, id_a),
    ).fetchall()


# ---------------------------------------------------------------------------
# Unit: _get_co_change_counts
# ---------------------------------------------------------------------------

_FAKE_GIT_OUTPUT = """\
COMMIT
src/foo.py
src/bar.py
COMMIT
src/foo.py
src/baz.py
COMMIT
src/foo.py
src/bar.py
"""


def _fake_run_ok(*args, **kwargs):
    m = MagicMock()
    m.returncode = 0
    m.stdout = _FAKE_GIT_OUTPUT
    return m


def _fake_run_fail(*args, **kwargs):
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    return m


def test_get_co_change_counts_parses_correctly():
    with patch("subprocess.run", side_effect=_fake_run_ok):
        counts = _get_co_change_counts(MagicMock(), lookback_commits=100)

    # foo+bar appear in 2 commits, foo+baz in 1
    assert counts[("src/bar.py", "src/foo.py")] == 2
    assert counts[("src/baz.py", "src/foo.py")] == 1


def test_get_co_change_counts_empty_on_failure():
    with patch("subprocess.run", side_effect=_fake_run_fail):
        counts = _get_co_change_counts(MagicMock(), lookback_commits=100)
    assert counts == {}


# ---------------------------------------------------------------------------
# Unit: _nodes_for_file
# ---------------------------------------------------------------------------

def test_nodes_for_file_finds_tagged_nodes(engine):
    nid = _make_node(engine, "Auth module", "src/auth.py")
    results = _nodes_for_file(engine, "src/auth.py")
    assert nid in results


def test_nodes_for_file_returns_empty_when_untagged(engine):
    _make_node(engine, "Untagged node", "src/other.py")
    results = _nodes_for_file(engine, "src/completely_different.py")
    assert results == []


# ---------------------------------------------------------------------------
# Unit: _already_linked
# ---------------------------------------------------------------------------

def test_already_linked_detects_existing_edge(engine):
    id_a = _make_node(engine, "Node A", "src/a.py")
    id_b = _make_node(engine, "Node B", "src/b.py")
    _write_edge(engine, id_a, id_b, commits=3)
    assert _already_linked(engine, id_a, id_b)
    assert _already_linked(engine, id_b, id_a)  # reverse direction


def test_already_linked_false_for_new_pair(engine):
    engine.settings.contradiction_prestorage_enabled = False
    id_a = _make_node(engine, "Database schema migration scripts", "src/migrations.py")
    id_b = _make_node(engine, "Frontend React component library", "src/ui_components.py")
    assert not _already_linked(engine, id_a, id_b)


# ---------------------------------------------------------------------------
# Unit: _write_edge
# ---------------------------------------------------------------------------

def test_write_edge_creates_related_to_edge(engine):
    engine.settings.contradiction_prestorage_enabled = False
    id_a = _make_node(engine, "Authentication token refresh logic", "src/auth_refresh.py")
    id_b = _make_node(engine, "Database connection pooling strategy", "src/db_pool.py")
    _write_edge(engine, id_a, id_b, commits=5)
    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 1
    assert edges[0][0] == "related_to"


def test_write_edge_weight_capped_at_1(engine):
    engine.settings.contradiction_prestorage_enabled = False
    id_a = _make_node(engine, "Heavy Kubernetes deployment manifest", "src/heavy_a.py")
    id_b = _make_node(engine, "Heavy React frontend router config", "src/heavy_b.py")
    _write_edge(engine, id_a, id_b, commits=100)
    row = engine.db.conn.execute(
        "SELECT weight FROM edges WHERE source_id = ? AND target_id = ?",
        (id_a, id_b),
    ).fetchone()
    assert row is not None
    assert row[0] <= 1.0


# ---------------------------------------------------------------------------
# Integration: run_co_change_mapping
# ---------------------------------------------------------------------------

_GIT_ROOT_OUTPUT = MagicMock(returncode=0, stdout="/repo\n")


def _build_git_log_output(file_pairs: list[tuple[str, str]], repeats: int = 2) -> str:
    lines = []
    for _ in range(repeats):
        lines.append("COMMIT")
        seen: set[str] = set()
        for fa, fb in file_pairs:
            if fa not in seen:
                lines.append(fa)
                seen.add(fa)
            if fb not in seen:
                lines.append(fb)
                seen.add(fb)
    return "\n".join(lines) + "\n"


def test_run_co_change_mapping_creates_edges(engine):
    engine.settings.co_change_mapping_enabled = True
    engine.settings.co_change_min_commits = 2
    engine.settings.co_change_lookback_commits = 100
    engine.settings.contradiction_prestorage_enabled = False

    id_a = _make_node(engine, "Module Alpha", "src/alpha.py")
    id_b = _make_node(engine, "Module Beta", "src/beta.py")

    git_log = _build_git_log_output([("src/alpha.py", "src/beta.py")], repeats=3)

    def fake_run(args, **kwargs):
        m = MagicMock()
        if "rev-parse" in args:
            m.returncode = 0
            m.stdout = "/repo\n"
        else:
            m.returncode = 0
            m.stdout = git_log
        return m

    with patch("subprocess.run", side_effect=fake_run):
        run_co_change_mapping(engine)

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 1


def test_run_co_change_mapping_skips_when_disabled(engine):
    engine.settings.co_change_mapping_enabled = False
    engine.settings.contradiction_prestorage_enabled = False

    id_a = _make_node(engine, "Logging telemetry pipeline", "src/skip_a.py")
    id_b = _make_node(engine, "Kubernetes autoscaling rules", "src/skip_b.py")

    with patch("subprocess.run") as mock_run:
        run_co_change_mapping(engine)
        mock_run.assert_not_called()

    assert not _already_linked(engine, id_a, id_b)


def test_run_co_change_mapping_skips_below_min_commits(engine):
    engine.settings.co_change_mapping_enabled = True
    engine.settings.co_change_min_commits = 5  # high threshold
    engine.settings.contradiction_prestorage_enabled = False

    id_a = _make_node(engine, "Payment gateway integration", "src/low_a.py")
    id_b = _make_node(engine, "Email notification worker", "src/low_b.py")

    git_log = _build_git_log_output([("src/low_a.py", "src/low_b.py")], repeats=2)

    def fake_run(args, **kwargs):
        m = MagicMock()
        if "rev-parse" in args:
            m.returncode = 0
            m.stdout = "/repo\n"
        else:
            m.returncode = 0
            m.stdout = git_log
        return m

    with patch("subprocess.run", side_effect=fake_run):
        run_co_change_mapping(engine)

    assert not _already_linked(engine, id_a, id_b)


def test_run_co_change_mapping_no_duplicate_edges(engine):
    engine.settings.co_change_mapping_enabled = True
    engine.settings.co_change_min_commits = 2
    engine.settings.co_change_lookback_commits = 100
    engine.settings.contradiction_prestorage_enabled = False

    id_a = _make_node(engine, "GraphQL schema definition", "src/dedup_a.py")
    id_b = _make_node(engine, "Redis cache invalidation layer", "src/dedup_b.py")

    git_log = _build_git_log_output([("src/dedup_a.py", "src/dedup_b.py")], repeats=3)

    def fake_run(args, **kwargs):
        m = MagicMock()
        if "rev-parse" in args:
            m.returncode = 0
            m.stdout = "/repo\n"
        else:
            m.returncode = 0
            m.stdout = git_log
        return m

    with patch("subprocess.run", side_effect=fake_run):
        run_co_change_mapping(engine)
        run_co_change_mapping(engine)  # run twice

    edges = _edges_between(engine, id_a, id_b)
    assert len(edges) == 1  # still just one edge
