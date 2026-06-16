from __future__ import annotations

from datetime import datetime, timedelta, timezone

import frontmatter
import pytest

from ormah.background.forgetting_manager import run_forgetting
from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType, Tier


def _exists(engine, node_id) -> bool:
    row = engine.db.conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row is not None


def _enable(engine):
    engine.settings.deletion_enabled = True


def _make_eligible(engine, content="dead weight", days=200):
    """Create an archival node eligible in BOTH file and index (the guard reads the file)."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content=content, type=NodeType.fact, tier=Tier.archival, title=content))
    old = datetime.now(timezone.utc) - timedelta(days=days)
    node = engine.file_store.load(node_id)
    node.importance = 0.1
    node.stability = 1.0
    node.last_review = old
    node.last_accessed = old
    node.archived_at = old
    path = engine.file_store.save(node)        # source of truth
    engine.builder.index_single(path)          # keep the index in sync
    engine.db.conn.commit()
    return node_id


def test_master_switch_off_is_noop(engine):
    node_id = _make_eligible(engine)
    run_forgetting(engine)  # deletion_enabled defaults to False
    assert _exists(engine, node_id) is True


def test_fully_eligible_node_is_soft_deleted(engine):
    _enable(engine)
    node_id = _make_eligible(engine)
    run_forgetting(engine)
    assert _exists(engine, node_id) is False


def test_idempotent_second_run_deletes_nothing(engine):
    _enable(engine)
    node_id = _make_eligible(engine)
    run_forgetting(engine)
    run_forgetting(engine)
    assert _exists(engine, node_id) is False


# --- conjunction matrix: passing all gates deletes; breaking exactly one keeps (council M1) ---

def _break(engine, node_id, gate):
    now = datetime.now(timezone.utc)
    recent = now.isoformat()
    if gate == "tier":
        engine.db.conn.execute("UPDATE nodes SET tier='working' WHERE id=?", (node_id,))
    elif gate == "archived_recent":
        engine.db.conn.execute("UPDATE nodes SET archived_at=? WHERE id=?", (recent, node_id))
    elif gate == "accessed_recent":
        engine.db.conn.execute("UPDATE nodes SET last_accessed=? WHERE id=?", (recent, node_id))
    elif gate == "retrievable":   # high stability ⇒ R well above floor
        engine.db.conn.execute("UPDATE nodes SET stability=100000.0 WHERE id=?", (node_id,))
    elif gate == "importance":
        engine.db.conn.execute("UPDATE nodes SET importance=0.9 WHERE id=?", (node_id,))
    elif gate == "archived_null":
        engine.db.conn.execute("UPDATE nodes SET archived_at=NULL WHERE id=?", (node_id,))
    elif gate == "feedback":
        engine.db.conn.execute(
            "INSERT INTO affinity (prompt_vec, node_id, signal, source, confirmed_at, session_id) "
            "VALUES (?, ?, 1, 'explicit', ?, 's1')", (b"\x00", node_id, recent))
    engine.db.conn.commit()


@pytest.mark.parametrize("gate", [
    "tier", "archived_recent", "accessed_recent", "retrievable",
    "importance", "archived_null", "feedback",
])
def test_breaking_one_gate_keeps_node(engine, gate):
    _enable(engine)
    node_id = _make_eligible(engine)
    _break(engine, node_id, gate)
    run_forgetting(engine)
    assert _exists(engine, node_id) is True, f"gate={gate} should have protected the node"


def test_strong_edge_protects_both_nodes(engine):
    _enable(engine)
    a = _make_eligible(engine, content="hub a")
    b = _make_eligible(engine, content="hub b")
    engine.connect(ConnectRequest(source_id=a, target_id=b, edge=EdgeType.related_to, weight=0.9))
    run_forgetting(engine)
    assert _exists(engine, a) is True and _exists(engine, b) is True


def test_user_node_never_deleted(engine):
    _enable(engine)
    uid = engine.user_node_id
    assert uid is not None
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET tier='archival', importance=0.1, stability=1.0, "
        "last_review=?, last_accessed=?, archived_at=? WHERE id=?",
        (old, old, old, uid))
    engine.db.conn.commit()
    run_forgetting(engine)
    assert _exists(engine, uid) is True


def test_guard_reads_file_over_stale_index(engine):
    """Cross-path race (council R3 C5): a promotion writes the FILE before the index.

    The pre-filter (index) still sees archival+stale and selects the node, but the hybrid guard
    reads the source file (tier=working) and aborts. Fails with an index-only guard.
    """
    _enable(engine)
    node_id = _make_eligible(engine)
    node = engine.file_store.load(node_id)
    node.tier = Tier.working
    engine.file_store.save(node)  # file promoted; index intentionally NOT updated
    run_forgetting(engine)
    assert _exists(engine, node_id) is True  # guard saw the fresh file → no deletion


def _make_archival_recent(engine, content, archived_days, importance=0.1):
    """Archival node NOT gate-stale (recent access), eligible in BOTH file and index."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content=content, type=NodeType.fact, tier=Tier.archival, title=content))
    recent = datetime.now(timezone.utc) - timedelta(days=3)
    archived = datetime.now(timezone.utc) - timedelta(days=archived_days)
    node = engine.file_store.load(node_id)
    node.importance = importance
    node.stability = 1.0
    node.last_review = recent
    node.last_accessed = recent
    node.archived_at = archived
    path = engine.file_store.save(node)
    engine.builder.index_single(path)
    engine.db.conn.commit()
    return node_id


def _archival_count(engine):
    return engine.db.conn.execute(
        "SELECT COUNT(*) AS c FROM nodes WHERE tier='archival'").fetchone()["c"]


def test_cap_evicts_worst_down_to_cap(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 2
    ids = [_make_archival_recent(engine, f"n{i}", archived_days=age)
           for i, age in enumerate([300, 250, 200, 150, 50])]
    run_forgetting(engine)  # none are gate-stale → Phase A deletes nothing
    assert _archival_count(engine) == 2
    assert _exists(engine, ids[4]) is True   # youngest (least forgettable) survives
    assert _exists(engine, ids[3]) is True
    assert _exists(engine, ids[0]) is False  # oldest evicted


def test_cap_never_evicts_protected_high_importance(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    old_imp = _make_archival_recent(engine, "old important", archived_days=400, importance=0.9)
    mid = _make_archival_recent(engine, "mid", archived_days=200)
    young = _make_archival_recent(engine, "young", archived_days=20)
    run_forgetting(engine)
    assert _exists(engine, old_imp) is True   # protected by importance despite worst age
    # spec §3 "evict worst-first down to the cap": both unprotected nodes are evicted so the
    # surviving total equals the cap (1 = the protected old_imp). The protected node is never
    # touched; the forget-score only orders the unprotected remainder (mid before young).
    assert _exists(engine, young) is False    # evicted: surviving total brought down to the cap
    assert _exists(engine, mid) is False      # evicted: worse forget-score, goes first
    assert _archival_count(engine) == 1       # only the protected node remains (== cap)


def test_cap_accepts_overflow_when_only_protected_remain(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    _make_archival_recent(engine, "fa", archived_days=300, importance=0.9)
    _make_archival_recent(engine, "fb", archived_days=250, importance=0.9)
    run_forgetting(engine)
    assert _archival_count(engine) == 2  # both protected → cap exceeded, nothing deleted


def test_cap_protects_feedback_node(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    fb = _make_archival_recent(engine, "fb node", archived_days=400)
    other = _make_archival_recent(engine, "other", archived_days=30)
    engine.db.conn.execute(
        "INSERT INTO affinity (prompt_vec, node_id, signal, source, confirmed_at, session_id) "
        "VALUES (?, ?, 1, 'explicit', ?, 's1')",
        (b"\x00", fb, datetime.now(timezone.utc).isoformat()))
    engine.db.conn.commit()
    run_forgetting(engine)
    assert _exists(engine, fb) is True       # feedback protects even the worst-scored
    assert _exists(engine, other) is False


def test_cap_disabled_by_default_zero(engine):
    _enable(engine)
    for i in range(4):
        _make_archival_recent(engine, f"z{i}", archived_days=300)
    run_forgetting(engine)
    assert _archival_count(engine) == 4  # cap off → no eviction


def test_cap_protects_strong_edge_hub(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 1
    a = _make_archival_recent(engine, "hub a", archived_days=400)
    b = _make_archival_recent(engine, "hub b", archived_days=380)
    filler = _make_archival_recent(engine, "filler", archived_days=20)
    engine.connect(ConnectRequest(source_id=a, target_id=b, edge=EdgeType.related_to, weight=0.9))
    run_forgetting(engine)
    assert _exists(engine, a) is True and _exists(engine, b) is True  # hub protected in cap
    assert _exists(engine, filler) is False


def test_cap_never_evicts_user_node(engine):
    _enable(engine)
    engine.settings.archival_soft_cap = 0  # force overflow regardless of count below
    uid = engine.user_node_id
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET tier='archival', importance=0.1, stability=1.0, "
        "last_review=?, last_accessed=?, archived_at=? WHERE id=?",
        (old, old, old, uid))
    engine.db.conn.commit()
    engine.settings.archival_soft_cap = 1
    _make_archival_recent(engine, "other", archived_days=30)
    run_forgetting(engine)
    assert _exists(engine, uid) is True  # self node never evicted by the cap


def _backdate_tombstone(engine, node_id, days):
    when = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat().replace("+00:00", "Z")
    for nid, _da, path in engine.file_store.list_deleted():
        if nid == node_id:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
            post["deleted_at"] = when
            path.write_text(frontmatter.dumps(post), encoding="utf-8")
            return
    raise AssertionError(f"tombstone for {node_id} not found")


def _tombstone_ids(engine):
    return {nid for nid, _da, _p in engine.file_store.list_deleted()}


def test_expired_tombstone_is_purged_and_audited(engine):
    _enable(engine)
    node_id, _ = engine.remember(CreateNodeRequest(
        content="bye", type=NodeType.fact, tier=Tier.working, title="bye"))
    engine.delete_node(node_id)              # soft-delete → deleted/
    _backdate_tombstone(engine, node_id, days=60)  # retention is 30

    run_forgetting(engine)

    assert node_id not in _tombstone_ids(engine)
    audited = engine.db.conn.execute(
        "SELECT 1 FROM audit_log WHERE operation='purge' AND node_id=?", (node_id,)
    ).fetchone()
    assert audited is not None


def test_tombstone_within_window_is_kept(engine):
    _enable(engine)
    node_id, _ = engine.remember(CreateNodeRequest(
        content="recent", type=NodeType.fact, tier=Tier.working, title="recent"))
    engine.delete_node(node_id)
    _backdate_tombstone(engine, node_id, days=5)  # inside the 30-day window

    run_forgetting(engine)

    assert node_id in _tombstone_ids(engine)


def test_purge_skipped_when_disabled(engine):
    # master switch OFF: even an old tombstone is not purged
    node_id, _ = engine.remember(CreateNodeRequest(
        content="keep", type=NodeType.fact, tier=Tier.working, title="keep"))
    engine.delete_node(node_id)
    _backdate_tombstone(engine, node_id, days=60)

    run_forgetting(engine)  # deletion_enabled defaults to False

    assert node_id in _tombstone_ids(engine)
