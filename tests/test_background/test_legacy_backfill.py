from __future__ import annotations

from ormah.background.forgetting_manager import _BACKFILL_META_KEY, run_forgetting
from ormah.models.node import CreateNodeRequest, NodeType, Tier


def _legacy_archival(engine, content="legacy"):
    """A node whose FILE lacks archived_at (remember(tier=archival) never stamps it)."""
    node_id, _ = engine.remember(CreateNodeRequest(
        content=content, type=NodeType.fact, tier=Tier.archival, title=content))
    assert engine.file_store.load(node_id).archived_at is None
    return node_id


def _meta_done(engine):
    row = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key=?", (_BACKFILL_META_KEY,)).fetchone()
    return row is not None


def test_backfill_stamps_legacy_files_and_survives_rebuild(engine):
    engine.settings.deletion_enabled = True
    node_id = _legacy_archival(engine)

    run_forgetting(engine)

    assert engine.file_store.load(node_id).archived_at is not None  # file stamped (durable)
    engine.builder.full_rebuild()
    row = engine.db.conn.execute(
        "SELECT archived_at FROM nodes WHERE id=?", (node_id,)).fetchone()
    assert row["archived_at"] is not None  # survives rebuild
    assert _meta_done(engine) is True


def test_backfill_skipped_when_disabled(engine):
    node_id = _legacy_archival(engine)
    run_forgetting(engine)  # deletion_enabled defaults to False
    assert engine.file_store.load(node_id).archived_at is None
    assert _meta_done(engine) is False


def test_backfill_write_failure_preserves_file_and_retries(engine, monkeypatch):
    engine.settings.deletion_enabled = True
    node_id = _legacy_archival(engine)

    def boom(node):
        raise OSError("disk full")
    monkeypatch.setattr(engine.file_store, "save", boom)

    run_forgetting(engine)  # must not raise; file untouched; not marked done

    reloaded = engine.file_store.load(node_id)
    assert reloaded is not None and reloaded.archived_at is None  # original intact
    assert _meta_done(engine) is False  # transient failure → retry next run
