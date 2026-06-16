from ormah.models.node import CreateNodeRequest, NodeType, Tier, UpdateNodeRequest


def test_archived_at_survives_full_rebuild(engine):
    node_id, _ = engine.remember(CreateNodeRequest(
        content="durable", type=NodeType.fact, tier=Tier.working, title="durable"))
    engine.update_node(node_id, UpdateNodeRequest(tier=Tier.archival))
    stamped = engine.file_store.load(node_id).archived_at
    assert stamped is not None

    engine.builder.full_rebuild()  # re-parse all files into a fresh index

    row = engine.db.conn.execute(
        "SELECT archived_at FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    assert row["archived_at"] is not None  # not wiped by the rebuild
