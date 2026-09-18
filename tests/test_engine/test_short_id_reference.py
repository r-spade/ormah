"""An update addressed by a Short id still resolves end to end (#280)."""

from __future__ import annotations

from ormah.models.node import CreateNodeRequest, NodeType, UpdateNodeRequest


def test_update_node_by_short_id_reaches_the_right_node(engine):
    """update_node forwards node_id straight to FileStore.load — this seam breaks
    if the engine ever normalises or pre-validates the reference before the store
    sees it. A Short id that matches exactly one node must resolve to that node."""
    node_id, _ = engine.remember(
        CreateNodeRequest(content="original content", type=NodeType.fact, title="Short id target")
    )
    short_id = engine.file_store.load(node_id).short_id

    result = engine.update_node(short_id, UpdateNodeRequest(content="updated content"))

    assert result is not None
    node = engine.file_store.load(node_id)
    assert node.content == "updated content"
