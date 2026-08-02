from datetime import datetime, timezone

from ormah.models.node import CreateNodeRequest, NodeType, UpdateNodeRequest


# --- duplicate_checked invalidation ---


def _create_node(engine, title, content):
    req = CreateNodeRequest(content=content, type=NodeType.fact, title=title, tags=["test"])
    node_id, _ = engine.remember(req, agent_id="test")
    return node_id


def _seed_duplicate_checked(engine, node_a, node_b):
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO duplicate_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'not_duplicate', ?)",
            (node_a, node_b, datetime.now(timezone.utc).isoformat()),
        )


def _count_duplicate_checked(engine):
    return engine.db.conn.execute("SELECT count(*) FROM duplicate_checked").fetchone()[0]


def test_update_node_invalidates_duplicate_checked(engine):
    id_a = _create_node(engine, "A", "Some content about A")
    id_b = _create_node(engine, "B", "Some content about B")
    _seed_duplicate_checked(engine, id_a, id_b)
    assert _count_duplicate_checked(engine) == 1

    engine.update_node(id_a, UpdateNodeRequest(content="Updated content about A"))

    assert _count_duplicate_checked(engine) == 0


def test_delete_node_invalidates_duplicate_checked(engine):
    id_a = _create_node(engine, "A", "Some content about A")
    id_b = _create_node(engine, "B", "Some content about B")
    _seed_duplicate_checked(engine, id_a, id_b)
    assert _count_duplicate_checked(engine) == 1

    engine.delete_node(id_b)

    assert _count_duplicate_checked(engine) == 0


def test_execute_merge_invalidates_duplicate_checked_for_both_nodes(engine):
    id_a = _create_node(engine, "A", "Short.")
    id_b = _create_node(engine, "B", "This is a much longer description with detail.")
    id_c = _create_node(engine, "C", "Unrelated third node content")

    # Seed a pair involving the removed node (id_a, kept as it has less content -> id_a removed)
    # and a pair involving the kept node (id_b), to prove BOTH sides are invalidated.
    _seed_duplicate_checked(engine, id_a, id_c)
    _seed_duplicate_checked(engine, id_b, id_c)
    assert _count_duplicate_checked(engine) == 2

    engine.execute_merge(id_a, id_b, merged_content="Merged content for kept node.")

    assert _count_duplicate_checked(engine) == 0


# --- conflict_checked table + invalidation ---


def test_conflict_checked_table_exists(engine):
    """conflict_checked appears on a standard engine fixture (schema.sql runs on every
    engine construction, before _migrate())."""
    engine.db.conn.execute("SELECT node_a, node_b, result, checked_at FROM conflict_checked").fetchall()


def _seed_conflict_checked(engine, node_a, node_b):
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO conflict_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'none', ?)",
            (node_a, node_b, datetime.now(timezone.utc).isoformat()),
        )


def _count_conflict_checked(engine):
    return engine.db.conn.execute("SELECT count(*) FROM conflict_checked").fetchone()[0]


def test_update_node_invalidates_conflict_checked_on_space_edit(engine):
    id_a = _create_node(engine, "A", "Some content about A")
    id_b = _create_node(engine, "B", "Some content about B")
    _seed_conflict_checked(engine, id_a, id_b)
    assert _count_conflict_checked(engine) == 1

    engine.update_node(id_a, UpdateNodeRequest(space="myproject"))

    assert _count_conflict_checked(engine) == 0


def test_update_node_invalidates_conflict_checked_on_type_edit(engine):
    id_a = _create_node(engine, "A", "Some content about A")
    id_b = _create_node(engine, "B", "Some content about B")
    _seed_conflict_checked(engine, id_a, id_b)
    assert _count_conflict_checked(engine) == 1

    engine.update_node(id_a, UpdateNodeRequest(type=NodeType.decision))

    assert _count_conflict_checked(engine) == 0


def test_update_node_invalidates_conflict_checked_on_content_edit(engine):
    id_a = _create_node(engine, "A", "Some content about A")
    id_b = _create_node(engine, "B", "Some content about B")
    _seed_conflict_checked(engine, id_a, id_b)
    assert _count_conflict_checked(engine) == 1

    engine.update_node(id_a, UpdateNodeRequest(content="Updated content about A"))

    assert _count_conflict_checked(engine) == 0


def test_delete_node_invalidates_conflict_checked(engine):
    id_a = _create_node(engine, "A", "Some content about A")
    id_b = _create_node(engine, "B", "Some content about B")
    _seed_conflict_checked(engine, id_a, id_b)
    assert _count_conflict_checked(engine) == 1

    engine.delete_node(id_b)

    assert _count_conflict_checked(engine) == 0


def test_execute_merge_invalidates_conflict_checked_for_both_nodes(engine):
    id_a = _create_node(engine, "A", "Short.")
    id_b = _create_node(engine, "B", "This is a much longer description with detail.")
    id_c = _create_node(engine, "C", "Unrelated third node content")

    _seed_conflict_checked(engine, id_a, id_c)
    _seed_conflict_checked(engine, id_b, id_c)
    assert _count_conflict_checked(engine) == 2

    engine.execute_merge(id_a, id_b, merged_content="Merged content for kept node.")

    assert _count_conflict_checked(engine) == 0
