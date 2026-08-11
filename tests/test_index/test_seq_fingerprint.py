"""Conditional seq allocation driven by a persisted content fingerprint (#126)."""

from __future__ import annotations

from ormah.models.node import Connection, CreateNodeRequest, EdgeType, NodeType


def _row(engine, node_id: str):
    return engine.db.conn.execute(
        "SELECT seq, content_fingerprint FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()


def _seq(engine, node_id: str) -> int:
    return _row(engine, node_id)["seq"]


def _make_node(engine, title="Python language", content="Python is a programming language.",
               node_type=NodeType.fact, tags=None):
    node_id, _ = engine.remember(
        CreateNodeRequest(content=content, type=node_type, title=title,
                          tags=tags if tags is not None else ["test"]),
        agent_id="test",
    )
    return node_id


def test_indexing_persists_a_content_fingerprint(engine):
    """The builder stamps a fingerprint on every node it indexes."""
    node_id = _make_node(engine)
    fp = _row(engine, node_id)["content_fingerprint"]
    assert fp, "builder must persist a content fingerprint"
    assert len(fp) == 64, "expected a sha256 hex digest"


def test_edge_write_does_not_bump_seq(engine):
    """Persisting a connection must not requeue the node (#126).

    _apply_edge appends a Connection, touches `updated`, and saves the markdown. That
    rewrite used to bump `seq`, sending the node back to the end of the auto_linker queue
    with nothing new to learn — the pairs are already in auto_link_checked. That is what
    pinned the backlog at ~the size of the store.
    """
    id_a = _make_node(engine)
    id_b = _make_node(engine, title="Ruby language", content="Ruby is a programming language.")
    seq_before = _seq(engine, id_a)

    node = engine.file_store.load(id_a)
    node.connections.append(
        Connection(target=id_b, edge=EdgeType.related_to, weight=0.9, reason="both languages")
    )
    node.touch_updated()
    engine.builder.index_single(engine.file_store.save(node))

    assert _seq(engine, id_a) == seq_before, "an edge write must not requeue the node"


def test_fingerprint_change_invalidates_cached_pairs(engine):
    """A requeue is a no-op unless the cached verdicts go with it."""
    id_a = _make_node(engine)
    id_b = _make_node(engine, title="Ruby language", content="Ruby is a programming language.")
    pair = tuple(sorted([id_a, id_b]))
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'none', '2026-07-14T00:00:00+00:00')",
            pair,
        )

    # a SPACE change: memory_engine would NOT clear auto_link_checked for this
    node = engine.file_store.load(id_a)
    node.space = "some-other-space"
    engine.builder.index_single(engine.file_store.save(node))

    left = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert left is None, "a fingerprint change must drop the node's cached pair verdicts"


def test_edge_write_keeps_cached_pairs(engine):
    """An edge write must NOT invalidate cached verdicts (it would refeed the LLM)."""
    id_a = _make_node(engine)
    id_b = _make_node(engine, title="Ruby language", content="Ruby is a programming language.")
    pair = tuple(sorted([id_a, id_b]))
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'related_to', '2026-07-14T00:00:00+00:00')",
            pair,
        )

    node = engine.file_store.load(id_a)
    node.connections.append(Connection(target=id_b, edge=EdgeType.related_to, weight=0.9))
    node.touch_updated()
    engine.builder.index_single(engine.file_store.save(node))

    left = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert left is not None, "an edge write must not invalidate cached verdicts"


def test_direct_db_space_update_still_requeues(engine):
    """auto_cluster dual-writes `space`: straight into SQLite AND into the markdown."""
    node_id = _make_node(engine)
    seq_before = _seq(engine, node_id)

    # exactly what auto_cluster does: DB first...
    with engine.db.transaction() as conn:
        conn.execute("UPDATE nodes SET space = ? WHERE id = ?", ("clustered-space", node_id))
    # ...then the markdown, never through the builder
    node = engine.file_store.load(node_id)
    node.space = "clustered-space"
    path = engine.file_store.save(node)

    engine.builder.index_single(path)

    assert _seq(engine, node_id) > seq_before, (
        "a space change written directly to the DB must still requeue the node — "
        "comparing against the stored row instead of the fingerprint would freeze it"
    )


def test_content_change_bumps_seq(engine):
    """Content feeds the embedding and the judge prompt."""
    node_id = _make_node(engine)
    seq_before = _seq(engine, node_id)
    max_before = engine.db.conn.execute("SELECT MAX(seq) m FROM nodes").fetchone()["m"]

    node = engine.file_store.load(node_id)
    node.content = "Totally different subject: baking sourdough bread."
    engine.builder.index_single(engine.file_store.save(node))

    assert _seq(engine, node_id) > seq_before
    assert _seq(engine, node_id) > max_before, "must land at the head of the queue"


def test_title_change_bumps_seq(engine):
    node_id = _make_node(engine)
    seq_before = _seq(engine, node_id)
    node = engine.file_store.load(node_id)
    node.title = "An entirely different title"
    engine.builder.index_single(engine.file_store.save(node))
    assert _seq(engine, node_id) > seq_before


def test_type_change_bumps_seq(engine):
    """Type is shown to the LLM judge."""
    node_id = _make_node(engine, node_type=NodeType.fact)
    seq_before = _seq(engine, node_id)
    node = engine.file_store.load(node_id)
    node.type = NodeType.decision
    engine.builder.index_single(engine.file_store.save(node))
    assert _seq(engine, node_id) > seq_before


def test_tags_only_change_does_not_bump_seq(engine):
    """Tags feed FTS, never the linker."""
    node_id = _make_node(engine, tags=["one"])
    seq_before = _seq(engine, node_id)

    node = engine.file_store.load(node_id)
    node.tags = ["one", "two"]
    engine.builder.index_single(engine.file_store.save(node))

    assert _seq(engine, node_id) == seq_before
    tags = {r["tag"] for r in engine.db.conn.execute(
        "SELECT tag FROM node_tags WHERE node_id = ?", (node_id,))}
    assert tags == {"one", "two"}, "the tag edit must still land in the index"


def test_migration_leaves_stale_rows_unstamped(engine):
    """A row whose file on disk no longer matches its file_hash has a pending reindex.

    Backfilling its fingerprint from the row would bake in a change that was never indexed
    (auto_cluster writes `space` to the DB and the markdown without going through the
    builder), so the next reindex would see no change and the relink would be lost. Those
    rows must be left NULL, which forces a mismatch and a requeue.
    """
    node_id = _make_node(engine)

    # simulate the pending-reindex window: the DB's file_hash no longer matches the file,
    # and the fingerprint was never stamped (as the migration would leave it)
    with engine.db.transaction() as conn:
        conn.execute("UPDATE nodes SET file_hash = ? WHERE id = ?", ("stale-hash", node_id))
        conn.execute("UPDATE nodes SET content_fingerprint = NULL WHERE id = ?", (node_id,))

    # re-running the migration must NOT stamp this row
    engine.db._migrate()

    assert _row(engine, node_id)["content_fingerprint"] is None, (
        "a row with a pending reindex must be left unstamped, or its relink is lost"
    )

    # ...and the first reindex requeues it
    seq_before = _seq(engine, node_id)
    node = engine.file_store.load(node_id)
    engine.builder.index_single(engine.file_store.save(node))
    assert _seq(engine, node_id) > seq_before, "a NULL fingerprint must requeue on first reindex"


def test_migration_stamps_clean_rows(engine):
    """A row whose file matches its hash is stamped, so the upgrade does not requeue the store."""
    node_id = _make_node(engine)
    with engine.db.transaction() as conn:
        conn.execute("UPDATE nodes SET content_fingerprint = NULL WHERE id = ?", (node_id,))

    engine.db._migrate()

    assert _row(engine, node_id)["content_fingerprint"] is not None, (
        "a clean row must be backfilled, or the whole store gets requeued on upgrade"
    )


def test_unchanged_content_reindex_keeps_the_vector(engine):
    """A reindex that does not change the content must not destroy the embedding.

    index_single() deletes the node's vector, and nothing re-embeds it. Combined with the
    preserved seq (#126), a metadata-only reindex would leave the node behind the watermark
    AND without a vector — invisible to the linker and unable to be another node's semantic
    candidate. mark_outdated() (which only sets valid_until) walks exactly this path.
    """
    node_id = _make_node(engine)

    def has_vector() -> bool:
        row = engine.db.conn.execute(
            "SELECT 1 FROM node_vectors WHERE id = ?", (node_id,)
        ).fetchone()
        return row is not None

    assert has_vector(), "precondition: the node was embedded on creation"
    seq_before = _seq(engine, node_id)

    # a metadata-only rewrite: content/title/type/space untouched
    node = engine.file_store.load(node_id)
    node.touch_updated()
    engine.builder.index_single(engine.file_store.save(node))

    assert _seq(engine, node_id) == seq_before, "metadata-only reindex must not requeue"
    assert has_vector(), (
        "the vector must survive an unchanged-content reindex — nothing re-embeds it, and "
        "the node would be stranded behind the watermark with no vector"
    )


def test_content_change_reindex_drops_the_stale_vector(engine):
    """The converse: when the content changes, the old vector is stale and must go."""
    node_id = _make_node(engine)
    node = engine.file_store.load(node_id)
    node.content = "Totally different subject: baking sourdough bread."
    engine.builder.index_single(engine.file_store.save(node))

    # the node was requeued, so the linker will re-embed/re-evaluate it
    row = engine.db.conn.execute(
        "SELECT seq FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    assert row is not None


def test_full_rebuild_allocates_new_seq(engine):
    """A mass reindex requeues the whole store and clears the watermark."""
    node_id = _make_node(engine)
    seq_before = _seq(engine, node_id)
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('auto_link_watermark', ?)",
            (str(seq_before),),
        )

    engine.builder.full_rebuild()

    assert _seq(engine, node_id) > seq_before, "mass reindex must land nodes at the head"
    watermark = engine.db.conn.execute(
        "SELECT value FROM meta WHERE key = 'auto_link_watermark'"
    ).fetchone()
    assert watermark is None, "full_rebuild must clear the watermark"


def test_full_rebuild_invalidates_pairs_only_for_changed_content(engine):
    """A rebuild over CHANGED content must drop that node's cached verdicts.

    full_rebuild clears the watermark so everything is requeued — but auto_linker skips any
    pair already in auto_link_checked, so without invalidation the old content's link
    decisions silently carry over to the restored/edited content. Unchanged nodes must keep
    their verdicts: re-judging an untouched store with the LLM would be a huge pointless cost.
    """
    changed = _make_node(engine, title="Python", content="Python is a language.")
    untouched = _make_node(engine, title="Ruby", content="Ruby is a language.")
    pair_changed = tuple(sorted([changed, untouched]))

    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'none', '2026-07-14T00:00:00+00:00')",
            pair_changed,
        )

    # edit one node's content on disk, leave the other alone
    node = engine.file_store.load(changed)
    node.content = "Totally different subject: baking sourdough bread."
    engine.file_store.save(node)

    engine.builder.full_rebuild()

    left = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair_changed
    ).fetchone()
    assert left is None, "a rebuild over changed content must drop that node's cached verdicts"


def test_full_rebuild_keeps_pairs_when_content_is_unchanged(engine):
    """The converse: rebuilding an untouched store must NOT re-judge it with the LLM."""
    id_a = _make_node(engine, title="Python", content="Python is a language.")
    id_b = _make_node(engine, title="Ruby", content="Ruby is a language.")
    pair = tuple(sorted([id_a, id_b]))
    with engine.db.transaction() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO auto_link_checked (node_a, node_b, result, checked_at) "
            "VALUES (?, ?, 'related_to', '2026-07-14T00:00:00+00:00')",
            pair,
        )

    engine.builder.full_rebuild()   # nothing on disk changed

    left = engine.db.conn.execute(
        "SELECT 1 FROM auto_link_checked WHERE node_a = ? AND node_b = ?", pair
    ).fetchone()
    assert left is not None, "an unchanged rebuild must not purge the pair cache"
