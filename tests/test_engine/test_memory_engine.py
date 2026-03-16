"""Tests for the memory engine."""

from ormah.engine.memory_engine import _embedding_text, _generate_title
from ormah.models.node import CreateNodeRequest, NodeType, Tier, ConnectRequest, EdgeType, UpdateNodeRequest


# ---------------------------------------------------------------------------
# _embedding_text helper
# ---------------------------------------------------------------------------


def test_embedding_text_truncates_long_content():
    title = "My Title"
    content = "a" * 1000
    result = _embedding_text(title, content, max_content_chars=512)
    assert result.startswith("My Title ")
    # title (8) + space (1) + 512 = 521
    assert len(result) == 8 + 1 + 512


def test_embedding_text_short_content_unchanged():
    result = _embedding_text("Title", "short content", max_content_chars=512)
    assert result == "Title short content"


def test_embedding_text_no_title():
    result = _embedding_text(None, "some content", max_content_chars=512)
    assert result == "some content"


def test_embedding_text_empty():
    assert _embedding_text(None, "") == ""


# ---------------------------------------------------------------------------
# _generate_title helper
# ---------------------------------------------------------------------------


def test_generate_title_short_content():
    assert _generate_title("Hello world") == "Hello world"


def test_generate_title_long_content():
    long = "This is a very long sentence that should be truncated at a word boundary to keep titles short and readable"
    result = _generate_title(long, max_chars=60)
    assert len(result) <= 61  # 60 + ellipsis char
    assert result.endswith("…")
    assert " " not in result[-1:]  # shouldn't end with space before ellipsis


def test_generate_title_multiline():
    content = "First line is the title\nSecond line is body text"
    assert _generate_title(content) == "First line is the title"


def test_generate_title_empty():
    assert _generate_title("") == ""


# ---------------------------------------------------------------------------
# Auto-title generation in remember()
# ---------------------------------------------------------------------------


def test_remember_generates_title_when_missing(engine):
    """Calling remember() without a title should auto-generate one from content."""
    req = CreateNodeRequest(
        content="Python is a high-level programming language used for scripting.",
        type=NodeType.fact,
        tags=["programming"],
    )
    node_id, text = engine.remember(req, agent_id="test")
    assert node_id is not None
    # The auto-generated title should appear in the formatted text
    assert "Python is a high-level programming language" in text

    # Verify the node on disk actually has a title
    node = engine.file_store.load(node_id)
    assert node.title is not None
    assert len(node.title) > 0


def test_remember_preserves_explicit_title(engine):
    """Calling remember() with an explicit title should not overwrite it."""
    req = CreateNodeRequest(
        content="Some content here.",
        type=NodeType.fact,
        title="My Custom Title",
    )
    node_id, text = engine.remember(req, agent_id="test")
    node = engine.file_store.load(node_id)
    assert node.title == "My Custom Title"


def test_remember(engine):
    req = CreateNodeRequest(
        content="Python is a programming language.",
        type=NodeType.fact,
        title="Python language",
        tags=["programming"],
    )
    node_id, text = engine.remember(req, agent_id="test")
    assert node_id is not None
    assert "Python language" in text


def test_recall_node(engine):
    req = CreateNodeRequest(
        content="FastAPI is a web framework.",
        type=NodeType.fact,
        title="FastAPI",
    )
    node_id, _ = engine.remember(req)

    text = engine.recall_node(node_id)
    assert text is not None
    assert "FastAPI" in text


def test_update_node(engine):
    req = CreateNodeRequest(content="Old content.", type=NodeType.fact)
    node_id, _ = engine.remember(req)

    update = UpdateNodeRequest(content="Updated content.", title="Updated")
    text = engine.update_node(node_id, update)
    assert text is not None
    assert "Updated" in text


def test_connect(engine):
    req1 = CreateNodeRequest(content="Node A.", type=NodeType.fact)
    req2 = CreateNodeRequest(content="Node B.", type=NodeType.fact)
    id1, _ = engine.remember(req1)
    id2, _ = engine.remember(req2)

    connect_req = ConnectRequest(
        source_id=id1, target_id=id2, edge=EdgeType.related_to
    )
    text = engine.connect(connect_req)
    assert "Connected" in text


def test_context_empty(engine):
    text = engine.get_context()
    assert "No core memories" in text


def test_context_with_core(engine):
    req = CreateNodeRequest(
        content="User prefers dark mode.",
        type=NodeType.preference,
        tier=Tier.core,
        title="Dark mode preference",
    )
    engine.remember(req)

    text = engine.get_context()
    assert "dark mode" in text.lower()


def test_stats(engine):
    req = CreateNodeRequest(content="A fact.", type=NodeType.fact)
    engine.remember(req)

    stats = engine.stats()
    # +1 for the self node created on startup
    assert stats["total_nodes"] == 2
