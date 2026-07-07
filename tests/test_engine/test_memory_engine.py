"""Tests for the memory engine."""

from unittest.mock import patch

from ormah.engine.maintenance_signal import MAINTENANCE_DUE_SIGNAL
from ormah.engine.memory_engine import MemoryEngine, _embedding_text, _generate_title
from ormah.models.node import ConnectRequest, CreateNodeRequest, EdgeType, NodeType, UpdateNodeRequest


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


def test_whisper_onboarding_nudge(engine):
    """whisper fires onboarding nudge exactly once when identity is empty."""
    engine.settings.claude_maintenance_enabled = True
    with patch.object(
        engine.context_builder,
        "build_whisper_context",
        return_value=f"# Ormah whispers\n{MAINTENANCE_DUE_SIGNAL}",
    ):
        text = engine.get_whisper_context("hello")
    assert "onboarding" in text.lower()
    assert "maintenance_due" not in text
    assert "STOP" in text
    assert "do not start any work" in text
    assert "What should I call you?" in text
    assert "LinkedIn may be blocked" in text
    assert "authenticated access" in text
    assert "mine it thoroughly" in text
    assert "not just a short summary" in text
    assert "5-15 self-contained memories" in text
    assert "Use 'profile' only as a tag" in text
    assert "never as the memory type" in text
    assert "do not infer details" in text
    assert "short bio instead" in text
    assert "Name + optional onboarding material" in text
    assert text.index("What should I call you?") < text.index("GitHub")
    assert "working with AI agents" not in text
    assert "User skipped onboarding" in text

    with patch.object(engine.context_builder, "build_whisper_context", return_value=""):
        text2 = engine.get_whisper_context("hello again")
    assert "onboarding" not in text2.lower()


def test_whisper_onboarding_debug_keeps_injected_ids(engine):
    engine.settings.whisper_reranker_enabled = False
    with patch.object(
        engine.context_builder,
        "build_whisper_context",
        return_value=(f"whisper text\n{MAINTENANCE_DUE_SIGNAL}", ["mem-1"]),
    ):
        text, injected_ids = engine.get_whisper_context("hello", _return_debug=True)

    assert "whisper text" in text
    assert "whisper text\n\n## Ormah" in text
    assert "maintenance_due" not in text
    assert "onboarding" in text.lower()
    assert injected_ids == ["mem-1"]


def test_whisper_onboarding_strips_legacy_maintenance_marker(engine):
    engine.settings.whisper_reranker_enabled = False
    with patch.object(
        engine.context_builder,
        "build_whisper_context",
        return_value=("whisper text\nmaintenance_due", ["mem-1"]),
    ):
        text, injected_ids = engine.get_whisper_context("hello", _return_debug=True)

    assert "whisper text" in text
    assert "maintenance_due" not in text
    assert "onboarding" in text.lower()
    assert injected_ids == ["mem-1"]


def test_whisper_onboarding_suppressed_when_identity_exists(engine):
    req = CreateNodeRequest(
        content="User's name is Alice.",
        type=NodeType.fact,
        title="User name",
        about_self=True,
    )
    engine.remember(req)

    with patch.object(engine.context_builder, "build_whisper_context", return_value=""):
        text = engine.get_whisper_context("hello")

    assert "onboarding" not in text.lower()


def test_stats(engine):
    req = CreateNodeRequest(content="A fact.", type=NodeType.fact)
    engine.remember(req)

    stats = engine.stats()
    # +1 for the self node created on startup
    assert stats["store"]["total_nodes"] == 2


def test_warmup_reranker_marks_unavailable_when_cache_missing(settings):
    with (
        patch("ormah.engine.memory_engine.MemoryEngine._warmup_embedder"),
        patch("ormah.embeddings.reranker.model_is_cached", return_value=False),
        patch("ormah.embeddings.reranker.preload_model") as preload_model,
    ):
        from ormah.engine.memory_engine import MemoryEngine

        engine = MemoryEngine(settings)
        engine.startup()

    assert engine._whisper_reranker_available is False
    preload_model.assert_not_called()
    engine.shutdown()


def test_get_whisper_context_disables_reranker_when_feature_disabled(engine):
    engine.settings.whisper_reranker_enabled = False
    engine._whisper_reranker_available = False

    with patch.object(engine.context_builder, "build_whisper_context", return_value="") as build:
        engine.get_whisper_context("auth prompt")

    assert build.call_args.kwargs["reranker_enabled"] is False


def test_get_whisper_context_uses_reranker_when_available(engine):
    engine._whisper_reranker_available = True

    with patch.object(engine.context_builder, "build_whisper_context", return_value="") as build:
        engine.get_whisper_context("auth prompt")

    assert build.call_args.kwargs["reranker_enabled"] is True


def test_startup_seeds_maintenance_grace_period(settings):
    settings.claude_maintenance_enabled = True
    engine = MemoryEngine(settings)
    try:
        engine.startup()
        row = engine.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'last_maintenance_run'"
        ).fetchone()
        assert row is not None
    finally:
        engine.shutdown()


def test_startup_preserves_existing_maintenance_timestamp(settings):
    settings.claude_maintenance_enabled = True
    existing = "2026-01-01T00:00:00+00:00"
    engine = MemoryEngine(settings)
    try:
        engine.db.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_maintenance_run', ?)",
            (existing,),
        )
        engine.db.conn.commit()
        engine.startup()
        row = engine.db.conn.execute(
            "SELECT value FROM meta WHERE key = 'last_maintenance_run'"
        ).fetchone()
        assert row["value"] == existing
    finally:
        engine.shutdown()


def test_get_whisper_context_returns_empty_when_reranker_required_but_unavailable(engine):
    engine.settings.whisper_reranker_enabled = True
    engine._whisper_reranker_available = False
    engine.settings.claude_maintenance_enabled = False
    engine.db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('onboarding_prompted', '1')"
    )
    engine.db.conn.commit()

    with patch.object(
        engine.context_builder, "build_whisper_context", return_value="unused"
    ) as build:
        result = engine.get_whisper_context("auth prompt")

    assert result == ""
    build.assert_not_called()


def test_whisper_onboarding_works_when_reranker_unavailable(engine):
    engine.settings.whisper_reranker_enabled = True
    engine._whisper_reranker_available = False
    engine.settings.claude_maintenance_enabled = True

    with patch.object(
        engine.context_builder, "build_whisper_context", return_value="unused"
    ) as build:
        result = engine.get_whisper_context("auth prompt")

    assert "onboarding" in result.lower()
    assert "maintenance_due" not in result
    build.assert_not_called()
