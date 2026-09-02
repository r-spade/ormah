"""Tests for the memory engine."""

import threading
import time
from unittest.mock import MagicMock, patch

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


def test_warmup_reranker_provisions_model_when_cache_missing(settings):
    with (
        patch("ormah.engine.memory_engine.MemoryEngine._warmup_embedder"),
        patch("ormah.embeddings.reranker.model_is_cached", return_value=False),
        patch("ormah.embeddings.reranker.preload_model") as preload_model,
    ):
        from ormah.engine.memory_engine import MemoryEngine

        engine = MemoryEngine(settings)
        engine.startup()

    assert engine._whisper_reranker_available is True
    preload_model.assert_called_once_with(settings.whisper_reranker_model)
    engine.shutdown()


def test_warmup_reranker_degrades_when_provisioning_fails(settings):
    with (
        patch("ormah.engine.memory_engine.MemoryEngine._warmup_embedder"),
        patch("ormah.embeddings.reranker.model_is_cached", return_value=False),
        patch(
            "ormah.embeddings.reranker.preload_model",
            side_effect=RuntimeError("network unavailable"),
        ),
    ):
        from ormah.engine.memory_engine import MemoryEngine

        engine = MemoryEngine(settings)
        engine.startup()

    assert engine._whisper_reranker_available is False
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


def test_get_whisper_context_passes_user_node_id(engine):
    """Identity protection must be active on the production call path (I3)."""
    engine._whisper_reranker_available = True

    with patch.object(engine.context_builder, "build_whisper_context", return_value="") as build:
        engine.get_whisper_context("where do I live")

    assert build.call_args.kwargs["user_node_id"] == engine.user_node_id
    assert engine.user_node_id is not None


def test_get_whisper_context_threads_pool_and_content_cap_settings(engine):
    engine._whisper_reranker_available = True
    engine.settings.whisper_candidate_pool_multiplier = 7
    engine.settings.whisper_injected_content_max_chars = 450

    with patch.object(engine.context_builder, "build_whisper_context", return_value="") as build:
        engine.get_whisper_context("auth prompt")

    assert build.call_args.kwargs["candidate_pool_multiplier"] == 7
    assert build.call_args.kwargs["injected_content_max_chars"] == 450


def test_get_whisper_context_threads_preference_applicability_settings(engine):
    engine._whisper_reranker_available = True
    engine.settings.whisper_preference_applicability_enabled = True
    engine.settings.whisper_preference_applicability_gate = 0.42
    engine.settings.whisper_preference_max_nodes = 1

    with patch.object(engine.context_builder, "build_whisper_context", return_value="") as build:
        engine.get_whisper_context("auth prompt")

    assert build.call_args.kwargs["preference_applicability_enabled"] is True
    assert build.call_args.kwargs["preference_applicability_gate"] == 0.42
    assert build.call_args.kwargs["preference_max_nodes"] == 1


def test_has_searchable_preferences_ignores_expired_and_archival(engine):
    now = "2026-07-12T00:00:00+00:00"
    with engine.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO nodes
                (id, type, tier, source, title, content, created, updated,
                 last_accessed, file_path, file_hash, valid_until)
            VALUES ('expired-pref', 'preference', 'working', 'test', 'Expired',
                    'Expired', ?, ?, ?, '/tmp/expired', 'expired',
                    '2020-01-01T00:00:00+00:00')
            """,
            (now, now, now),
        )
        conn.execute(
            """
            INSERT INTO nodes
                (id, type, tier, source, title, content, created, updated,
                 last_accessed, file_path, file_hash)
            VALUES ('archival-pref', 'preference', 'archival', 'test', 'Archival',
                    'Archival', ?, ?, ?, '/tmp/archival', 'archival')
            """,
            (now, now, now),
        )

    assert engine.has_searchable_preferences() is False

    with engine.db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO nodes
                (id, type, tier, source, title, content, created, updated,
                 last_accessed, file_path, file_hash)
            VALUES ('active-pref', 'preference', 'working', 'test', 'Active',
                    'Active', ?, ?, ?, '/tmp/active', 'active')
            """,
            (now, now, now),
        )

    assert engine.has_searchable_preferences() is True


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


def test_get_whisper_context_degrades_when_reranker_unavailable(engine):
    """Reranker unavailable (fresh install, model downloading) must degrade to
    embedding-only whisper with the raised cosine-scale gate — never go dark (I4)."""
    engine.settings.whisper_reranker_enabled = True
    engine._whisper_reranker_available = False
    engine.settings.claude_maintenance_enabled = False
    engine.db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('onboarding_prompted', '1')"
    )
    engine.db.conn.commit()

    with (
        patch.object(engine, "_refresh_whisper_reranker_if_cached", return_value=False),
        patch.object(engine.context_builder, "build_whisper_context", return_value="degraded whisper") as build,
    ):
        result = engine.get_whisper_context("auth prompt")

    assert result == "degraded whisper"
    build.assert_called_once()
    assert build.call_args.kwargs["reranker_enabled"] is False
    assert (
        build.call_args.kwargs["injection_gate"]
        == engine.settings.whisper_injection_gate_no_reranker
    )


def test_get_whisper_context_normal_gate_when_reranker_available(engine):
    engine._whisper_reranker_available = True

    with patch.object(
        engine.context_builder, "build_whisper_context", return_value=""
    ) as build:
        engine.get_whisper_context("auth prompt")

    assert build.call_args.kwargs["injection_gate"] == engine.settings.whisper_injection_gate


def test_get_whisper_context_loads_reranker_if_cached_after_startup(engine):
    """Desktop setup may cache the reranker after the server already started.

    The next whisper should notice the cached model, load it, and use the
    normal reranker gate without requiring a server restart.
    """
    engine.settings.whisper_reranker_enabled = True
    engine._whisper_reranker_available = False
    engine.settings.claude_maintenance_enabled = False
    engine.db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('onboarding_prompted', '1')"
    )
    engine.db.conn.commit()

    with (
        patch("ormah.embeddings.reranker.model_is_cached", return_value=True),
        patch("ormah.embeddings.reranker.preload_model") as preload_model,
        patch.object(
            engine.context_builder,
            "build_whisper_context",
            return_value="reranked",
        ) as build,
    ):
        result = engine.get_whisper_context("where do I live?")

    assert result == "reranked"
    assert engine._whisper_reranker_available is True
    preload_model.assert_called_once_with(engine.settings.whisper_reranker_model)
    assert build.call_args.kwargs["reranker_enabled"] is True
    assert build.call_args.kwargs["injection_gate"] == engine.settings.whisper_injection_gate


def test_get_whisper_context_does_not_download_reranker_when_not_cached(engine):
    """Whisper may load an already-cached reranker, but must not download it."""
    engine.settings.whisper_reranker_enabled = True
    engine._whisper_reranker_available = False
    engine.settings.claude_maintenance_enabled = False
    engine.db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('onboarding_prompted', '1')"
    )
    engine.db.conn.commit()

    with (
        patch("ormah.embeddings.reranker.model_is_cached", return_value=False),
        patch("ormah.embeddings.reranker.preload_model") as preload_model,
        patch.object(
            engine.context_builder,
            "build_whisper_context",
            return_value="degraded",
        ) as build,
    ):
        result = engine.get_whisper_context("where do I live?")

    assert result == "degraded"
    assert engine._whisper_reranker_available is False
    preload_model.assert_not_called()
    assert build.call_args.kwargs["reranker_enabled"] is False
    assert (
        build.call_args.kwargs["injection_gate"]
        == engine.settings.whisper_injection_gate_no_reranker
    )


def test_whisper_onboarding_works_when_reranker_unavailable(engine):
    engine.settings.whisper_reranker_enabled = True
    engine._whisper_reranker_available = False
    engine.settings.claude_maintenance_enabled = True

    with patch.object(
        engine.context_builder, "build_whisper_context", return_value="whisper body"
    ) as build:
        result = engine.get_whisper_context("auth prompt")

    assert "onboarding" in result.lower()
    assert "maintenance_due" not in result
    build.assert_called_once()


class TestRecallFloorAndSpaceOrdering:
    """Deliberate recall: wider pool, space scores before the cut, relevance
    floor instead of padding (I8)."""

    def _node(self, node_id, space=None):
        return {
            "id": node_id, "type": "fact", "tier": "working",
            "title": node_id, "content": f"content of {node_id}",
            "space": space, "access_count": 0,
            "last_accessed": "2026-01-01T00:00:00Z",
            "created": "2026-01-01T00:00:00Z",
        }

    def _search_mock(self, engine, results):
        mock_search = MagicMock()
        mock_search.search.return_value = results
        return patch.object(engine, "_get_hybrid_search", return_value=mock_search), mock_search

    def test_pool_widened_and_floor_drops_cross_space_noise(self, engine):
        """Cross-space noise penalized below the floor is dropped, not padded."""
        query_vec = [0.1, 0.2, 0.3]
        results = [
            {"node": self._node("good-1", space="proj"), "score": 0.80, "source": "hybrid"},
            {"node": self._node("good-2", space="proj"), "score": 0.62, "source": "hybrid"},
            {"node": self._node("noise-1", space="other"), "score": 0.50, "source": "hybrid"},
            {"node": self._node("noise-2", space="other"), "score": 0.48, "source": "hybrid"},
        ]
        ctx, mock_search = self._search_mock(engine, results)
        with ctx:
            out = engine.recall_search_structured(
                "project question", limit=4, default_space="proj",
                query_vec=query_vec,
            )

        # Pool widened to limit*3
        assert mock_search.search.call_args.kwargs.get("limit") == 12
        assert mock_search.search.call_args.kwargs["query_vec"] is query_vec
        ids = [r["node"]["id"] for r in out if r.get("source") == "hybrid"]
        # other-space results: 0.50*0.6=0.30 and 0.48*0.6=0.288 < 0.35 floor
        assert ids == ["good-1", "good-2"]
        assert len(out) < 4  # returns fewer rather than padding

    def test_space_penalty_decides_survival_not_just_order(self, engine):
        """A current-space match outside the old `limit` window survives the cut."""
        # limit=2: previously only the first 2 raw results were fetched at all.
        results = [
            {"node": self._node("cross-1", space="other"), "score": 0.90, "source": "hybrid"},
            {"node": self._node("cross-2", space="other"), "score": 0.88, "source": "hybrid"},
            {"node": self._node("local-1", space="proj"), "score": 0.60, "source": "hybrid"},
        ]
        ctx, _ = self._search_mock(engine, results)
        with ctx:
            out = engine.recall_search_structured(
                "project question", limit=2, default_space="proj",
            )

        ids = [r["node"]["id"] for r in out if r.get("source") == "hybrid"]
        # cross-1: 0.9*0.6=0.54, cross-2: 0.88*0.6=0.528, local-1: 0.60
        # local-1 now wins the ordering AND survives the cut.
        assert ids[0] == "local-1"

    def test_temporal_supplements_exempt_from_floor(self, engine):
        results = [
            {"node": self._node("recent-1", space="proj"), "score": 0.001, "source": "temporal"},
        ]
        ctx, _ = self._search_mock(engine, results)
        with ctx:
            out = engine.recall_search_structured(
                "project question", limit=4, default_space="proj",
            )

        assert any(r["node"]["id"] == "recent-1" for r in out)

    def test_temporal_preprocessing_invalidates_original_query_vector(self, engine):
        query_vec = [0.1, 0.2, 0.3]
        ctx, mock_search = self._search_mock(engine, [])

        with ctx:
            engine.recall_search_structured(
                "auth changes yesterday",
                limit=4,
                query_vec=query_vec,
            )

        assert mock_search.search.call_args.args[0] == "auth changes"
        assert mock_search.search.call_args.kwargs["query_vec"] is None

    def test_temporal_supplements_respect_space_priority(self, engine):
        """A newer other-space node must NOT outrank an older current-space node."""
        newer_other = self._node("newer-other", space="other")
        newer_other["created"] = "2026-02-01T00:00:00Z"
        older_current = self._node("older-current", space="proj")
        older_current["created"] = "2026-01-01T00:00:00Z"

        # No semantic hits: the supplement pulls SQL-recent nodes, which
        # get_recent_nodes returns in recency order (newer other-space first).
        ctx, _ = self._search_mock(engine, [])
        with ctx, patch.object(
            engine.graph, "get_recent_nodes",
            return_value=[newer_other, older_current],
        ):
            out = engine.recall_search_structured(
                "yesterday", limit=4, default_space="proj",
                created_after="2026-01-01T00:00:00Z",
            )

        ids = [r["node"]["id"] for r in out if r.get("source") == "temporal"]
        # Current-space wins despite being older; the other-space node is
        # demoted, not deleted.
        assert ids[0] == "older-current"
        assert set(ids) == {"older-current", "newer-other"}


# ---------------------------------------------------------------------------
# _get_hybrid_search concurrency (#27)
# ---------------------------------------------------------------------------


def test_get_hybrid_search_constructs_once_under_concurrency(engine):
    """Concurrent recalls must not each construct a HybridSearch (#27).

    Without locking, N threads all see ``_hybrid_search is None`` and each build
    their own instance (last-write-wins). The double-checked lock must serialize
    init so exactly one instance is constructed and every caller gets it.
    """
    import threading
    import time

    engine._hybrid_search = None
    n = 8
    construct_count = 0
    count_lock = threading.Lock()

    class FakeHybridSearch:
        def __init__(self, db, settings):
            nonlocal construct_count
            with count_lock:
                construct_count += 1
            time.sleep(0.05)  # widen the race window deterministically

    barrier = threading.Barrier(n)
    results: list = [None] * n

    def worker(i):
        barrier.wait()  # release all threads at once
        results[i] = engine._get_hybrid_search()

    with patch("ormah.embeddings.hybrid_search.HybridSearch", FakeHybridSearch):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert construct_count == 1
    assert all(r is results[0] for r in results)
    assert results[0] is engine._hybrid_search


def test_memory_restore_lock_excludes_live_remember(engine):
    started = threading.Event()
    finished = threading.Event()
    result: list[str] = []

    def writer():
        started.set()
        node_id, _ = engine.remember(
            CreateNodeRequest(content="Written after the restore swap.", type=NodeType.fact)
        )
        result.append(node_id)
        finished.set()

    with engine.memory_operation():
        thread = threading.Thread(target=writer)
        thread.start()
        assert started.wait(timeout=1)
        time.sleep(0.05)
        assert not finished.is_set()

    thread.join(timeout=2)
    assert finished.is_set()
    assert engine.file_store.load(result[0]) is not None


def test_a_restore_mid_run_aborts_every_job_without_partial_writes(engine):
    """#210 acceptance criterion, exercised across the real scheduler entry points."""
    import json
    from datetime import datetime, timedelta, timezone
    from unittest.mock import patch

    from ormah.background.auto_linker import run_auto_linker
    from ormah.background.decay_manager import run_decay
    from ormah.models.node import CreateNodeRequest, NodeType, Tier

    # remember() auto-links similar nodes at creation time; disable that for the
    # second node below so it stays unlinked to `stale_id` -- otherwise
    # auto_linker would find zero candidates and its bump would never fire,
    # making the epoch_before + 2 guard below vacuous rather than a real check.
    stale_id, _ = engine.remember(CreateNodeRequest(
        content="a stale working node", type=NodeType.fact, tier=Tier.working,
        title="stale"))
    real_auto_link = engine._auto_link_node
    engine._auto_link_node = lambda node: []
    engine.remember(CreateNodeRequest(
        content="a stale working node", type=NodeType.fact, title="stale twin"))
    engine._auto_link_node = real_auto_link
    # A bare SQL `datetime('now', '-30 days')` literal produces a naive,
    # space-separated string; decay_manager's anchor parse still succeeds on it
    # (fromisoformat accepts the space), but the later `now - anchor` then mixes
    # a tz-aware `now` with a naive anchor, raises TypeError, and the node is
    # silently skipped -- vacuous for this test. Use the same tz-aware ISO
    # string tests/test_background/test_decay_manager.py's _make_stale uses.
    stale_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?", (stale_date, stale_id))
    engine.db.conn.commit()

    edges_before = engine.db.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
    epoch_before = engine.restore_epoch

    # The bump must land AFTER the job read its entry epoch, never before the call:
    # restore_aware_job reads engine.restore_epoch at call time, so a pre-call bump would
    # simply hand the job the new value and there would be no mismatch to detect. Each job
    # below gets the bump at the point where a real restore would land in its own run.
    from ormah import lifecycle

    real_retrievability = lifecycle.retrievability
    bumped = {"done": False}

    def bump_then_compute(days_since, stability, **kwargs):
        """decay's seam: fires once per candidate in the unlocked outer scan."""
        if not bumped["done"]:
            bumped["done"] = True
            engine._restore_epoch += 1
        return real_retrievability(days_since, stability, **kwargs)

    lifecycle.retrievability = bump_then_compute
    try:
        run_decay(engine)  # returns cleanly
    finally:
        lifecycle.retrievability = real_retrievability

    row = engine.db.conn.execute(
        "SELECT tier FROM nodes WHERE id = ?", (stale_id,)).fetchone()
    assert row["tier"] == "working"  # not demoted

    engine.settings.llm_provider = "ollama"

    def bump_then_link(*args, **kwargs):
        """auto_linker's seam: the unlocked LLM call, right before its apply step."""
        engine._restore_epoch += 1
        return json.dumps({"relationship": "supports", "reason": "x"})

    with patch("ormah.background.llm_client.llm_generate", side_effect=bump_then_link):
        run_auto_linker(engine)  # also returns cleanly

    # Guard against silent vacuousness: both assertions hold trivially if neither job ever
    # reached an apply step. Each bump lives inside that job's own seam, so an epoch that
    # moved twice is proof both jobs actually got there.
    assert engine.restore_epoch == epoch_before + 2, \
        "a job never reached its apply step — the fixture stopped exercising it"
    edges_after = engine.db.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
    assert edges_after == edges_before
