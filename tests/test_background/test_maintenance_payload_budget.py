"""Phase 1 of run_maintenance stays inline: the result size has an upper bound."""

from __future__ import annotations

import re
import uuid

import pytest

from ormah.adapters.mcp_adapter import _format_maintenance_batches

INLINE_LIMIT = 50_000  # Claude Code saves a larger MCP text result to a file
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
PAIR_BATCHES = ("link_candidates", "conflict_candidates", "merge_candidates")


def _node(title_len: int = 60, content_len: int = 400, title: str | None = None) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": title if title is not None else "t" * title_len,
        "type": "observation",
        "space": "ormah",
        "content": "c" * content_len,
    }


def _pair(a: dict | None = None, b: dict | None = None) -> dict:
    return {"node_a": a or _node(), "node_b": b or _node(), "similarity": 0.91}


def _cluster(n: int = 5) -> list[dict]:
    return [_node() for _ in range(n)]


def _non_bmp_node() -> dict:
    # Every field over its cap, in a character that JS ``.length`` counts as 2.
    return {"id": str(uuid.uuid4()), "title": "\U0001F600" * 300, "type": "fact",
            "space": "\U0001F600" * 100, "content": "\U0001F600" * 500}


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _patch_finders(monkeypatch, link=(), conflict=(), merge=(), clusters=()):
    from ormah.background import auto_linker, conflict_detector, consolidator, duplicate_merger

    monkeypatch.setattr(auto_linker, "_find_link_candidates", lambda engine, limit: list(link))
    monkeypatch.setattr(
        conflict_detector, "_find_conflict_candidates", lambda engine, limit: list(conflict)
    )
    monkeypatch.setattr(
        duplicate_merger, "_find_merge_candidates", lambda engine, limit: list(merge)
    )
    monkeypatch.setattr(
        consolidator, "_find_consolidation_clusters", lambda engine, limit: list(clusters)
    )


def _format(batches: dict) -> str:
    return _format_maintenance_batches(batches, "job-" + "0" * 32, "2026-10-13T12:00:00+00:00")


def _ids_in_batches(batches: dict) -> set[str]:
    ids = set()
    for name in PAIR_BATCHES:
        for c in batches[name]:
            ids.update((c["node_a"]["id"], c["node_b"]["id"]))
    for cluster in batches["consolidation_clusters"]:
        ids.update(n["id"] for n in cluster)
    return ids


def _full_backlog(monkeypatch) -> dict:
    """The worked example: 25 pairs per batch and 4 clusters of 5 nodes."""
    backlog = {
        "link": [_pair() for _ in range(25)],
        "conflict": [_pair() for _ in range(25)],
        "merge": [_pair() for _ in range(25)],
        "clusters": [_cluster() for _ in range(4)],
    }
    _patch_finders(monkeypatch, **backlog)
    return backlog


class TestPhase1ResultSize:

    def test_full_backlog_formats_under_inline_limit(self, engine, monkeypatch):
        _full_backlog(monkeypatch)

        text = _format(engine.get_maintenance_batches())

        assert len(text) < INLINE_LIMIT

    def test_text_and_issued_batches_hold_the_same_candidates(self, engine, monkeypatch):
        # Passes on the pre-fix code too (nothing was cut); guards a future formatter cut.
        _full_backlog(monkeypatch)

        batches = engine.get_maintenance_batches()
        text = _format(batches)

        assert set(UUID_RE.findall(text)) == _ids_in_batches(batches)

    def test_long_title_stays_under_inline_limit(self, engine, monkeypatch):
        backlog = _full_backlog(monkeypatch)
        backlog["link"][0] = _pair(a=_node(title="T" * 100_001))
        _patch_finders(monkeypatch, **backlog)

        text = _format(engine.get_maintenance_batches())

        assert len(text) < INLINE_LIMIT

    def test_small_backlog_is_not_cut(self, engine, monkeypatch):
        _patch_finders(
            monkeypatch, link=[_pair(), _pair()], merge=[_pair()], clusters=[_cluster(3)]
        )

        bounded = engine.get_maintenance_batches()
        engine.settings.claude_maintenance_payload_max_chars = 10**9
        unbounded = engine.get_maintenance_batches()

        assert bounded == unbounded
        assert bounded["summary"] == "2 link candidates, 1 merge candidates, 1 cluster(s)"


class TestBudgetAllocation:

    def test_tight_budget_gives_every_batch_some_work(self, engine, monkeypatch):
        _full_backlog(monkeypatch)
        engine.settings.claude_maintenance_payload_max_chars = 10_000

        batches = engine.get_maintenance_batches()

        for name in (*PAIR_BATCHES, "consolidation_clusters"):
            assert batches[name], f"{name} starved"
        assert sum(len(batches[name]) for name in PAIR_BATCHES) < 75

    def test_summary_counts_only_emitted_candidates(self, engine, monkeypatch):
        _full_backlog(monkeypatch)
        engine.settings.claude_maintenance_payload_max_chars = 9_000

        batches = engine.get_maintenance_batches()

        # A pair costs 1,104, a cluster 2,712; section headers 128 (link, conflict), 184 (merge),
        # 88 (clusters). Turn 1 costs 6,552; turn 2 fits link and conflict, then 240 remain.
        assert batches["summary"] == (
            "2 link candidates, 2 conflict candidates, 1 merge candidates, 1 cluster(s)"
        )
        assert [len(batches[n]) for n in (*PAIR_BATCHES, "consolidation_clusters")] == [2, 2, 1, 1]

    def test_batch_that_does_not_fit_leaves_the_turn_and_others_continue(
        self, engine, monkeypatch
    ):
        big_cluster = [_node(title_len=200) for _ in range(5)]
        _patch_finders(
            monkeypatch,
            link=[_pair() for _ in range(5)],
            conflict=[_pair() for _ in range(5)],
            merge=[_pair() for _ in range(5)],
            clusters=[big_cluster],
        )
        # One turn of pairs costs 1,232 + 1,232 + 1,288; the cluster then needs 3,500 but
        # 3,400 remain, and the second turn of pairs needs 3 x 1,104 = 3,312.
        engine.settings.claude_maintenance_payload_max_chars = 3_752 + 3_400

        batches = engine.get_maintenance_batches()

        assert batches["consolidation_clusters"] == []
        assert [len(batches[n]) for n in PAIR_BATCHES] == [2, 2, 2]

    @pytest.mark.parametrize("budget", [2_000, 3_000])
    def test_pair_in_two_batches_is_emitted_or_cut_whole(self, engine, monkeypatch, budget):
        pair = _pair()
        _patch_finders(monkeypatch, link=[pair], merge=[dict(pair, score=0.9)])
        # The link occurrence with its section header costs 1,232; both cost 1,232 + 1,288.
        engine.settings.claude_maintenance_payload_max_chars = budget

        batches = engine.get_maintenance_batches()

        in_link = len(batches["link_candidates"]) == 1
        in_merge = len(batches["merge_candidates"]) == 1
        assert in_link == in_merge
        assert in_link == (budget >= 2_520)


class TestFormattedSizeBound:
    """The estimate is an upper bound: the formatted sections never pass the budget."""

    @staticmethod
    def _worst_case_node(content_len: int) -> dict:
        # Empty title and space and a content the formatter prints whole maximize labels per char.
        return {"id": str(uuid.uuid4()), "title": "", "type": "fact", "space": "",
                "content": "c" * content_len}

    @pytest.mark.parametrize("budget", [5_000, 20_000, 45_000, 90_000])
    def test_sections_fit_the_budget(self, engine, monkeypatch, budget):
        def pair():
            return {"node_a": self._worst_case_node(300), "node_b": self._worst_case_node(300),
                    "similarity": -0.5}

        _patch_finders(
            monkeypatch,
            link=[pair() for _ in range(150)],
            conflict=[pair() for _ in range(150)],
            merge=[pair() for _ in range(150)],
            clusters=[[self._worst_case_node(200) for _ in range(5)] for _ in range(4)],
        )
        engine.settings.claude_maintenance_payload_max_chars = budget

        batches = engine.get_maintenance_batches()
        text = _format(batches)
        sections = text[text.index("## "):]

        assert len(batches["link_candidates"]) < 150  # the budget binds
        assert len(sections) <= budget

    @pytest.mark.parametrize("budget", [5_000, 45_000, 90_000])
    def test_non_bmp_sections_fit_the_budget_in_utf16_units(self, engine, monkeypatch, budget):
        # Claude Code measures the result with JS ``.length``, so the bound is in UTF-16 units.
        def pair():
            return {"node_a": _non_bmp_node(), "node_b": _non_bmp_node(), "similarity": 0.9}

        _patch_finders(
            monkeypatch,
            link=[pair() for _ in range(40)],
            conflict=[pair() for _ in range(40)],
            merge=[pair() for _ in range(40)],
            clusters=[[_non_bmp_node() for _ in range(5)] for _ in range(4)],
        )
        engine.settings.claude_maintenance_payload_max_chars = budget

        batches = engine.get_maintenance_batches()
        text = _format(batches)
        sections = text[text.index("## "):]

        assert batches["link_candidates"]  # the run still issues work
        assert _utf16_len(sections) <= budget

    def test_non_bmp_cluster_fits_the_validator_minimum(self, engine, monkeypatch):
        # The settings validator accepts a budget of max_item_cost: a cluster at the field
        # caps must fit it, or the consolidation batch never goes out.
        from ormah.engine.maintenance_budget import max_item_cost

        nodes = engine.settings.consolidation_max_cluster_nodes
        _patch_finders(monkeypatch, clusters=[[_non_bmp_node() for _ in range(nodes)]])
        engine.settings.claude_maintenance_payload_max_chars = max_item_cost(nodes)

        assert len(engine.get_maintenance_batches()["consolidation_clusters"]) == 1


class TestOverheadCalibration:
    """The estimate of one item is never below what the formatter prints for it."""

    @staticmethod
    def _smallest_budget_that_emits_all(batches: dict) -> int:
        from ormah.engine.maintenance_budget import fit_to_budget

        def emits_all(budget: int) -> bool:
            cut = fit_to_budget(batches, budget)
            return all(len(cut[name]) == len(batches[name]) for name in batches)

        low, high = 0, 10**7
        while low < high:
            mid = (low + high) // 2
            low, high = (low, mid) if emits_all(mid) else (mid + 1, high)
        return low

    @pytest.mark.parametrize("count", [1, 100])
    @pytest.mark.parametrize("batch", [*PAIR_BATCHES, "consolidation_clusters"])
    def test_formatted_section_fits_its_estimate(self, batch, count):
        def node(content_len):
            return {"id": str(uuid.uuid4()), "title": "", "type": "fact", "space": "",
                    "content": "c" * content_len}

        if batch == "consolidation_clusters":
            items = [[node(200) for _ in range(5)] for _ in range(count)]
        else:
            items = [{"node_a": node(300), "node_b": node(300), "similarity": -0.5}
                     for _ in range(count)]
        batches = {name: [] for name in (*PAIR_BATCHES, "consolidation_clusters")}
        batches[batch] = items

        estimate = self._smallest_budget_that_emits_all(batches)
        text = _format(dict(batches, summary="x"))

        assert len(text[text.index("## "):]) <= estimate


class TestCutPairStaysPending:

    def test_cut_pair_group_returns_in_a_later_run(self, engine):
        """A link verdict on an emitted occurrence must not hide the merge occurrence."""
        from ormah.background.duplicate_merger import _find_merge_candidates
        from ormah.models.node import CreateNodeRequest, NodeType

        engine.settings.auto_link_similarity_threshold = 999.0  # remember() must not link them
        ids = [
            engine.remember(CreateNodeRequest(
                content="Python uses indentation to define code block scope.",
                type=NodeType.fact,
                title="Python indentation defines block scope",
            ))[0]
            for _ in range(2)
        ]
        engine.settings.auto_link_similarity_threshold = 0.0
        pair = tuple(sorted(ids))

        def pairs(batch: list[dict]) -> set[tuple[str, str]]:
            return {tuple(sorted((c["node_a"]["id"], c["node_b"]["id"]))) for c in batch}

        def apply_none(checked: set[tuple[str, str]]) -> None:
            engine.apply_maintenance_results({
                "edges": [
                    {"node_a_id": a, "node_b_id": b, "edge_type": "none", "reason": "phase 2"}
                    for a, b in checked
                ]
            })

        # Settle every other pair (the self node pairs with both), so this pair leads each batch.
        engine.settings.claude_maintenance_payload_max_chars = 10**9
        full = engine.get_maintenance_batches()
        apply_none(set().union(*(pairs(full[name]) for name in PAIR_BATCHES)) - {pair})
        full = engine.get_maintenance_batches()
        assert pairs(full["link_candidates"]) == {pair}
        assert pairs(full["merge_candidates"]) == {pair}

        # One occurrence with its section header fits; the whole group does not.
        engine.settings.claude_maintenance_payload_max_chars = 600
        issued = engine.get_maintenance_batches()
        apply_none(set().union(*(pairs(issued[name]) for name in PAIR_BATCHES)))

        assert pair in pairs(_find_merge_candidates(engine, limit=25))
