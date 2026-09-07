"""Contract tests for issue #220: surfacing must not be confirmed use.

Every assertion reads the four lifecycle fields from BOTH the markdown file and
the SQLite row. A test that checked only the database would pass while the file
rotted, and vice versa.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api.routes_ui import router as ui_router
from ormah.config import Settings
from ormah.engine.memory_engine import MemoryEngine
from ormah.models.node import CreateNodeRequest

LIFECYCLE_FIELDS = ("access_count", "last_accessed", "stability", "last_review")


def _snapshot(engine, node_id):
    """Capture the four lifecycle fields from the markdown file and the DB row."""
    node = engine.file_store.load(node_id)
    row = engine.db.conn.execute(
        "SELECT access_count, last_accessed, stability, last_review FROM nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    return {
        "file": tuple(getattr(node, f) for f in LIFECYCLE_FIELDS),
        "db": tuple(row[f] for f in LIFECYCLE_FIELDS),
    }


def _make_nodes(engine, count=2):
    """Create *count* nodes that a search for 'caching' will match."""
    ids = []
    for i in range(count):
        node_id, _ = engine.remember(CreateNodeRequest(
            content=f"caching architecture note number {i}",
            title=f"Caching {i}",
            type="fact",
            tier="working",
        ))
        ids.append(node_id)
    return ids


@pytest.fixture
def fts_only(engine):
    """Force the FTS fallback path by removing hybrid search."""
    with patch.object(engine, "_get_hybrid_search", return_value=None):
        yield engine


# --- Non-mutation contracts (issue #220 acceptance criteria) ---------------

def test_recall_search_does_not_write_lifecycle(engine):
    """Contract 1: broad formatted recall over N nodes mutates nothing."""
    ids = _make_nodes(engine)
    before = {i: _snapshot(engine, i) for i in ids}

    engine.recall_search("caching architecture", limit=10)

    for node_id in ids:
        assert _snapshot(engine, node_id) == before[node_id], (
            f"recall_search mutated lifecycle fields on {node_id}"
        )


def test_recall_search_fts_fallback_does_not_write_lifecycle(fts_only):
    """Contract 2: the FTS fallback path mutates nothing either."""
    engine = fts_only
    ids = _make_nodes(engine)
    before = {i: _snapshot(engine, i) for i in ids}

    engine.recall_search("caching architecture", limit=10)

    for node_id in ids:
        assert _snapshot(engine, node_id) == before[node_id]


def test_recall_search_structured_does_not_write_lifecycle(engine):
    """Contract 3: called with no lifecycle kwarg — the default was the bug."""
    ids = _make_nodes(engine)
    before = {i: _snapshot(engine, i) for i in ids}

    engine.recall_search_structured("caching architecture", limit=10)

    for node_id in ids:
        assert _snapshot(engine, node_id) == before[node_id]


def test_recall_search_structured_fts_fallback_does_not_write_lifecycle(fts_only):
    """Contract 4: same for the FTS fallback."""
    engine = fts_only
    ids = _make_nodes(engine)
    before = {i: _snapshot(engine, i) for i in ids}

    engine.recall_search_structured("caching architecture", limit=10)

    for node_id in ids:
        assert _snapshot(engine, node_id) == before[node_id]


def test_ui_search_route_does_not_write_lifecycle(tmp_memory_dir):
    """Contract 5: the UI search route.

    This is the test that fails on clean upstream/main: routes_ui.search_nodes
    calls recall_search_structured without the kwarg, so the True default
    reinforced every result. Exercised through the route, not the engine.
    """
    settings = Settings(memory_dir=tmp_memory_dir, backup_dir=tmp_memory_dir.parent / "backups")
    engine = MemoryEngine(settings)
    engine.startup()
    try:
        ids = _make_nodes(engine)
        before = {i: _snapshot(engine, i) for i in ids}

        app = FastAPI()
        app.include_router(ui_router)
        app.state.engine = engine
        with TestClient(app) as client:
            resp = client.get("/ui/search", params={"q": "caching architecture"})
        assert resp.status_code == 200

        for node_id in ids:
            assert _snapshot(engine, node_id) == before[node_id], (
                f"UI search mutated lifecycle fields on {node_id}"
            )
    finally:
        engine.shutdown()


def test_whisper_does_not_write_lifecycle(engine):
    """Contract 6: whisper still mutates nothing after losing its flag.

    Whisper was already correct (it passed touch_access=False). This pins that
    it stays correct once the flag is gone.
    """
    from ormah.engine.context_builder import ContextBuilder

    ids = _make_nodes(engine)
    before = {i: _snapshot(engine, i) for i in ids}

    builder = ContextBuilder(engine.graph, engine=engine)
    builder.build_whisper_context("caching architecture", space=None, max_nodes=8)

    for node_id in ids:
        assert _snapshot(engine, node_id) == before[node_id]


def test_concurrent_confirmed_use_does_not_lose_increments(engine):
    """Issue #220: _record_confirmed_use is atomic across its read-modify-write.

    Without @_serialized_memory_operation, two threads can both load the same
    access_count and both save count+1, collapsing two confirmations into one.
    """
    import threading

    ids = _make_nodes(engine, count=1)
    target = ids[0]
    before = engine.file_store.load(target).access_count

    threads = [threading.Thread(target=engine._record_confirmed_use, args=(target,))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    after = engine.file_store.load(target)
    assert after.access_count == before + 8, (
        f"lost increments: expected {before + 8}, got {after.access_count}"
    )
    row = engine.db.conn.execute(
        "SELECT access_count FROM nodes WHERE id = ?", (target,)
    ).fetchone()
    assert row["access_count"] == after.access_count, "file and DB disagree after concurrency"


# --- Confirmed-use contracts ----------------------------------------------

def _seed_whisper_log(engine, node_id, prompt="what about caching?"):
    """Insert a whisper_log row so submit_feedback can resolve one.

    submit_feedback attaches feedback to a whisper/recall event; without a row
    it returns an error string instead of recording anything.
    """
    engine.recall_search(prompt, limit=10)
    row = engine.db.conn.execute(
        "SELECT id FROM whisper_log WHERE node_id = ? ORDER BY id DESC LIMIT 1",
        (node_id,),
    ).fetchone()
    assert row is not None, "no whisper_log row was created — check the surface used"
    return row["id"]


def test_recall_node_confirms_only_the_requested_node(engine):
    """Contract 7: recall_node confirms the node asked for, never its neighbours."""
    from ormah.models.node import CreateNodeRequest

    target, _ = engine.remember(CreateNodeRequest(
        content="caching architecture target node", title="Target", type="fact", tier="working",
    ))
    neighbour, _ = engine.remember(CreateNodeRequest(
        content="caching architecture neighbour node", title="Neighbour", type="fact",
        tier="working",
    ))
    engine.graph.conn.execute(
        "INSERT INTO edges (source_id, target_id, edge_type, weight, created) "
        "VALUES (?, ?, 'related_to', 1.0, '2026-01-01T00:00:00Z')",
        (target, neighbour),
    )

    before_target = _snapshot(engine, target)
    before_neighbour = _snapshot(engine, neighbour)

    engine.recall_node(target)

    assert _snapshot(engine, target) != before_target, "recall_node did not confirm its node"
    assert _snapshot(engine, neighbour) == before_neighbour, (
        "recall_node confirmed a neighbour — only the requested node counts"
    )


@pytest.mark.parametrize("source", ["explicit", "implicit", "auto_llm_judge"])
def test_qualified_positive_feedback_confirms_use(engine, source):
    """Contract 8: the three allowlisted sources confirm, with signal == 1."""
    ids = _make_nodes(engine, count=2)
    target, other = ids[0], ids[1]
    log_id = _seed_whisper_log(engine, target)

    before_target = _snapshot(engine, target)
    before_other = _snapshot(engine, other)

    engine.submit_feedback(target, signal=1, source=source, whisper_log_id=log_id)

    assert _snapshot(engine, target) != before_target, (
        f"positive {source} feedback did not confirm use"
    )
    assert _snapshot(engine, other) == before_other, "an unrelated node was confirmed"


def test_auto_heuristic_positive_does_not_confirm(engine):
    """Contract 9: auto_heuristic is excluded pending #218 — fail-closed."""
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    before = _snapshot(engine, target)
    engine.submit_feedback(target, signal=1, source="auto_heuristic", whisper_log_id=log_id)

    assert _snapshot(engine, target) == before, "auto_heuristic must not confirm use"


@pytest.mark.parametrize("source", ["explicit", "implicit", "auto_llm_judge", "auto_heuristic"])
def test_negative_feedback_never_confirms(engine, source):
    """Contract 10: -1 is evidence about the prompt/node pair, never a confirmed use."""
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    before = _snapshot(engine, target)
    engine.submit_feedback(target, signal=-1, source=source, whisper_log_id=log_id)

    assert _snapshot(engine, target) == before, (
        f"negative {source} feedback changed lifecycle fields"
    )


@pytest.mark.parametrize(
    "signal,source,was_injected,should_promote",
    [
        (1, "explicit", 1, True),
        (1, "auto_heuristic", 1, False),
        (-1, "explicit", 1, False),
        (1, "explicit", 0, False),
    ],
)
def test_only_qualified_positives_promote(engine, signal, source, was_injected, should_promote):
    """#223: promotion needs a qualified positive on an event the agent actually saw."""
    from ormah.models.node import Tier

    node_id = _make_nodes(engine, count=1)[0]

    # Seed the whisper-log event BEFORE demoting: _seed_whisper_log goes through
    # recall_search, and an archival node scores below the relevance floor, so no
    # row would be created. The matrix is about signal/source/was_injected, not
    # about ranking — seeding first keeps the event identical either way.
    if was_injected:
        whisper_log_id = _seed_whisper_log(engine, node_id)
    else:
        whisper_log_id = _seed_held_back_whisper_log(engine, node_id)

    node = engine.file_store.load(node_id)
    node.tier = Tier.archival
    engine.builder.index_single(engine.file_store.save(node))

    engine.submit_feedback(node_id, signal=signal, source=source, whisper_log_id=whisper_log_id)

    expected = Tier.working if should_promote else Tier.archival
    assert engine.file_store.load(node_id).tier is expected, (
        f"signal={signal} source={source} was_injected={was_injected}: "
        f"expected {expected}, got {engine.file_store.load(node_id).tier}"
    )


# --- Idempotency contracts (second council round: the latch, not affinity) ---

def test_replaying_the_same_positive_feedback_confirms_once(engine):
    """Contract 10a: one confirmed-use event reinforces at most once.

    affinity and signals both use ON CONFLICT DO NOTHING, so a replayed request
    records no new evidence yet still returns success. Reinforcing on every call
    would let a retried tool call or a double-click manufacture retention.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)
    after_first = _snapshot(engine, target)

    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)

    assert _snapshot(engine, target) == after_first, (
        "replaying the same positive feedback reinforced twice"
    )


def test_negative_then_positive_feedback_confirms(engine):
    """Contract 10b: a first-time positive confirms even after a negative.

    The negative claims nothing (it does not qualify), so the later positive is
    still the event's first confirmation. This is the case a naive 'did the
    signals INSERT add a row?' gate gets wrong: the unique key is
    (whisper_log_id, signal_type, source) with no polarity, so the second call
    hits ON CONFLICT DO NOTHING even though it is a genuine first confirmation.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    engine.submit_feedback(target, signal=-1, source="explicit", whisper_log_id=log_id)
    after_negative = _snapshot(engine, target)

    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)

    assert _snapshot(engine, target) != after_negative, (
        "the event's first qualified positive did not confirm use"
    )


def test_second_source_on_an_already_confirmed_event_does_not_reconfirm(engine):
    """Contract 10c: the event is confirmed once, not once per source.

    This is the mirror failure: source is part of the signals unique key, so an
    implicit-positive followed by an explicit-positive DOES insert a second
    signals row. The event was already claimed; it must not reinforce again.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    engine.submit_feedback(target, signal=1, source="implicit", whisper_log_id=log_id)
    after_first = _snapshot(engine, target)

    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)

    assert _snapshot(engine, target) == after_first, (
        "a second positive source reconfirmed an already-confirmed event"
    )


def test_polarity_cycle_confirms_once(engine):
    """Contract 10d: +1 / -1 / +1 reinforces at most once — not twice.

    This is the false positive that killed the affinity-derived gate. affinity
    has one row per (node_id, whisper_log_id) and explicit feedback UPDATEs its
    signal in place, so reading affinity would see false->true twice and
    reinforce twice. The claim latch is never deleted, so the third call takes
    nothing.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)
    after_first_positive = _snapshot(engine, target)

    engine.submit_feedback(target, signal=-1, source="explicit", whisper_log_id=log_id)
    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)

    assert _snapshot(engine, target) == after_first_positive, (
        "a polarity cycle reinforced the same event twice"
    )


def test_unqualified_affinity_does_not_block_a_later_qualified_positive(engine):
    """Contract 10e: a pre-existing auto_heuristic row must not swallow a real use.

    This is the false negative that killed the affinity-derived gate. The
    affinity unique key is (node_id, whisper_log_id) and only explicit feedback
    UPDATEs the row, so an auto_heuristic positive makes a later implicit
    positive a no-op INSERT that leaves source = auto_heuristic. Reading
    affinity would keep the gate false forever and lose the reinforcement in
    silence. The claim latch does not consult affinity at all.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    engine.submit_feedback(target, signal=1, source="auto_heuristic", whisper_log_id=log_id)
    after_heuristic = _snapshot(engine, target)

    engine.submit_feedback(target, signal=1, source="implicit", whisper_log_id=log_id)

    assert _snapshot(engine, target) != after_heuristic, (
        "a prior auto_heuristic affinity row blocked a genuine confirmed use"
    )


def test_reinforcement_failure_does_not_fail_the_feedback_call(engine):
    """Contract 10f: a raising mutator is logged, not propagated.

    The route returns submit_feedback's value directly, so an exception after
    COMMIT would 500 a call whose affinity and signals rows are already durably
    written. ZeroDivisionError is the realistic case, not a contrived one:
    stability is Field(default=1.0, ge=0.0), so zero is legal, and the mutator
    divides by it. Under the at-most-once contract this reinforcement is lost —
    that is the accepted cost, but it must be a logged miss, not an API error.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    before = _snapshot(engine, target)

    with patch.object(
        engine, "_record_confirmed_use", side_effect=ZeroDivisionError("float division by zero")
    ):
        message = engine.submit_feedback(
            target, signal=1, source="explicit", whisper_log_id=log_id
        )

    assert "Feedback recorded" in message, "a failed reinforcement broke the feedback contract"
    assert _snapshot(engine, target) == before, "lifecycle advanced despite the failure"

    # The evidence itself is committed — this is about lifecycle, not observability.
    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ? AND node_id = ?",
        (log_id, target),
    ).fetchone()
    assert affinity is not None, "the feedback evidence was rolled back"


def test_recall_node_claims_its_own_event(engine):
    """Contract 7a: one deliberate fetch reinforces once, even when the agent
    then submits feedback on the event recall_node handed it.

    recall_node calls _log_feedback_candidates, which creates a whisper_log row
    for the very node it just confirmed and returns its id — the formatter
    attaches it as _whisper_log_id, and the agent instructions tell the model to
    submit_feedback(+1) with that id when it draws on the memory. Without a claim
    taken by recall_node itself, that feedback finds the event unclaimed and
    reinforces a second time, so one fetch counts twice. Found by whole-branch
    review, reproduced before the fix as access_count 0 -> 1 -> 2.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]

    before = _snapshot(engine, target)
    engine.recall_node(target)
    after_recall = _snapshot(engine, target)
    assert after_recall != before, "recall_node did not confirm its own node"

    row = engine.db.conn.execute(
        "SELECT id FROM whisper_log WHERE node_id = ? ORDER BY id DESC LIMIT 1",
        (target,),
    ).fetchone()
    assert row is not None, "recall_node logged no feedback candidate for its own node"

    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=row["id"])

    assert _snapshot(engine, target) == after_recall, (
        "feedback on the event recall_node itself surfaced reinforced it a second time"
    )


def test_recall_node_does_not_reinforce_when_it_loses_the_claim(engine):
    """Contract 7b: recall_node reinforces only if it actually took the claim.

    _log_feedback_candidates commits the new whisper_log row in its own
    transaction, and only afterwards does recall_node open the claim
    transaction. In that gap the event is committed but unclaimed, so a
    concurrent submit_feedback using the supported no-whisper_log_id fallback
    resolves that very row — it is the newest one for the node — and claims it
    first. recall_node discards its own claim result and reinforces regardless,
    so one fetch counts twice and the at-most-once latch is violated by the
    caller that introduced it.

    The barrier is deterministic rather than timed: the competing feedback runs
    inside a wrapper around _log_feedback_candidates, which is exactly the
    committed-but-unclaimed window, with no transaction open and no lock held.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]

    before = _snapshot(engine, target)
    real_log_candidates = engine._log_feedback_candidates

    def claim_the_event_first(*args, **kwargs):
        logged = real_log_candidates(*args, **kwargs)
        engine.submit_feedback(target, signal=1, source="explicit")
        return logged

    with patch.object(engine, "_log_feedback_candidates", side_effect=claim_the_event_first):
        engine.recall_node(target)

    after = _snapshot(engine, target)
    assert after["file"][0] == before["file"][0] + 1, (
        "one whisper event reinforced twice: access_count "
        f"{before['file'][0]} -> {after['file'][0]}"
    )
    assert after["db"][0] == after["file"][0], "file and DB disagree on access_count"


def test_recall_node_does_not_reinforce_without_an_event_to_claim(engine):
    """Contract 7c: no claim, no reinforcement — not even on the deliberate surface.

    _log_feedback_candidates swallows its own failures and returns {}, leaving
    recall_node with no whisper_log row to latch on. Reinforcing anyway would be
    the request-driven path this issue removes: the plan's constraint is that
    reinforcement fires on the claim, never on the request. Pins the deliberate
    side of the fix for contract 7b, which would otherwise look like an oversight
    and invite a `claimed or target_log_id is None` regression.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]

    before = _snapshot(engine, target)

    with patch.object(engine, "_log_feedback_candidates", return_value={}):
        engine.recall_node(target)

    assert _snapshot(engine, target) == before, (
        "recall_node reinforced with no event to claim — reinforcement followed the "
        "request instead of the claim"
    )


# --- Held-back relevance is not confirmed use (2026-08-16 council round) ----

def _seed_held_back_whisper_log(engine, node_id, prompt="what about caching?"):
    """Insert a historical event for a memory whisper held back.

    The automatic retrospective review producer is retired, but existing
    was_injected = 0 rows remain valid feedback provenance. _seed_whisper_log
    cannot be used here: it goes through recall_search, which writes
    was_injected = 1.

    logged_at is a Python ISO timestamp, not SQLite's datetime('now'), because
    _log_feedback_candidates writes ISO and the fallback orders by this column
    as TEXT. The two formats differ at index 10 — 'T' (0x54) against ' '
    (0x20) — so a datetime('now') row sorts BEFORE an ISO row written in the
    same second, and the fallback would silently resolve the wrong event.
    """
    from datetime import datetime, timezone

    cursor = engine.db.conn.execute(
        "INSERT INTO whisper_log "
        "(session_id, space, prompt_hash, prompt_text, prompt_vec, node_id, "
        "score, decision_stage, was_injected, logged_at) "
        "VALUES ('sess-review', 'myspace', 'hash-review', ?, X'', ?, 0.31, "
        "'injection_gate', 0, ?)",
        (prompt, node_id, datetime.now(timezone.utc).isoformat()),
    )
    engine.db.conn.commit()
    return cursor.lastrowid


def test_held_back_relevance_feedback_does_not_confirm_use(engine):
    """Contract 11: judging a held-back memory relevant is not using it.

    Historical feedback may still be attached to a was_injected = 0 event, but
    the memory was never shown to the agent. Treating the judgment as use would
    fabricate retention, which issue #220 forbids.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    held_back_id = _seed_held_back_whisper_log(engine, target)

    before = _snapshot(engine, target)

    engine.submit_feedback(target, signal=1, source="implicit", whisper_log_id=held_back_id)

    assert _snapshot(engine, target) == before, (
        "relevance feedback on a memory that was never surfaced reinforced it"
    )
    claims = engine.db.conn.execute(
        "SELECT COUNT(*) FROM confirmed_use_claims WHERE whisper_log_id = ?",
        (held_back_id,),
    ).fetchone()[0]
    assert claims == 0, "a held-back event took a confirmed-use claim"

    # The judgement itself is still evidence — only the lifecycle is off limits.
    affinity = engine.db.conn.execute(
        "SELECT * FROM affinity WHERE whisper_log_id = ? AND node_id = ?",
        (held_back_id, target),
    ).fetchone()
    assert affinity is not None, "review feedback stopped recording affinity"


def test_legacy_fallback_on_a_held_back_event_does_not_confirm(engine):
    """Contract 11a: the fallback's accepted loss, pinned deliberately.

    submit_feedback without whisper_log_id resolves to the node's newest
    whisper row, injected or not. When that row is a held-back review
    candidate, no claim is taken even though an older injected event exists —
    a legitimate reinforcement is lost in silence. Accepted: failing closed is
    the right side to err on under the at-most-once contract, and the fallback
    already documents itself as not exact. Fixing the fallback's selection
    would also move which event affinity and signals attach to, which is a
    different defect. This test exists so that loss stays a decision rather
    than becoming a surprise.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    injected_id = _seed_whisper_log(engine, target)
    held_back_id = _seed_held_back_whisper_log(engine, target)
    assert held_back_id > injected_id, "the held-back event must be the newer row"

    before = _snapshot(engine, target)

    engine.submit_feedback(target, signal=1, source="implicit")

    assert _snapshot(engine, target) == before, (
        "the legacy fallback reinforced through a held-back event"
    )
    # The fallback still attaches its evidence to the newest event — unchanged.
    affinity = engine.db.conn.execute(
        "SELECT whisper_log_id FROM affinity WHERE node_id = ?", (target,)
    ).fetchone()
    assert affinity["whisper_log_id"] == held_back_id


# --- Reinforcement must survive its own hazards (2026-08-16 council R1) -----

def test_confirmed_use_reinforces_a_node_whose_stability_is_zero(engine):
    """Contract 12: stability = 0 must not silently swallow the reinforcement.

    Node.stability is Field(ge=0.0), so 0 is a valid persisted value, and
    _record_confirmed_use divides by it (retrievability = exp(-days / stability)).
    The resulting ZeroDivisionError is caught by submit_feedback's isolating
    except, which by design never propagates — so the caller is told "Feedback
    recorded" while the lifecycle stays frozen. The claim is already committed,
    so the retry hits ON CONFLICT and the reinforcement is lost for good.

    decay_manager.py:50 and importance_scorer.py:80 already guard this exact
    division; _record_confirmed_use is the one consumer that does not.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]

    node = engine.file_store.load(target)
    node.stability = 0.0
    engine.file_store.save(node)
    engine.db.conn.execute("UPDATE nodes SET stability = 0.0 WHERE id = ?", (target,))
    engine.db.conn.commit()

    injected_id = _seed_whisper_log(engine, target)
    before = _snapshot(engine, target)
    assert before["file"][2] == 0.0, "the fixture failed to persist stability = 0"

    engine.submit_feedback(target, signal=1, source="implicit", whisper_log_id=injected_id)

    after = _snapshot(engine, target)
    assert after["file"][0] == before["file"][0] + 1, (
        "a zero-stability node took the claim but was never reinforced"
    )
    assert after["db"][0] == before["db"][0] + 1, "file advanced but the DB row did not"
    assert after["file"][2] > 0.0, "stability stayed at zero — the node can never recover"


def test_recall_node_returns_the_node_when_reinforcement_fails(engine):
    """Contract 13: a reinforcement failure must not cost the agent its answer.

    submit_feedback (2604-2610) and the session watcher (611-615) both isolate
    _record_confirmed_use behind try/except: the claim is already committed and
    the evidence durably recorded, so a mutator failure is a logged miss, never
    the caller's problem. recall_node called it bare, so the same failure threw
    the fetch away — the agent gets nothing, the event stays claimed, and the
    retry logs a second event that can never confirm the first.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]

    with patch.object(engine, "_record_confirmed_use", side_effect=RuntimeError("disk gone")):
        formatted = engine.recall_node(target)

    assert formatted, "recall_node propagated a reinforcement failure instead of the node"
    assert "Caching 0" in formatted, "recall_node returned something other than the node"

    # The claim is still taken: at-most-once holds, the miss is only the mutator's.
    claims = engine.db.conn.execute(
        "SELECT COUNT(*) FROM confirmed_use_claims WHERE node_id = ?", (target,)
    ).fetchone()[0]
    assert claims == 1, "the claim was rolled back — at-most-once no longer holds"


def test_recall_search_structured_rejects_positional_tuning_args(engine):
    """Contract 14: tuning parameters are keyword-only, so a stale positional
    call cannot silently redefine itself.

    #220 removed the `touch_access` parameter, which held the 4th positional
    slot. `min_relevance` inherited that slot, so a pre-existing positional
    call passing False in position 4 would mean min_relevance=0 — silently
    dropping the deliberate-recall relevance floor and admitting results below
    it. The bare `*` turns that silent redefinition into an immediate TypeError.
    """
    _make_nodes(engine, count=1)

    with pytest.raises(TypeError) as excinfo:
        # The exact shape of a stale caller: `False` where touch_access used to be.
        engine.recall_search_structured("caching architecture", 10, None, False)

    assert "positional" in str(excinfo.value), (
        f"raised for the wrong reason: {excinfo.value}"
    )

    # The supported shapes must keep working — this is the other half of the
    # contract. `isinstance(..., list)` rather than `is not None`: the point is
    # that the call completes and still returns the documented type.
    assert isinstance(engine.recall_search_structured("caching architecture"), list)
    assert isinstance(engine.recall_search_structured("caching architecture", limit=4), list)
    assert isinstance(engine.recall_search_structured("caching architecture", 4, None), list)
    assert isinstance(engine.recall_search_structured(
        "caching architecture", limit=4, min_relevance=0.0, spread_activation=False,
    ), list)
