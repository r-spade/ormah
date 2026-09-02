"""Contract tests for issue #220: surfacing must not be confirmed use.

Every assertion reads the four lifecycle fields from BOTH the markdown file and
the SQLite row. A test that checked only the database would pass while the file
rotted, and vice versa.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah import lifecycle
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


def _snapshot_stores_agree(snapshot):
    """True if a _snapshot's `file` and `db` halves carry the same VALUES.

    `_snapshot`'s `file` tuple carries parsed datetime objects (MemoryNode's
    last_accessed/last_review are typed datetime); its `db` tuple carries the
    raw TEXT sqlite3 returns for the same columns. A plain `file == db` tuple
    comparison therefore never matches, even when the two stores fully agree —
    proven by executing it: both sides print the identical instant, one as a
    datetime, one as its isoformat string. Only used where the two are compared
    directly against each other; every other _snapshot comparison in this file
    is same-shape-vs-same-shape and is unaffected.
    """
    file_vals, db_vals = snapshot["file"], snapshot["db"]

    def _dt(value):
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    return all(
        _dt(f) == _dt(d)
        for f, d in zip(file_vals, db_vals, strict=True)
    )


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

    # Issue #272: each thread needs its OWN claimed event — the mutator is now
    # at-most-once per (whisper_log_id, node_id), so 8 threads sharing one claim
    # would only reinforce once. Claimed up front, sequentially, since claiming
    # must happen inside its own transaction before the mutator runs.
    log_ids = [_claim_fresh_event(engine, target) for _ in range(8)]
    threads = [
        threading.Thread(
            target=engine._record_confirmed_use,
            kwargs={"node_id": target, "whisper_log_id": log_id},
        )
        for log_id in log_ids
    ]
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


def _claim_fresh_event(engine, node_id):
    """Insert a whisper_log row for node_id and take its confirmed-use claim.

    Issue #272 made whisper_log_id a required keyword on _record_confirmed_use:
    every real caller reaches it only after _claim_confirmed_use has taken a claim
    inside its own transaction, so a direct call needs the same shape. Bypasses
    recall_search (unlike _seed_whisper_log above) so cooldown/stamping tests that
    call the mutator many times on one node get one fresh, independently-claimable
    event per call rather than depending on search surfacing the same node again.
    """
    with engine.db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO whisper_log "
            "(session_id, prompt_hash, prompt_vec, node_id, score, was_injected, logged_at) "
            "VALUES ('test-direct-claim', ?, X'00', ?, 1.0, 1, datetime('now'))",
            (uuid.uuid4().hex, node_id),
        )
        whisper_log_id = cursor.lastrowid
        engine._claim_confirmed_use(
            conn, whisper_log_id, node_id, signal=1, source="explicit", strength=1.0,
        )
        # _claim_confirmed_use's INSERT stamps claimed_at with SQL datetime('now'),
        # which truncates to whole seconds. The mutator's clock is claimed_at, so a
        # test that calls _reinforce several times faster than 1 second apart (the
        # cooldown tests do) would see identical or barely-advancing timestamps —
        # and near an exact cooldown-day boundary, that truncation alone can flip
        # `reinforcement_due`. Overwritten here with a microsecond-precision Python
        # timestamp, the same precision _age_for_fsrs already relies on for the
        # same reason.
        conn.execute(
            "UPDATE confirmed_use_claims SET claimed_at = ? "
            "WHERE whisper_log_id = ? AND node_id = ?",
            (datetime.now(timezone.utc).isoformat(), whisper_log_id, node_id),
        )
    return whisper_log_id


def _reinforce(engine, node_id):
    """Claim a fresh event for node_id and reinforce it in one step.

    The direct replacement for the pre-#272 `engine._record_confirmed_use(node_id)`
    call shape, used by tests that exercise the mutator's lifecycle arithmetic
    (cooldown, stamping) rather than the claiming path itself.
    """
    engine._record_confirmed_use(
        node_id, whisper_log_id=_claim_fresh_event(engine, node_id)
    )


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
    """Contract 9: submit_feedback(auto_heuristic) is below the #272 evidence floor.

    Not an exclusion by source any more — auto_heuristic IS in the allowlist since
    #272. feedback_strength maps it to UNKNOWN (0.40), under HEURISTIC_CONFIRM_FLOOR,
    because a submit_feedback call carries no evidence of a verbatim match.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    before = _snapshot(engine, target)
    engine.submit_feedback(target, signal=1, source="auto_heuristic", whisper_log_id=log_id)

    assert _snapshot(engine, target) == before, "auto_heuristic must not confirm use"


@pytest.mark.parametrize("strength,should_confirm", [
    (0.80, True),    # exactly the floor — inclusive
    (0.7999, False), # just below
    (0.98, True),    # node_id
    (0.40, False),   # token_overlap floor
])
def test_heuristic_claim_respects_the_evidence_floor(engine, strength, should_confirm):
    """#272 D1/D2: the floor lives in the claim helper, not in its callers."""
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    with engine.db.transaction() as conn:
        claimed = engine._claim_confirmed_use(
            conn, log_id, target,
            signal=1, source="auto_heuristic", strength=strength,
        )

    assert claimed is should_confirm


def test_the_floor_does_not_gate_the_other_sources(engine):
    """#272 D2: the floor is scoped to auto_heuristic only.

    explicit is 1.00 on the ladder, so a low strength reaching this helper would
    mean the caller computed it wrong — but gating it here would silently drop a
    real confirmation instead of surfacing that bug.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)

    with engine.db.transaction() as conn:
        claimed = engine._claim_confirmed_use(
            conn, log_id, target, signal=1, source="explicit", strength=0.0,
        )

    assert claimed is True, "the floor must not apply to non-heuristic sources"


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


# --- Review relevance is not confirmed use (2026-08-16 council round) -------

def _seed_held_back_whisper_log(engine, node_id, prompt="what about caching?"):
    """Insert the kind of event the session-start review hands to the agent.

    _find_review_candidate selects rows with was_injected = 0 — memories Ormah
    held back and never surfaced — and _REVIEW_FRAMING hands that id to the
    agent asking for source="implicit" feedback. _seed_whisper_log cannot be
    used here: it goes through recall_search, which writes was_injected = 1.

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


def test_review_relevance_feedback_does_not_confirm_use(engine):
    """Contract 11: judging a held-back memory relevant is not using it.

    The review path deliberately surfaces an event with was_injected = 0 and
    asks "would this have been useful?" — a relevance adjudication, not a use.
    _claim_confirmed_use allowlists "implicit" and checks no provenance, so the
    claim is taken and the lifecycle advances on a memory the agent never saw.
    That is fabricated retention entering through the review door, which is what
    issue #220 exists to close.
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


# --- Task 4: backfill the rows the defect already wrote (#272) -------------

def _seed_heuristic_signal(
    engine, node_id, whisper_log_id, strength, match="node_id", overlap_ratio=None
):
    """Write a positive heuristic signal row as the pre-#272 code would have.

    ``strength`` only fills the cosmetic ``signals.strength`` column, which the
    backfill deliberately ignores — it recomputes from ``evidence`` instead
    (issue #272). ``overlap_ratio``, when given, is written into ``evidence``
    and is what actually drives that recompute for a token_overlap row via
    ``strength_from_evidence``. Defaults to omitted so existing callers keep
    their current evidence shape unchanged.
    """
    evidence = {"match": match}
    if overlap_ratio is not None:
        evidence["overlap_ratio"] = overlap_ratio
    engine.db.conn.execute(
        """
        INSERT INTO signals
            (whisper_log_id, node_id, signal_type, polarity, strength, source,
             session_id, surface, space, prompt_hash, evidence, created)
        VALUES (?, ?, 'whisper_referenced', 1, ?, 'transcript_watcher_heuristic',
                's1', 'transcript', 'myproject', 'h', ?, datetime('now'))
        """,
        (whisper_log_id, node_id, strength, json.dumps(evidence)),
    )
    engine.db.conn.commit()


def test_backfill_claims_and_reinforces_historical_verbatim_rows(engine):
    """#272 D4: the rows the defect already wrote are repaired at boot."""
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)

    before = _snapshot(engine, target)
    engine._migrate_heuristic_confirmed_use()

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ? AND node_id = ?",
        (log_id, target),
    ).fetchone()
    assert claim is not None, "the backfill claimed nothing"
    assert _snapshot(engine, target) != before, "the backfill claimed but did not reinforce"


def test_backfill_is_idempotent(engine):
    """#272 D4: a second boot must not reinforce the same event again."""
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)

    engine._migrate_heuristic_confirmed_use()
    after_first = _snapshot(engine, target)

    engine._migrate_heuristic_confirmed_use()

    assert _snapshot(engine, target) == after_first, "the backfill reinforced twice"


@pytest.mark.parametrize("strength,match,overlap_ratio", [
    (0.40, "token_overlap", 0.5),      # ratio <= OVERLAP_GATE recomputes to the floor itself
    (0.7799, "token_overlap", 8.7428), # asymptotic near-supremum, still strictly under 0.80
])
def test_backfill_skips_rows_below_the_floor(engine, strength, match, overlap_ratio):
    """#272 D4: the backfill uses the same floor as the live path.

    The seeded ``strength`` only fills the cosmetic ``signals.strength`` column,
    which the backfill ignores by design — it recomputes from ``evidence`` via
    ``strength_from_evidence``. ``overlap_ratio`` is what actually drives that
    recompute, so the two cases genuinely land at different points on the
    token_overlap band (0.40 and ~0.7799) instead of both silently falling
    back to the same OVERLAP_FLOOR default.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)
    _seed_heuristic_signal(
        engine, target, log_id, strength=strength, match=match, overlap_ratio=overlap_ratio
    )

    before = _snapshot(engine, target)
    engine._migrate_heuristic_confirmed_use()

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (log_id,)
    ).fetchone()
    assert claim is None, "a below-floor row was backfilled"
    assert _snapshot(engine, target) == before


def test_backfill_skips_a_never_injected_event(engine):
    """#272 D4: was_injected = 1 is the provenance test the claim helper enforces.

    A memory the agent never saw cannot have been used, however the signal reads.
    """
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)
    engine.db.conn.execute("UPDATE whisper_log SET was_injected = 0 WHERE id = ?", (log_id,))
    engine.db.conn.commit()
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)

    before = _snapshot(engine, target)
    engine._migrate_heuristic_confirmed_use()

    assert _snapshot(engine, target) == before, "a non-injected event was backfilled"


def test_backfill_skips_an_already_claimed_event(engine):
    """#272 D4: an event confirmed through another caller must not reinforce twice."""
    ids = _make_nodes(engine, count=1)
    target = ids[0]
    log_id = _seed_whisper_log(engine, target)
    engine.submit_feedback(target, signal=1, source="explicit", whisper_log_id=log_id)
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)

    after_feedback = _snapshot(engine, target)
    engine._migrate_heuristic_confirmed_use()

    assert _snapshot(engine, target) == after_feedback, "an already-claimed event reinforced again"


def test_backfill_cutoff_advances_to_the_highest_processed_id(engine):
    """#272 D4: the cutoff advances by processed id, never to MAX(id).

    Same defence-in-depth _migrate_signal_strength documents: a row committed by
    another writer between the SELECT and the stamp must not be skipped forever.
    A below-floor row is still 'processed' — it was examined and rejected.
    """
    strong, weak = _make_nodes(engine, count=2)
    strong_log = _seed_whisper_log(engine, strong, prompt="caching strong")
    _seed_heuristic_signal(engine, strong, strong_log, strength=0.98)
    # Seeded LAST and below the floor: the cutoff must still clear it, or the scan
    # window would grow forever on a store whose newest rows are all token_overlap.
    weak_log = _seed_whisper_log(engine, weak, prompt="caching weak")
    _seed_heuristic_signal(engine, weak, weak_log, strength=0.40, match="token_overlap")

    highest = engine.db.conn.execute(
        "SELECT MAX(id) AS m FROM signals WHERE source = 'transcript_watcher_heuristic'"
    ).fetchone()["m"]

    engine._migrate_heuristic_confirmed_use()

    cutoff = engine._meta_int("heuristic_confirmed_use_cutoff")
    assert cutoff == highest, (
        f"cutoff {cutoff} stopped short of the last processed id {highest} — "
        "a below-floor row was examined but not counted as processed"
    )
    assert engine._meta_int("heuristic_confirmed_use_version") == 1


def test_backfill_skips_a_signal_whose_node_is_not_the_events_node(engine):
    """#272, council R1 (Codex HIGH): the claim helper does not check event/node ownership.

    It inserts the node id it is handed after testing only was_injected. The live
    path reads both ids off one whisper_log row so they always agree; the backfill
    reads them from different tables, so a legacy or hand-repaired signal could
    reinforce a node the agent never saw for that event.
    """
    victim, other = _make_nodes(engine, count=2)
    log_id = _seed_whisper_log(engine, victim)
    # The event belongs to `victim`, but the signal names `other`.
    _seed_heuristic_signal(engine, other, log_id, strength=0.98)

    before_other = _snapshot(engine, other)
    engine._migrate_heuristic_confirmed_use()

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ? AND node_id = ?",
        (log_id, other),
    ).fetchone()
    assert claim is None, "a signal claimed an event that belonged to a different node"
    assert _snapshot(engine, other) == before_other


def test_backfill_cutoff_clears_every_kind_of_ineligible_tail(engine):
    """#272, council R1+R3 (Codex): no eligibility predicate may live in the WHERE.

    Three shapes, because the defect reappeared in three disguises across the review:
      - polarity 0            -> excluded by a WHERE predicate (round 1)
      - was_injected = 0      -> excluded by a WHERE predicate (round 1)
      - whisper_log_id NULL   -> excluded by an INNER JOIN (round 3). The column is
                                 nullable and ON DELETE SET NULL, so whisper_log_cleanup
                                 orphans rows routinely — this is the common case, not
                                 an exotic one.
    Any of them at the high-id tail must still advance the cutoff, or every boot
    rescans a growing tail forever.
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    not_injected_log = _seed_whisper_log(engine, target, prompt="caching not injected")
    engine.db.conn.execute(
        "UPDATE whisper_log SET was_injected = 0 WHERE id = ?", (not_injected_log,)
    )

    def _tail(whisper_log_id, polarity):
        engine.db.conn.execute(
            """
            INSERT INTO signals
                (whisper_log_id, node_id, signal_type, polarity, strength, source,
                 session_id, surface, space, prompt_hash, evidence, created)
            VALUES (?, ?, 'whisper_referenced', ?, 0.98, 'transcript_watcher_heuristic',
                    's1', 'transcript', 'myproject', 'h', ?, datetime('now'))
            """,
            (whisper_log_id, target, polarity, json.dumps({"match": "node_id"})),
        )

    _tail(log_id, 0)                # polarity 0
    _tail(not_injected_log, 1)      # was_injected = 0
    _tail(None, 1)                  # orphaned: no whisper_log parent at all
    engine.db.conn.commit()

    highest = engine.db.conn.execute(
        "SELECT MAX(id) AS m FROM signals WHERE source = 'transcript_watcher_heuristic'"
    ).fetchone()["m"]

    engine._migrate_heuristic_confirmed_use()

    assert engine._meta_int("heuristic_confirmed_use_cutoff") == highest, (
        "an ineligible trailing row pinned the cutoff — the scan window will grow forever"
    )
    assert engine.db.conn.execute(
        "SELECT COUNT(*) AS n FROM confirmed_use_claims"
    ).fetchone()["n"] == 0, "an ineligible row was claimed"


def test_backfill_runs_from_startup_and_ignores_stale_stored_strength(engine):
    """#272, council R1+R2 (Cursor) + the final-plan run: call site AND recompute.

    This test USED to pin the migration order, back when eligibility read
    `signals.strength`. It no longer can: eligibility is recomputed from `evidence`,
    so both assertions hold whichever order the two migrations run in. That is the
    point — the order stopped being load-bearing, and a test claiming to prove an
    order it can no longer falsify would be worse than no test.

    What it pins now, both of which are real:
      - `startup()` actually calls the backfill (swap the call out and the verbatim
        assertion goes red);
      - the stored column does not decide anything. Both seeds carry a strength that
        contradicts their evidence, and the outcome follows the evidence:
          * token_overlap stored at a stale 1.0 (above the floor) -> recomputes to
            ~0.55 -> must NOT claim;
          * node_id stored at a stale 0.50 (below the floor) -> recomputes to 0.98
            -> must claim.
        An implementation that reads `row["strength"]` turns BOTH red.

    The inter-transaction window itself is covered by
    test_backfill_ignores_a_stale_row_written_after_the_ladder_committed, which a
    sequential startup() test cannot express.
    """
    overlap_node, verbatim_node = _make_nodes(engine, count=2)
    overlap_log = _seed_whisper_log(engine, overlap_node, prompt="caching overlap")
    verbatim_log = _seed_whisper_log(engine, verbatim_node, prompt="caching verbatim")

    _seed_heuristic_signal(
        engine, overlap_node, overlap_log, strength=1.0, match="token_overlap",
    )
    engine.db.conn.execute(
        "UPDATE signals SET evidence = ? WHERE whisper_log_id = ?",
        (json.dumps({"match": "token_overlap", "overlap_ratio": 1.0}), overlap_log),
    )
    _seed_heuristic_signal(engine, verbatim_node, verbatim_log, strength=0.50)

    # Force both migrations to re-run on the next startup.
    engine.db.conn.execute(
        "DELETE FROM meta WHERE key IN "
        "('heuristic_confirmed_use_version', 'heuristic_confirmed_use_cutoff', "
        "'signal_strength_ladder_version', 'signal_strength_ladder_cutoff')"
    )
    engine.db.conn.commit()

    engine.startup()

    verbatim_claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (verbatim_log,)
    ).fetchone()
    assert verbatim_claim is not None, (
        "startup() never ran the backfill, or eligibility read the stale stored 0.50 "
        "instead of recomputing 0.98 from evidence.match = node_id"
    )

    overlap_claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (overlap_log,)
    ).fetchone()
    assert overlap_claim is None, (
        "a stale DEFAULT-1.0 token_overlap row confirmed — eligibility trusted the stored "
        "column instead of recomputing from evidence"
    )


def test_backfill_ignores_a_stale_row_written_after_the_ladder_committed(engine):
    """#272, final-plan council (Codex HIGH + Cursor MEDIUM, converging independently).

    The falsifier for the inter-transaction window. `_migrate_signal_strength` and this
    backfill commit SEPARATELY, so an old binary — the second unmanaged process of #238
    — can write a pre-ladder row carrying the schema default of 1.0 *after* the ladder
    has committed and *before* this SELECT begins. Ordering the two calls cannot close
    that window, and a sequential startup() test cannot expose it.

    Simulated exactly: run the ladder to completion, THEN insert the stale row, THEN run
    only the backfill. An implementation that reads `signals.strength` claims it, and the
    claim is a monotonic latch with a markdown write already on disk — no undo.
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target, prompt="caching stale window")

    # The ladder runs first and commits — exactly as startup() orders it.
    engine._migrate_signal_strength()

    # The window: an old binary commits a token_overlap row at the schema default of
    # 1.0, which the ladder has already finished and will not revisit this boot.
    _seed_heuristic_signal(
        engine, target, log_id, strength=1.0, match="token_overlap",
    )
    engine.db.conn.execute(
        "UPDATE signals SET evidence = ? WHERE whisper_log_id = ?",
        (json.dumps({"match": "token_overlap", "overlap_ratio": 1.0}), log_id),
    )
    engine.db.conn.commit()

    before = _snapshot(engine, target)
    engine._migrate_heuristic_confirmed_use()

    claim = engine.db.conn.execute(
        "SELECT 1 FROM confirmed_use_claims WHERE whisper_log_id = ?", (log_id,)
    ).fetchone()
    assert claim is None, (
        "a stale DEFAULT-1.0 row written in the window between the two migrations took "
        "an irreversible claim — eligibility must recompute from evidence, not read the column"
    )
    assert _snapshot(engine, target) == before

    # The cutoff still advanced past it: ineligible is not unprocessed.
    assert engine._meta_int("heuristic_confirmed_use_cutoff") == engine.db.conn.execute(
        "SELECT MAX(id) AS m FROM signals WHERE source = 'transcript_watcher_heuristic'"
    ).fetchone()["m"]


def test_backfill_isolates_one_nodes_failure(engine):
    """#272 D4: one unreadable node must not cost every later node its repair."""
    first, second = _make_nodes(engine, count=2)
    for node_id in (first, second):
        log_id = _seed_whisper_log(engine, node_id, prompt=f"caching {node_id}")
        _seed_heuristic_signal(engine, node_id, log_id, strength=0.98)

    before_second = _snapshot(engine, second)
    real = engine._record_confirmed_use

    def flaky(node_id, *, whisper_log_id):
        if node_id == first:
            raise ZeroDivisionError("simulated mutator failure")
        return real(node_id, whisper_log_id=whisper_log_id)

    with patch.object(engine, "_record_confirmed_use", side_effect=flaky):
        engine._migrate_heuristic_confirmed_use()

    assert _snapshot(engine, second) != before_second, "node 2 lost its backfill"


def test_backfill_does_not_advance_last_accessed_to_boot_time(engine):
    """Council #272 finding 1: a historical use must not be recorded as use now.

    last_accessed sits BETWEEN the signal's logged_at and boot time, so with the
    truthful clock max(claimed_at, last_accessed) keeps it; with the buggy boot
    clock the claim wins and drags it to now — which is exactly the RED.
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)
    event_time = datetime.now(timezone.utc) - timedelta(days=10)
    anchor_time = datetime.now(timezone.utc) - timedelta(days=2)
    engine.db.conn.execute(
        "UPDATE whisper_log SET logged_at = ? WHERE id = ?",
        (event_time.isoformat(), log_id),
    )
    engine.db.conn.execute(
        "UPDATE nodes SET last_accessed = ? WHERE id = ?",
        (anchor_time.isoformat(), target),
    )
    engine.db.conn.commit()

    engine._migrate_heuristic_confirmed_use()

    row = engine.db.conn.execute(
        "SELECT last_accessed FROM nodes WHERE id = ?", (target,)
    ).fetchone()
    assert datetime.fromisoformat(row["last_accessed"]) == anchor_time, (
        "backfilling a 10-day-old signal moved last_accessed to boot time — "
        "the claim must carry the event's clock, not the wall clock"
    )


def test_backfill_claim_carries_the_events_time_normalized(engine):
    """Council #272 finding 1: claimed_at = logged_at, in datetime('now') shape.

    The space-format assertion is the sweeper's contract: reinforcement_retry
    compares claimed_at lexicographically against datetime('now', ...), where
    'T' (0x54) sorts above ' ' (0x20).
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)
    engine.db.conn.execute(
        "UPDATE whisper_log SET logged_at = '2026-08-15T12:00:00.123456+00:00' "
        "WHERE id = ?",
        (log_id,),
    )
    engine.db.conn.commit()

    engine._migrate_heuristic_confirmed_use()

    row = _claim_row(engine, log_id, target)
    assert row is not None, "the backfill claimed nothing"
    assert row["claimed_at"] == "2026-08-15 12:00:00", (
        f"claimed_at is {row['claimed_at']!r}: the backfill must stamp the "
        "event's logged_at, normalized to SQLite's space-format UTC"
    )


def test_backfill_survives_a_malformed_logged_at(engine):
    """Guard, not RED: a malformed logged_at falls back to the boot clock.

    datetime('not-a-timestamp') is NULL and claimed_at is NOT NULL — without the
    COALESCE the INSERT raises IntegrityError and the whole backfill transaction
    dies at boot. Falling back keeps today's behavior for that one row instead of
    losing the signal forever (the cutoff advances regardless).
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _seed_heuristic_signal(engine, target, log_id, strength=0.98)
    engine.db.conn.execute(
        "UPDATE whisper_log SET logged_at = 'not-a-timestamp' WHERE id = ?",
        (log_id,),
    )
    engine.db.conn.commit()

    engine._migrate_heuristic_confirmed_use()

    row = _claim_row(engine, log_id, target)
    assert row is not None, "a malformed logged_at must not cost the claim"
    assert "T" not in row["claimed_at"]
    stamped = datetime.fromisoformat(row["claimed_at"]).replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 60


def test_live_claim_still_stamps_now(engine):
    """Guard, not RED: the historical clock is backfill-only (decision 2026-08-28).

    A live claim's skew from its event is minutes; its sources' semantics
    (explicit/implicit/judge) are out of this fix's scope.
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    engine.db.conn.execute(
        "UPDATE whisper_log SET logged_at = ? WHERE id = ?",
        ((datetime.now(timezone.utc) - timedelta(days=10)).isoformat(), log_id),
    )
    engine.db.conn.commit()

    _take_claim(engine, log_id, target)

    row = _claim_row(engine, log_id, target)
    assert "T" not in row["claimed_at"]
    stamped = datetime.fromisoformat(row["claimed_at"]).replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - stamped).total_seconds()) < 60, (
        "a live claim must keep the wall clock even when its event is old"
    )


# --- Task 5: durable reinforcement (#220 debt) ------------------------------


def _claim_row(engine, log_id, node_id):
    return engine.db.conn.execute(
        "SELECT claimed_at, state, reinforced_at FROM confirmed_use_claims "
        "WHERE whisper_log_id = ? AND node_id = ?",
        (log_id, node_id),
    ).fetchone()


def _take_claim(engine, log_id, node_id):
    with engine.db.transaction() as conn:
        engine._claim_confirmed_use(
            conn, log_id, node_id, signal=1, source="explicit", strength=1.0,
        )


def _age_for_fsrs(engine, log_id, node_id, node_days=3, claim_days=1):
    """Push the node's last_accessed and the claim's claimed_at into the past.

    Without this the FSRS assertions are vacuous. _make_nodes seeds last_accessed
    from datetime.now(), and a claim taken milliseconds later gives days_since ~= 0
    — so `now = claimed_at` and `now = datetime.now()` produce the SAME stability to
    any tolerance, and the test would pass with the bug fully present.

    Two different ages are required, not one. days_since is `now - last_accessed`
    clamped at 0, so ageing only the claim would leave both variants clamped to 0
    and still indistinguishable. With last_accessed 3 days back and claimed_at 1 day
    back, the claimed_at clock yields days_since = 1 and the wall clock yields
    days_since = 3 — a gap the assertions can see.

    Both ages are kept UNDER the spacing-factor's saturation point, not just apart
    from each other. spacing_factor caps at fsrs_spacing_cap (2.0) once
    0.2 * days_since / stability reaches log(2) ~= 3.47 days at the default
    stability = 1.0 seeded by _make_nodes — verified by executing
    reinforced_stability(1.0, days, ...): 10 and 20 days both round to the SAME
    2.0, indistinguishable regardless of which clock produced them, while 1 and 3
    days round to 1.61 and 1.91. Widening the gap without checking the cap would
    silently recreate the same vacuous test this helper exists to prevent.

    The nodes row is the source the mutator reads, so ageing the row is enough; the
    markdown is overwritten from it on the next write.
    """
    node_at = datetime.now(timezone.utc) - timedelta(days=node_days)
    claim_at = datetime.now(timezone.utc) - timedelta(days=claim_days)
    with engine.db.transaction() as conn:
        conn.execute(
            "UPDATE nodes SET last_accessed = ? WHERE id = ?",
            (node_at.isoformat(), node_id),
        )
        conn.execute(
            "UPDATE confirmed_use_claims SET claimed_at = ? "
            "WHERE whisper_log_id = ? AND node_id = ?",
            (claim_at.strftime("%Y-%m-%d %H:%M:%S"), log_id, node_id),
        )


def _as_utc(claimed_at):
    """The mutator's clock: claimed_at, as SQLite wrote it, made tz-aware.

    datetime('now') emits 'YYYY-MM-DD HH:MM:SS' in UTC with no offset, so the
    conversion is a fromisoformat plus an explicit tzinfo — never an astimezone,
    which would treat the naive value as local time and shift it.
    """
    return datetime.fromisoformat(claimed_at).replace(tzinfo=timezone.utc)


def _expected_reinforced_stability(engine, base_row, claimed_at):
    """ONE application of the growth formula, mirroring the mutator step for step.

    Recomputed rather than hardcoded: a literal would still pass if the mutator
    stopped calling reinforced_stability at all. It mirrors the mutator exactly —
    same gate, same anchor expression, same clock — so if the two ever diverge the
    test fails instead of quietly asserting a different formula.
    """
    now = _as_utc(claimed_at)
    last_review = (
        datetime.fromisoformat(base_row["last_review"])
        if base_row["last_review"]
        else None
    )
    last_accessed = (
        datetime.fromisoformat(base_row["last_accessed"])
        if base_row["last_accessed"]
        else None
    )
    if not lifecycle.reinforcement_due(
        last_review, now, engine.settings.fsrs_reinforcement_cooldown_days
    ):
        return base_row["stability"]

    anchor = last_accessed or last_review
    days_since = max((now - anchor).total_seconds() / 86400, 0.0)
    return lifecycle.reinforced_stability(
        base_row["stability"],
        days_since,
        growth_factor=engine.settings.fsrs_growth_factor,
        growth_exponent=engine.settings.fsrs_growth_exponent,
        spacing_cap=engine.settings.fsrs_spacing_cap,
        max_stability=engine.settings.fsrs_max_stability,
        initial_stability=engine.settings.fsrs_initial_stability,
    )


def test_mutator_failure_leaves_no_residue(engine, monkeypatch):
    """#272 D5-1: a failed save rolls back the claim state AND the nodes row."""
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _take_claim(engine, log_id, target)

    before = _snapshot(engine, target)

    def boom(node):
        raise OSError("disk full")

    monkeypatch.setattr(engine.file_store, "save", boom)
    with pytest.raises(OSError):
        engine._record_confirmed_use(target, whisper_log_id=log_id)

    assert _snapshot(engine, target) == before, "the failed mutator left a partial write"
    assert _claim_row(engine, log_id, target)["state"] == "pending", (
        "the claim left 'pending' even though nothing was applied"
    )


def test_failed_commit_does_not_inflate_the_counter(engine, monkeypatch):
    """#272 D5-2: the convergence claim, tested at the one place it can break.

    os.replace cannot be rolled back and COMMIT runs after the transaction body, so a
    failing COMMIT leaves the markdown one step ahead of the nodes row. Because the new
    values are computed FROM the nodes row and the clock comes FROM claimed_at, the
    retry recomputes the same target and overwrites the phantom — it must not add a
    second increment, nor a second stability step, on top of it.

    The failure is injected through Database.transaction, NOT through conn.execute.
    Council round 1 (Codex, MEDIUM, 1.0) proved by execution that patching the
    connection raises AttributeError: 'sqlite3.Connection' object attribute 'execute'
    is read-only, so the earlier draft of this test could never have run. Raising after
    the with-body returns is behaviourally identical to a failing COMMIT: file_store.save
    has already run and os.replace is irreversible, and the real transaction's
    except-clause issues the ROLLBACK that discards the row and the claim.
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _take_claim(engine, log_id, target)
    # Required, not decorative: without it claimed_at and the wall clock are
    # milliseconds apart and every FSRS assertion below passes with the bug present.
    _age_for_fsrs(engine, log_id, target)

    base_row = engine.db.conn.execute(
        "SELECT access_count, stability, last_accessed, last_review FROM nodes "
        "WHERE id = ?",
        (target,),
    ).fetchone()
    baseline, baseline_stability = base_row["access_count"], base_row["stability"]
    # The whole row is kept, not just two fields: _expected_reinforced_stability
    # mirrors the mutator, which reads last_accessed and last_review from it too.
    # Asserting against a value derived from what _make_nodes actually seeded, rather
    # than from an assumption about it, is what keeps this from passing for the wrong
    # reason if the fixture changes.

    real_transaction = type(engine.db).transaction

    @contextmanager
    def commit_fails(self):
        with real_transaction(self) as conn:
            yield conn
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(type(engine.db), "transaction", commit_fails)
    with pytest.raises(sqlite3.OperationalError):
        engine._record_confirmed_use(target, whisper_log_id=log_id)
    monkeypatch.undo()

    # The markdown ran ahead; the row and the claim did not.
    phantom = engine.file_store.load(target)
    assert phantom.access_count == baseline + 1
    assert engine.db.conn.execute(
        "SELECT access_count FROM nodes WHERE id = ?", (target,)
    ).fetchone()["access_count"] == baseline
    assert _claim_row(engine, log_id, target)["state"] == "pending"

    engine._record_confirmed_use(target, whisper_log_id=log_id)

    after = _snapshot(engine, target)
    assert _snapshot_stores_agree(after), "the stores did not converge"
    final = engine.db.conn.execute(
        "SELECT access_count, stability, last_accessed FROM nodes WHERE id = ?",
        (target,),
    ).fetchone()
    assert final["access_count"] == baseline + 1, (
        "one event produced more than one increment"
    )

    # The FSRS half of convergence: the retry must land on the SAME stability the
    # phantom did, which is only true because the clock came from claimed_at. With
    # datetime.now() the two attempts feed different days_since to
    # reinforced_stability and this equality fails — that is the bug this pins.
    assert final["stability"] == pytest.approx(phantom.stability), (
        "the retry recomputed a different FSRS target than the failed attempt"
    )
    assert final["stability"] != pytest.approx(baseline_stability), (
        "stability never moved, so the equality above proves nothing"
    )

    # And it is ONE application of the growth formula, not two compounded.
    claimed_at = _claim_row(engine, log_id, target)["claimed_at"]
    expected = _expected_reinforced_stability(engine, base_row, claimed_at)
    assert final["stability"] == pytest.approx(expected), (
        "stability compounded across the failed attempt and the retry"
    )

    # The assertion that actually kills the bug. With the wall clock the mutator would
    # feed days_since = (now - last_accessed) = 3 days instead of the claim's 1, so
    # this value must NOT be reachable. Computed, not hardcoded, so it tracks whatever
    # the settings say.
    wall_clock_target = lifecycle.reinforced_stability(
        baseline_stability,
        max(
            (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(base_row["last_accessed"])
            ).total_seconds()
            / 86400,
            0.0,
        ),
        growth_factor=engine.settings.fsrs_growth_factor,
        growth_exponent=engine.settings.fsrs_growth_exponent,
        spacing_cap=engine.settings.fsrs_spacing_cap,
        max_stability=engine.settings.fsrs_max_stability,
        initial_stability=engine.settings.fsrs_initial_stability,
    )
    assert wall_clock_target != pytest.approx(expected), (
        "the two clocks agree, so this test cannot tell them apart — widen the ages "
        "in _age_for_fsrs"
    )
    assert final["stability"] != pytest.approx(wall_clock_target), (
        "the mutator used datetime.now() instead of the claim's claimed_at"
    )

    assert final["last_accessed"] == _as_utc(claimed_at).isoformat(), (
        "last_accessed records the retry's clock instead of when the memory was used"
    )


def test_happy_path_agrees_across_claim_row_and_markdown(engine):
    """#272 D5-9: on success all three carry the same values."""
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _take_claim(engine, log_id, target)

    before = _snapshot(engine, target)
    engine._record_confirmed_use(target, whisper_log_id=log_id)
    after = _snapshot(engine, target)

    assert after != before, "nothing was reinforced"
    assert _snapshot_stores_agree(after), "markdown and database disagree"
    row = _claim_row(engine, log_id, target)
    assert row["state"] == "applied"
    assert row["reinforced_at"] is not None


def test_mutator_is_at_most_once_on_an_applied_claim(engine):
    """#272 D5-4: a second call on an applied claim is a no-op."""
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _take_claim(engine, log_id, target)
    engine._record_confirmed_use(target, whisper_log_id=log_id)

    after_first = _snapshot(engine, target)
    engine._record_confirmed_use(target, whisper_log_id=log_id)

    assert _snapshot(engine, target) == after_first, "the second call reinforced again"


def test_a_failed_latch_never_loads_the_file(engine, monkeypatch):
    """Council #272 finding 2: the load belongs inside the transaction, after
    the latch. Observable consequence — and this test's RED: when the claim is
    already applied, the mutator must return before any file I/O. Today the
    load runs unconditionally before the transaction even opens.
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    _take_claim(engine, log_id, target)
    engine._record_confirmed_use(target, whisper_log_id=log_id)  # applies the claim

    calls = []
    real_load = engine.file_store.load
    monkeypatch.setattr(
        engine.file_store, "load",
        lambda node_id: calls.append(node_id) or real_load(node_id),
    )

    engine._record_confirmed_use(target, whisper_log_id=log_id)  # latch fails

    assert calls == [], (
        "the mutator loaded the markdown for a claim it then refused to apply — "
        "the load must sit inside the transaction, after the at-most-once latch"
    )


def test_missing_node_ends_orphaned_not_applied(engine):
    """#272 D5-7: a deleted node is terminal, and is not recorded as a success.

    The claim is inserted directly for a node_id that has no markdown file. Only
    whisper_log_id carries a foreign key (PRAGMA foreign_keys=ON), so a claim can
    legitimately outlive its node — which is exactly the state being tested.

    'applied' would be a lie of the same kind the legacy migration refuses to write,
    so the assertion pins the distinction, not merely "it stopped being pending".
    """
    target = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, target)
    with engine.db.transaction() as conn:
        # `state` named explicitly: the DEFAULT is terminal (Step 3), and a claim that
        # starts terminal would reach 'orphaned' by never being touched at all.
        conn.execute(
            "INSERT INTO confirmed_use_claims "
            "(whisper_log_id, node_id, claimed_at, state) "
            "VALUES (?, 'ghost-node', datetime('now'), 'pending')",
            (log_id,),
        )

    engine._record_confirmed_use("ghost-node", whisper_log_id=log_id)

    row = _claim_row(engine, log_id, "ghost-node")
    assert row["state"] == "orphaned", (
        "a claim for a deleted node must be orphaned, never pending (retried forever) "
        "nor applied (a reinforcement that never happened)"
    )
    assert row["reinforced_at"] is None


def test_a_claim_written_without_state_is_not_swept(engine):
    """#272 D5-10: an old binary's INSERT must land terminal, not pending.

    Council round 1 (Codex, HIGH, 0.99) found this. The pre-#272 _claim_confirmed_use
    inserts (whisper_log_id, node_id, claimed_at) without naming `state`, so it takes
    whatever the column DEFAULT gives. That binary's _record_confirmed_use has no idea
    the column exists and never marks the row applied — so if the DEFAULT were
    'pending', the row would sit there forever and the new sweeper would reinforce it
    again every hour, on top of the reinforcement the old binary already did.

    This is not hypothetical: CLAUDE.md documents issue #238, where `make server`
    starts a second, launchd-unmanaged process against the same store. Two binaries on
    one database is a state this project has already been in.

    The DEFAULT is therefore terminal ('legacy_unknown') and only the new code inserts
    'pending' explicitly. This test writes the OLD statement verbatim.
    """
    node_id = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, node_id)

    with engine.db.transaction() as conn:
        # Verbatim the pre-#272 statement: the column list omits `state`.
        conn.execute(
            "INSERT INTO confirmed_use_claims (whisper_log_id, node_id, claimed_at) "
            "VALUES (?, ?, datetime('now'))",
            (log_id, node_id),
        )

    row = _claim_row(engine, log_id, node_id)
    assert row["state"] == "legacy_unknown", (
        "an old binary's claim landed in a state the sweeper will retry forever"
    )


def test_the_new_claim_path_writes_pending(engine):
    """#272 D5-10b: the guard above must not disable the feature it protects.

    If _claim_confirmed_use stopped naming `state`, every new claim would inherit the
    terminal DEFAULT and nothing would ever be swept — the durability this whole task
    adds would be silently off, and every other test here would still pass because
    they exercise the mutator directly. This is the test that fails in that case.
    """
    node_id = _make_nodes(engine, count=1)[0]
    log_id = _seed_whisper_log(engine, node_id)
    _take_claim(engine, log_id, node_id)

    assert _claim_row(engine, log_id, node_id)["state"] == "pending", (
        "a fresh claim is not pending, so the sweeper can never pick it up"
    )


def test_migration_marks_preexisting_claims_legacy_unknown(tmp_path):
    """#272 D5-8b: pre-#272 claims are neither swept nor recorded as successes.

    Council round 1 (Codex, HIGH) killed the first draft, which stamped them
    reinforced. The premise of this task is that SOME of those claims lost their
    reinforcement; the old schema cannot tell which, so calling them applied would
    hide exactly the data loss the task exists to repair. 'pending' is equally wrong
    — the majority did apply, and re-running them is mass over-reinforcement of an
    at-most-once latch. The assertion pins the third state, not "not pending".
    """
    from ormah.index.db import Database

    db = Database(tmp_path / "m.db")
    db.init_schema()
    db.conn.executescript(
        """
        DROP TABLE confirmed_use_claims;
        CREATE TABLE confirmed_use_claims (
            whisper_log_id INTEGER NOT NULL,
            node_id        TEXT NOT NULL,
            claimed_at     TEXT NOT NULL,
            PRIMARY KEY (whisper_log_id, node_id)
        );
        INSERT INTO confirmed_use_claims VALUES (1, 'n1', '2026-01-01 00:00:00');
        """
    )

    db._migrate()

    row = db.conn.execute(
        "SELECT state, reinforced_at FROM confirmed_use_claims"
    ).fetchone()
    assert row["state"] == "legacy_unknown", (
        "a pre-existing claim was classified, but its outcome is not knowable"
    )
    assert row["reinforced_at"] is None, (
        "reinforced_at asserts a reinforcement happened — it did not, or is unknown"
    )
