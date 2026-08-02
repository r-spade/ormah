"""Issue #90: maintenance runs return a stats dict."""
import pytest

from ormah.background import auto_linker
from ormah.background import duplicate_merger
from ormah.background.auto_linker import run_auto_linker
from ormah.background.conflict_detector import run_conflict_detection
from ormah.background.duplicate_merger import run_duplicate_detection
from ormah.background.job_tracker import JobTracker, tracked


def test_auto_linker_returns_stats(engine):
    engine.settings.llm_provider = "none"  # llm disabled -> early return, still a dict
    stats = run_auto_linker(engine)
    assert stats == {"skipped": "llm_disabled"}


def test_conflict_detector_stats_shape(engine):
    # enable llm; no candidates exist, so no LLM call happens
    engine.settings.llm_provider = "ollama"
    stats = run_conflict_detection(engine)
    for key in ("candidates_found", "pairs_evaluated", "edges_created", "duration_s"):
        assert key in stats


def test_duplicate_merger_stats_shape(engine):
    engine.settings.llm_provider = "ollama"
    stats = run_duplicate_detection(engine)
    for key in ("nodes_scanned", "pairs_evaluated", "proposals_created", "duration_s"):
        assert key in stats


def test_run_failure_is_visible_as_error_stats(engine, monkeypatch):
    """A run whose internals raise must NOT look like a clean, empty success."""
    engine.settings.llm_provider = "ollama"

    def boom(conn):
        raise RuntimeError("watermark read failed")

    monkeypatch.setattr(auto_linker, "_get_watermark", boom)

    stats = run_auto_linker(engine)
    assert stats is not None
    assert stats["error"] == "watermark read failed"

    # tracked() must route the error dict to record_failure, not record_success
    # (council R2 finding 4: last_stats alone doesn't prove the routing).
    tracker = JobTracker()
    job = tracked(tracker, "auto_linker", run_auto_linker, engine)
    job()
    snap = tracker.snapshot()["auto_linker"]
    assert snap["error_count"] == 1
    assert snap["last_error"] == "watermark read failed"
    assert snap["last_success"] is None
    assert snap["last_stats"] is not None
    assert "error" in snap["last_stats"]


def test_find_link_candidates_propagates_finder_failure(engine, monkeypatch):
    """Issue #90 council R2 finding 1: a DB/encoder failure inside the finder
    must not be swallowed as "no candidates". _find_link_candidates is only
    called by get_maintenance_batches (run_auto_linker has its own inline
    candidate loop and never calls this finder), so there is no run_*/tracked()
    path to exercise here — the finder itself must raise."""

    def boom(*a, **kw):
        raise RuntimeError("encoder boom")

    # get_encoder is imported locally inside the function; patch the source module.
    import ormah.embeddings.encoder as encoder_mod
    monkeypatch.setattr(encoder_mod, "get_encoder", boom)

    with pytest.raises(RuntimeError, match="encoder boom"):
        auto_linker._find_link_candidates(engine)


def test_find_merge_candidates_propagates_finder_failure(engine, monkeypatch):
    """Same as above for duplicate_merger's finder (also only reachable via
    get_maintenance_batches, never via the scheduled run_duplicate_detection)."""

    def boom(*a, **kw):
        raise RuntimeError("encoder boom")

    import ormah.embeddings.encoder as encoder_mod
    monkeypatch.setattr(encoder_mod, "get_encoder", boom)

    with pytest.raises(RuntimeError, match="encoder boom"):
        duplicate_merger._find_merge_candidates(engine)


def test_conflict_detector_finder_failure_is_visible_via_tracked(engine, monkeypatch):
    """Issue #90 council R2 finding 1: unlike auto_linker/duplicate_merger,
    run_conflict_detection DOES call _find_conflict_candidates directly, so an
    internal finder failure (not the finder-doesn't-exist case) must surface
    through tracked() as a recorded failure, not a green run. Patches the
    encoder (called unconditionally near the top of the finder) rather than
    the finder function itself, so this actually exercises the finder's own
    (now-removed) blanket except."""
    import ormah.embeddings.encoder as encoder_mod

    def boom(*a, **kw):
        raise RuntimeError("encoder boom")

    monkeypatch.setattr(encoder_mod, "get_encoder", boom)
    engine.settings.llm_provider = "ollama"

    tracker = JobTracker()
    job = tracked(tracker, "conflict_detector", run_conflict_detection, engine)
    job()
    snap = tracker.snapshot()["conflict_detector"]
    assert snap["error_count"] == 1
    assert snap["last_success"] is None
    assert snap["last_error"] is not None


def test_consolidator_finder_failure_is_visible_via_tracked(engine, monkeypatch):
    """run_consolidation calls _find_consolidation_clusters directly and has no
    catch-all of its own — an internal VectorStore-construction failure must
    reach tracked()'s own except, not the finder's (now-removed) blanket except."""
    from ormah.background.consolidator import run_consolidation
    from ormah.models.node import CreateNodeRequest, NodeType

    # Need >= 2 working-tier nodes to get past the "nothing to cluster" early
    # return and actually reach the VectorStore(...) construction.
    for i in range(2):
        engine.remember(CreateNodeRequest(content=f"note number {i}", type=NodeType.fact, title=f"n{i}"))

    import ormah.embeddings.vector_store as vs_mod

    class _BoomVectorStore:
        def __init__(self, *a, **kw):
            raise RuntimeError("vector store boom")

    monkeypatch.setattr(vs_mod, "VectorStore", _BoomVectorStore)
    engine.settings.llm_provider = "ollama"

    tracker = JobTracker()
    job = tracked(tracker, "consolidator", run_consolidation, engine)
    job()
    snap = tracker.snapshot()["consolidator"]
    assert snap["error_count"] == 1
    assert snap["last_success"] is None
    assert snap["last_error"] is not None


def _spy_add_job(monkeypatch):
    from apscheduler.schedulers.background import BackgroundScheduler

    recorded = {}
    orig_add = BackgroundScheduler.add_job

    def spy(self, func, *a, **kw):
        recorded[kw.get("id")] = kw.get("next_run_time")
        return orig_add(self, func, *a, **kw)

    monkeypatch.setattr(BackgroundScheduler, "add_job", spy)
    monkeypatch.setattr(BackgroundScheduler, "start", lambda self: None)
    return recorded


def test_staggered_offsets_exact_at_default_intervals(engine, monkeypatch):
    """At the 1440-minute defaults the nominal offsets (5/15/30/45) are
    unscaled — shortest configured interval (1440) >> reference (60)."""
    from datetime import datetime, timedelta, timezone
    from ormah.background import scheduler as sched_mod

    recorded = _spy_add_job(monkeypatch)
    before = datetime.now(timezone.utc)
    sched_mod.start_scheduler(engine)
    after = datetime.now(timezone.utc)

    expected_offsets = {"auto_linker": 5, "conflict_detector": 15,
                         "duplicate_merger": 30, "consolidator": 45}
    for job_id, nominal in expected_offsets.items():
        t = recorded[job_id]
        assert before + timedelta(minutes=nominal) <= t <= after + timedelta(minutes=nominal)


def test_staggered_offsets_share_one_factor_across_mixed_intervals(engine, monkeypatch):
    """Issue #90 council R3 finding 2: scaling each job by ITS OWN interval
    let jobs with different intervals collide (e.g. auto_link=3min nominal-5
    -> 0.25min, conflict_check=1min nominal-15 -> 0.25min: both fire at 15s
    and re-collide every 3 minutes). All four jobs must share ONE factor
    derived from the shortest configured interval, so distinct nominal
    offsets (5/15/30/45) stay distinct regardless of which job has the
    shortest interval."""
    from datetime import datetime, timedelta, timezone
    from ormah.background import scheduler as sched_mod

    engine.settings.auto_link_interval_minutes = 3
    engine.settings.conflict_check_interval_minutes = 1
    # duplicate_check / consolidation stay at their 1440-minute default.

    recorded = _spy_add_job(monkeypatch)

    before = datetime.now(timezone.utc)
    sched_mod.start_scheduler(engine)

    llm_jobs_intervals = {
        "auto_linker": engine.settings.auto_link_interval_minutes,
        "conflict_detector": engine.settings.conflict_check_interval_minutes,
        "duplicate_merger": engine.settings.duplicate_check_interval_minutes,
        "consolidator": engine.settings.consolidation_interval_minutes,
    }
    times = []
    for job_id, interval_minutes in llm_jobs_intervals.items():
        t = recorded[job_id]
        assert t is not None
        assert t < before + timedelta(minutes=interval_minutes)
        times.append(t)

    assert len({t.replace(microsecond=0) for t in times}) == len(times), (
        f"offsets collapsed across mixed intervals: {times}"
    )
