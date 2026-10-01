"""Issue #320 regressions: real manager, routes and MCP; synthetic memories only.

Adapted from the supplied reproduction. Fake elapsed time and thread events make
interleavings explicit; no daemon, external HTTP, model or production store runs.
"""

import asyncio
import copy
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from ormah.adapters import mcp_adapter
from ormah.api.routes_agent import router
from ormah.background.maintenance_manager import MaintenanceManager
from ormah.config import Settings
from ormah.engine.maintenance_signal import MAINTENANCE_DUE_SIGNAL, maintenance_due_signal
from ormah.engine.memory_engine import MemoryEngine


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class WorkQueue:
    def __init__(self):
        self.settings = SimpleNamespace(
            maintenance_timeout_minutes=30,
            claude_maintenance_enabled=True,
            claude_maintenance_interval_hours=24,
        )
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        self.graph = SimpleNamespace(conn=self.conn)
        self.batch_count = 0
        self.applied = []
        self.prep_started = threading.Event()
        self.release_prep = threading.Event()
        self.apply_started = threading.Event()
        self.release_apply = threading.Event()
        self.release_prep.set()
        self.release_apply.set()
        self.fail_prep = False
        self.fail_apply = False

    def get_maintenance_batches(self):
        self.batch_count += 1
        self.prep_started.set()
        assert self.release_prep.wait(5), "test did not release preparation"
        if self.fail_prep:
            raise RuntimeError("synthetic preparation failure")
        return {"summary": f"batch {self.batch_count}", "link_candidates": []}

    def apply_maintenance_results(self, results):
        self.applied.append(copy.deepcopy(results))
        self.apply_started.set()
        assert self.release_apply.wait(5), "test did not release application"
        if self.fail_apply:
            raise RuntimeError("synthetic application failure")
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('last_maintenance_run', ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        self.conn.commit()
        return {"accepted": results}

    def last_success(self):
        return self.conn.execute("SELECT value FROM meta").fetchone()


class ObservedManager(MaintenanceManager):
    """Expose thread completion events without timing-dependent status loops."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = threading.Event()
        self.finished = threading.Event()

    def _run_phase1(self, job_id):
        try:
            super()._run_phase1(job_id)
        finally:
            self.prepared.set()

    def _run_phase2(self, job_id, results):
        try:
            super()._run_phase2(job_id, results)
        finally:
            self.finished.set()


class Harness:
    def __init__(self):
        self.engine = WorkQueue()
        self.clock = Clock()
        self.manager = ObservedManager(self.engine, clock=self.clock)

    def acquire_ready(self):
        self.manager.prepared.clear()
        grant = self.manager.start_phase1()
        assert self.manager.prepared.wait(5)
        return self.manager.get_status(grant["job_id"])

    def due(self):
        # Both engine and context builder delegate to this eligibility helper.
        signal = maintenance_due_signal(self.engine, self.engine.conn)
        assert MemoryEngine._maybe_get_maintenance_due_signal(self.engine) == signal
        return signal


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.engine.release_prep.set()
    harness.engine.release_apply.set()
    if harness.engine.prep_started.is_set():
        assert harness.manager.prepared.wait(5)
    if harness.engine.apply_started.is_set():
        assert harness.manager.finished.wait(5)
    harness.engine.conn.close()


def test_atomic_acquisition_only_winner_gets_receipt_and_work(h):
    h.engine.release_prep.clear()
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(5)
        return h.manager.start_phase1()

    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(lambda _: claim(), range(2)))
    winner, = [r for r in replies if r["status"] == "running_phase1"]
    busy, = [r for r in replies if r["status"] == "busy"]
    assert set(busy) == {"status", "message"}
    assert h.engine.prep_started.wait(5)
    assert h.engine.batch_count == 1
    assert h.manager.get_status(winner["job_id"])["status"] == "running_phase1"
    h.engine.release_prep.set()
    assert h.manager.prepared.wait(5)
    assert h.manager.get_status(winner["job_id"])["batches"]["summary"] == "batch 1"
    assert h.manager.start_phase1()["status"] == "busy"


def test_deadline_starts_when_ready_and_status_never_renews_it(h):
    h.engine.release_prep.clear()
    receipt = h.manager.start_phase1()["job_id"]
    h.clock.now = 10_000
    assert h.manager.get_status(receipt)["status"] == "running_phase1"
    assert not h.due()
    h.engine.release_prep.set()
    assert h.manager.prepared.wait(5)
    ready = h.manager.get_status(receipt)
    assert ready["expires_at"] is not None
    h.clock.now += 1799.999
    for _ in range(3):
        assert h.manager.get_status(receipt) == ready
    assert h.manager.has_active_reservation()
    h.clock.now = 11_800
    assert h.manager.get_status(receipt)["status"] == "expired"
    assert not h.manager.has_active_reservation()
    assert h.due() == MAINTENANCE_DUE_SIGNAL
    assert h.engine.last_success() is None


@pytest.mark.parametrize("expiry_check", ["submit", "status", "due", "acquire"])
def test_expiry_checked_at_every_boundary_and_stale_cannot_release_replacement(h, expiry_check):
    old = h.acquire_ready()["job_id"]
    h.clock.now = 1800
    if expiry_check == "submit":
        with pytest.raises(RuntimeError, match="expired.*Discard stale"):
            h.manager.submit_results({}, old)
    elif expiry_check == "status":
        assert h.manager.get_status(old)["status"] == "expired"
    elif expiry_check == "due":
        assert h.due() == MAINTENANCE_DUE_SIGNAL
    if expiry_check != "acquire":
        with pytest.raises(RuntimeError, match="expired"):
            h.manager.submit_results({}, old)  # no replacement yet
    current = h.acquire_ready()
    assert current["job_id"] != old
    with pytest.raises(ValueError, match="mismatch.*Discard stale"):
        h.manager.submit_results({"old": True}, old)
    assert h.manager.get_status() == current
    assert h.manager.get_status(old)["status"] == "replaced"
    assert h.engine.applied == []
    assert h.engine.last_success() is None
    assert not h.due()


def test_valid_results_apply_once_survive_deadline_and_duplicates_are_honest(h):
    assert h.due() == MAINTENANCE_DUE_SIGNAL
    ready = h.acquire_ready()
    assert not h.due()
    job_id = ready["job_id"]
    for bad in (None, "wrong"):
        with pytest.raises(ValueError):
            h.manager.submit_results({}, bad)
    h.engine.release_apply.clear()
    payload = {"edges": [{"edge_type": "supports"}]}
    assert h.manager.submit_results(payload, job_id)["status"] == "running_phase2"
    assert h.engine.apply_started.wait(5)
    h.clock.now = 9999
    assert not h.due()
    assert h.manager.get_status(job_id)["status"] == "running_phase2"
    assert h.manager.start_phase1()["status"] == "busy"
    with pytest.raises(RuntimeError, match="already applying.*Poll"):
        h.manager.submit_results({"edges": []}, job_id)
    assert h.engine.applied == [payload]
    assert h.engine.last_success() is None
    h.engine.release_apply.set()
    assert h.manager.finished.wait(5)
    assert h.manager.get_status(job_id)["apply_summary"] == {"accepted": payload}
    assert not h.manager.has_active_reservation()
    assert not h.due()  # normal interval after actual successful application
    assert h.engine.last_success() is not None
    with pytest.raises(RuntimeError, match="completed"):
        h.manager.submit_results(payload, job_id)
    assert h.engine.applied == [payload]


@pytest.mark.parametrize("phase", ["prep", "apply"])
def test_failure_restores_due_without_recording_success(h, phase):
    setattr(h.engine, f"fail_{phase}", True)
    ready = h.acquire_ready()
    if phase == "apply":
        h.manager.submit_results({}, ready["job_id"])
        assert h.manager.finished.wait(5)
    assert h.manager.get_status()["status"] == "failed"
    assert h.due() == MAINTENANCE_DUE_SIGNAL
    assert h.engine.last_success() is None
    setattr(h.engine, f"fail_{phase}", False)
    assert h.acquire_ready()["job_id"] != ready["job_id"]


@pytest.mark.parametrize("at_deadline", [False, True])
def test_submission_races_claim_without_overlapping_workers(h, at_deadline):
    job_id = h.acquire_ready()["job_id"]
    h.engine.release_apply.clear()
    h.clock.now = 1800 if at_deadline else 1799.999
    if at_deadline:
        h.manager.prepared.clear()
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait(5)
        try:
            return h.manager.submit_results({}, job_id)["status"]
        except (ValueError, RuntimeError):
            return "rejected"

    def claim():
        barrier.wait(5)
        return h.manager.start_phase1()

    with ThreadPoolExecutor(2) as pool:
        submission = pool.submit(submit)
        acquisition = pool.submit(claim)
        result, grant = submission.result(5), acquisition.result(5)
    if at_deadline:
        assert result == "rejected"
        assert grant["job_id"] != job_id
        assert h.engine.applied == []
    else:
        assert result == "running_phase2"
        assert grant["status"] == "busy"
        assert h.engine.apply_started.wait(5)
        assert h.engine.applied == [{}]
        h.clock.now = 1800
        assert h.manager.start_phase1()["status"] == "busy"


def test_restart_loses_receipt_safely(h):
    old = h.acquire_ready()["job_id"]
    restarted = MaintenanceManager(h.engine, clock=h.clock)
    assert restarted.get_status(old)["status"] == "idle"
    with pytest.raises(LookupError, match="lost.*Discard stale"):
        restarted.submit_results({}, old)
    assert h.engine.applied == []
    assert h.due() == MAINTENANCE_DUE_SIGNAL


@pytest.mark.parametrize("minutes", [0, -1])
def test_timeout_must_be_positive(minutes):
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(_env_file=None, maintenance_timeout_minutes=minutes)


def test_timeout_default_and_environment(monkeypatch):
    monkeypatch.delenv("ORMAH_MAINTENANCE_TIMEOUT_MINUTES", raising=False)
    assert Settings(_env_file=None).maintenance_timeout_minutes == 30
    monkeypatch.setenv("ORMAH_MAINTENANCE_TIMEOUT_MINUTES", "7")
    assert Settings(_env_file=None).maintenance_timeout_minutes == 7


@pytest.fixture
def adapter(h, monkeypatch):
    app = FastAPI()
    app.include_router(router)
    app.state.maintenance_manager = h.manager
    app.state.engine = h.engine
    transport = httpx.ASGITransport(app=app)
    real_client = httpx.AsyncClient
    h.posts = []

    async def record(response):
        if response.request.method == "POST":
            h.posts.append((json.loads(response.request.content), response.status_code))

    def client(**kwargs):
        return real_client(**kwargs, transport=transport, event_hooks={"response": [record]})

    async def wait_for_phase():
        state = h.manager.get_status()["status"]
        if state in {"running_phase1", "running_phase2"}:
            done = h.manager.prepared if state == "running_phase1" else h.manager.finished
            assert await asyncio.to_thread(done.wait, 5)

    monkeypatch.setattr(mcp_adapter.httpx, "AsyncClient", client)
    monkeypatch.setattr(mcp_adapter, "_sleep_for_poll_interval", wait_for_phase)

    async def call(args, session="shared-session"):
        return await mcp_adapter._dispatch(
            "http://isolated-test", "run_maintenance", args, session_id=session,
        )

    h.call = call
    h.client = client
    return h


async def test_http_simultaneous_claims_and_receipt_required_for_polling(adapter):
    h = adapter
    h.engine.release_prep.clear()
    async with h.client(base_url="http://isolated-test") as client:
        a, b = await asyncio.gather(*[
            client.post("/agent/maintenance", json={}) for _ in range(2)
        ])
        winner, = [r for r in (a, b) if r.status_code == 202]
        loser, = [r for r in (a, b) if r.json()["status"] == "busy"]
        assert set(loser.json()) == {"status", "message"}
        assert (await client.get("/agent/maintenance")).status_code == 422
        assert (await client.post("/agent/maintenance", json={"job_id": ""})).status_code == 422
        assert (await client.post("/agent/maintenance", json={"results": {}})).status_code == 409
        assert (await client.post("/agent/maintenance", json={"results": None})).status_code == 409
        assert (await client.post("/agent/maintenance", json={
            "job_id": "wrong", "results": {},
        })).status_code == 409
        assert h.engine.prep_started.wait(5)
        assert h.engine.batch_count == 1
        h.engine.release_prep.set()
        assert await asyncio.to_thread(h.manager.prepared.wait, 5)
        ready = await client.get("/agent/maintenance", params={"job_id": winner.json()["job_id"]})
        assert ready.json()["batches"]["summary"] == "batch 1"


async def test_adapter_two_calls_empty_results_and_busy_does_not_steal_receipt(adapter):
    h = adapter
    assignment = await h.call({})
    job_id = h.manager.get_status()["job_id"]
    assert f"job_id: {job_id}" in assignment
    assert "Analysis expires at:" in assignment
    assert json.loads(await h.call({}))["status"] == "busy"
    assert assignment == await h.call({"job_id": job_id})
    assert "required" in await h.call({"results": {}})
    assert h.engine.applied == []
    result = json.loads(await h.call({"job_id": job_id, "results": {}}))
    assert result == {"status": "applied", "job_id": job_id, "summary": {"accepted": {}}}
    assert h.engine.applied == [{}]
    assert "409" in await h.call({"job_id": job_id, "results": {}})


async def test_adapter_shared_session_never_relabels_expired_decisions(adapter):
    h = adapter
    await h.call({})
    old = h.manager.get_status()["job_id"]
    h.clock.now = 1800
    assert "expired" in await h.call({"job_id": old, "results": {"old": True}})
    h.manager.prepared.clear()
    await h.call({})
    current = h.manager.get_status()["job_id"]
    assert current != old
    assert "mismatch" in await h.call({"job_id": old, "results": {"old": True}})
    assert "required" in await h.call({"results": {"old": True}})
    assert h.manager.get_status()["job_id"] == current
    assert h.manager.has_active_reservation()
    assert h.engine.applied == []
    assert json.loads(await h.call({"job_id": old}))["status"] == "replaced"


async def test_adapter_different_concurrent_payload_cannot_receive_false_success(adapter):
    h = adapter
    await h.call({})
    job_id = h.manager.get_status()["job_id"]
    h.engine.release_apply.clear()
    accepted = {"edges": [{"edge_type": "supports"}]}
    task = asyncio.create_task(h.call({"job_id": job_id, "results": accepted}))
    try:
        assert await asyncio.to_thread(h.engine.apply_started.wait, 5)
        rejected = await h.call({"job_id": job_id, "results": {"edges": []}})
        assert "409" in rejected and "already applying" in rejected
        polled = json.loads(await h.call({"job_id": job_id}))
        assert polled["status"] == "running_phase2"
    finally:
        h.engine.release_apply.set()
        applied = json.loads(await task)
    assert applied["summary"] == {"accepted": accepted}
    assert h.engine.applied == [accepted]
    polled = json.loads(await h.call({"job_id": job_id}))
    assert polled["status"] == "completed"
    assert polled["apply_summary"] == {"accepted": accepted}


@pytest.mark.parametrize("phase", ["phase1", "phase2"])
def test_thread_start_failure_releases_reservation(h, monkeypatch, phase):
    if phase == "phase2":
        receipt = h.acquire_ready()["job_id"]

    def cannot_start(_thread):
        raise RuntimeError("synthetic thread startup failure")

    monkeypatch.setattr(threading.Thread, "start", cannot_start)
    status = h.manager.start_phase1() if phase == "phase1" else h.manager.submit_results({}, receipt)
    assert status["status"] == "failed"
    assert not h.manager.has_active_reservation()
    assert h.due() == MAINTENANCE_DUE_SIGNAL
    assert h.engine.applied == []
    assert h.engine.last_success() is None


def test_configured_timeout_controls_deadline(h):
    h.engine.settings.maintenance_timeout_minutes = 7
    h.manager = ObservedManager(h.engine, clock=h.clock)
    job_id = h.acquire_ready()["job_id"]
    h.clock.now = 419.999
    assert h.manager.get_status(job_id)["status"] == "awaiting_results"
    h.clock.now = 420
    assert h.manager.get_status(job_id)["status"] == "expired"


def test_late_preparation_telemetry_failure_cannot_release_applying_job(h):
    telemetry_started = threading.Event()

    class Tracker:
        def record_success(self, name, _duration):
            if name == "maintenance_phase1":
                telemetry_started.set()
                assert h.engine.apply_started.wait(5)
                raise RuntimeError("late preparation telemetry failure")

    h.manager._tracker = Tracker()
    h.engine.release_apply.clear()
    job_id = h.manager.start_phase1()["job_id"]
    assert telemetry_started.wait(5)
    h.manager.submit_results({}, job_id)
    assert h.manager.prepared.wait(5)
    assert h.manager.get_status(job_id)["status"] == "running_phase2"
    assert h.manager.start_phase1()["status"] == "busy"
    h.engine.release_apply.set()
    assert h.manager.finished.wait(5)
    assert h.manager.get_status(job_id)["status"] == "completed"
