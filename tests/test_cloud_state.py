from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import uuid

from ormah.cloud.state import (
    CloudState,
    cloud_status_payload,
    load_state,
    save_state,
    state_path,
    update_state,
)
from ormah.config import Settings


def test_state_round_trip_is_atomic_and_owner_only(tmp_path):
    store_id = str(uuid.uuid4())
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    state = CloudState(
        last_upload_at=now,
        last_upload_snapshot_id="01UPLOAD",
        last_verify_at=now + timedelta(hours=1),
        last_verify_ok=False,
        last_verify_snapshot_id="01VERIFY",
        last_verify_error="hash mismatch",
    )

    save_state(store_id, state, state_dir=tmp_path)

    path = state_path(store_id, state_dir=tmp_path)
    assert load_state(store_id, state_dir=tmp_path) == state
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_update_preserves_unmodified_fields(tmp_path):
    store_id = str(uuid.uuid4())
    save_state(
        store_id,
        CloudState(last_upload_snapshot_id="old", last_verify_ok=True),
        state_dir=tmp_path,
    )

    updated = update_state(
        store_id,
        state_dir=tmp_path,
        last_upload_error="offline",
    )

    assert updated.last_upload_snapshot_id == "old"
    assert updated.last_verify_ok is True
    assert updated.last_upload_error == "offline"


def test_update_preserves_future_state_fields(tmp_path):
    store_id = str(uuid.uuid4())
    path = state_path(store_id, state_dir=tmp_path)
    path.write_text(
        json.dumps({"last_synced_snapshot_id": "future-e06", "last_verify_ok": True}),
        encoding="utf-8",
    )

    update_state(store_id, state_dir=tmp_path, last_upload_error="retrying")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["last_synced_snapshot_id"] == "future-e06"
    assert payload["last_upload_error"] == "retrying"


def test_two_stores_never_share_state(tmp_path):
    first = str(uuid.uuid4())
    second = str(uuid.uuid4())

    update_state(first, state_dir=tmp_path, last_upload_snapshot_id="first")
    update_state(second, state_dir=tmp_path, last_upload_snapshot_id="second")

    assert load_state(first, state_dir=tmp_path).last_upload_snapshot_id == "first"
    assert load_state(second, state_dir=tmp_path).last_upload_snapshot_id == "second"
    assert state_path(first, state_dir=tmp_path) != state_path(second, state_dir=tmp_path)


def test_missing_or_malformed_state_loads_empty(tmp_path):
    store_id = str(uuid.uuid4())
    assert load_state(store_id, state_dir=tmp_path) == CloudState()
    state_path(store_id, state_dir=tmp_path).write_text("{not-json", encoding="utf-8")
    assert load_state(store_id, state_dir=tmp_path) == CloudState()


def test_state_path_rejects_path_traversal(tmp_path):
    try:
        state_path("../outside", state_dir=tmp_path)
    except ValueError as exc:
        assert "UUIDv4" in str(exc)
    else:
        raise AssertionError("invalid store id was accepted")


def test_cloud_status_derives_stale_and_verification_warnings(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    save_state(
        store_id,
        CloudState(
            last_upload_at=now - timedelta(hours=49),
            last_upload_snapshot_id="01OLD",
            last_verify_at=now - timedelta(days=1),
            last_verify_ok=False,
            last_verify_error="bundle truncated",
        ),
        state_dir=tmp_path / "state",
    )
    settings = Settings(
        memory_dir=memory_dir,
        cloud_backup_enabled=True,
        cloud_backup_interval_hours=24,
    )

    payload = cloud_status_payload(
        settings,
        entitlement="expired",
        now=now,
        state_dir=tmp_path / "state",
    )

    assert payload["store_id"] == store_id
    assert payload["last_upload_age_seconds"] == 49 * 3600
    assert payload["entitlement"] == "expired"
    assert any("stale" in warning for warning in payload["warnings"])
    assert any("bundle truncated" in warning for warning in payload["warnings"])


def test_cloud_state_json_contains_only_plain_metadata(tmp_path):
    store_id = str(uuid.uuid4())
    save_state(store_id, CloudState(last_upload_snapshot_id="01SAFE"), state_dir=tmp_path)

    payload = json.loads(state_path(store_id, state_dir=tmp_path).read_text(encoding="utf-8"))

    assert payload["last_upload_snapshot_id"] == "01SAFE"
    assert set(payload) == {
        "last_upload_at",
        "last_upload_snapshot_id",
        "last_upload_error",
        "last_verify_at",
        "last_verify_ok",
        "last_verify_snapshot_id",
        "last_verify_error",
    }


def test_admin_cloud_status_is_thin_wrapper(monkeypatch):
    from ormah.api import routes_admin

    settings = SimpleNamespace(memory_dir=Path("/tmp/memory"))
    expected = {"last_verify_ok": True, "warnings": []}
    monkeypatch.setattr("ormah.cloud.state.cloud_status_payload", lambda value: expected)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine=SimpleNamespace(settings=settings))))

    assert routes_admin.cloud_status(request) == expected
    assert any(route.path == "/admin/cloud-status" for route in routes_admin.router.routes)
