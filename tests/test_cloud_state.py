from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import uuid

import pytest

from ormah.cloud.state import (
    CURRENT_CLOUD_STATE_SCHEMA_VERSION,
    CloudState,
    CloudStateLoadError,
    CloudStateVersionError,
    ProtectionIntentStatus,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    ProtectionReasonCode,
    ProtectionState,
    UploadJournalPhase,
    cloud_status_payload,
    is_device_loss_recovery_ready,
    is_protected_and_verified,
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
        protection_enabled_at=now,
        protection_state=ProtectionState.ATTENTION_REQUIRED,
        pending_protection_intent_id="intent-1",
        pending_protection_status=ProtectionIntentStatus.RUNNING,
        pending_protection_snapshot_id="snapshot-1",
        pending_protection_created_store=True,
        pending_protection_origin_state=ProtectionState.STOPPED,
        last_operation_id="operation-1",
        last_operation_kind=ProtectionOperationKind.VERIFY,
        last_operation_phase=ProtectionOperationPhase.FAILED,
        last_error_code=ProtectionReasonCode.VERIFICATION_FAILED,
        pending_upload_id="upload-1",
        pending_upload_snapshot_id="01PENDING",
        pending_upload_operation_id="operation-1",
        pending_upload_protection_intent_id="intent-1",
        pending_upload_phase=UploadJournalPhase.FINALIZING,
        pending_upload_expires_at=now + timedelta(minutes=15),
    )

    save_state(store_id, state, memory_dir=tmp_path / "memory", state_dir=tmp_path)

    path = state_path(store_id, state_dir=tmp_path)
    assert load_state(store_id, state_dir=tmp_path) == state
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_reserved_upload_does_not_hide_last_verified_protection():
    values = {
        "last_successful_backup_snapshot_id": "01J00000000000000000000000",
        "last_verified_snapshot_id": "01J00000000000000000000000",
        "last_verify_ok": True,
    }

    assert is_protected_and_verified(
        CloudState(**values, pending_upload_phase=UploadJournalPhase.RESERVED),
        enabled=True,
    )
    assert not is_protected_and_verified(
        CloudState(**values, pending_upload_phase=UploadJournalPhase.FINALIZING),
        enabled=True,
    )


def test_device_loss_readiness_requires_saved_kit_and_current_verified_snapshot():
    verified_at = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
    current = CloudState(
        last_successful_backup_snapshot_id="snapshot-a",
        last_verified_snapshot_id="snapshot-a",
        last_verify_ok=True,
        recovery_kit_verified_at=verified_at,
    )

    assert is_device_loss_recovery_ready(current, enabled=True)
    assert not is_device_loss_recovery_ready(
        CloudState(
            last_successful_backup_snapshot_id="snapshot-a",
            last_verified_snapshot_id="snapshot-a",
            last_verify_ok=True,
        ),
        enabled=True,
    )
    assert not is_device_loss_recovery_ready(
        CloudState(
            last_successful_backup_snapshot_id="snapshot-b",
            last_verified_snapshot_id="snapshot-a",
            last_verify_ok=True,
            recovery_kit_verified_at=verified_at,
        ),
        enabled=True,
    )
    assert not is_device_loss_recovery_ready(current, enabled=False)


def test_cloud_status_exposes_only_derived_recovery_readiness(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    verified_at = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.PROTECTED,
            last_successful_backup_snapshot_id="snapshot-a",
            last_verified_snapshot_id="snapshot-a",
            last_verify_ok=True,
            recovery_kit_verified_at=verified_at,
        ),
        memory_dir=memory_dir,
        state_dir=tmp_path / "state",
    )

    payload = cloud_status_payload(
        Settings(memory_dir=memory_dir, cloud_backup_enabled=True),
        entitlement="active",
        state_dir=tmp_path / "state",
    )

    assert payload["device_loss_recovery_ready"] is True
    assert payload["recovery_kit_verified_at"] == verified_at.isoformat()
    serialized = json.dumps(payload).lower()
    assert "age-secret-key" not in serialized
    assert "ormah-recovery-kit.md" not in serialized


def test_update_preserves_unmodified_fields(tmp_path):
    store_id = str(uuid.uuid4())
    save_state(
        store_id,
        CloudState(last_upload_snapshot_id="old", last_verify_ok=True),
        memory_dir=tmp_path / "memory",
        state_dir=tmp_path,
    )

    updated = update_state(
        store_id,
        memory_dir=tmp_path / "memory",
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

    update_state(
        store_id,
        memory_dir=tmp_path / "memory",
        state_dir=tmp_path,
        last_upload_error="retrying",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["last_synced_snapshot_id"] == "future-e06"
    assert payload["last_upload_error"] == "retrying"


def test_legacy_state_migrates_success_fields_and_protection_state(tmp_path):
    store_id = str(uuid.uuid4())
    now = "2026-07-13T12:00:00+00:00"
    path = state_path(store_id, state_dir=tmp_path)
    path.write_text(
        json.dumps(
            {
                "last_upload_at": now,
                "last_upload_snapshot_id": "01VERIFIED",
                "last_verify_at": now,
                "last_verify_ok": True,
                "last_verify_snapshot_id": "01VERIFIED",
            }
        ),
        encoding="utf-8",
    )

    state = load_state(store_id, state_dir=tmp_path)

    assert state.schema_version == CURRENT_CLOUD_STATE_SCHEMA_VERSION
    assert state.protection_state is ProtectionState.PROTECTED
    assert state.last_successful_upload_at == state.last_upload_at
    assert state.last_successful_backup_snapshot_id == "01VERIFIED"
    assert state.last_successful_verify_at == state.last_verify_at
    assert state.last_verified_snapshot_id == "01VERIFIED"


def test_newer_schema_is_readable_but_an_older_writer_refuses_to_overwrite_it(tmp_path):
    store_id = str(uuid.uuid4())
    path = state_path(store_id, state_dir=tmp_path)
    path.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "protection_state": "future_safe_state",
                "last_operation_kind": "future_operation",
                "future_c03_field": {"head": "01HEAD"},
            }
        ),
        encoding="utf-8",
    )

    before = path.read_bytes()
    state = load_state(store_id, state_dir=tmp_path)

    assert state.schema_version == 99
    assert state.protection_state == "future_safe_state"
    assert state.last_operation_kind == "future_operation"
    assert state.extra["future_c03_field"] == {"head": "01HEAD"}
    with pytest.raises(CloudStateVersionError, match="newer than this client's schema"):
        update_state(
            store_id,
            memory_dir=tmp_path / "memory",
            state_dir=tmp_path,
            last_upload_error="offline",
        )
    with pytest.raises(CloudStateVersionError, match="newer than this client's schema"):
        save_state(
            store_id,
            CloudState(last_upload_error="offline"),
            memory_dir=tmp_path / "memory",
            state_dir=tmp_path,
        )

    assert path.read_bytes() == before


def test_two_stores_never_share_state(tmp_path):
    first = str(uuid.uuid4())
    second = str(uuid.uuid4())

    update_state(
        first,
        memory_dir=tmp_path / "first-memory",
        state_dir=tmp_path,
        last_upload_snapshot_id="first",
    )
    update_state(
        second,
        memory_dir=tmp_path / "second-memory",
        state_dir=tmp_path,
        last_upload_snapshot_id="second",
    )

    assert load_state(first, state_dir=tmp_path).last_upload_snapshot_id == "first"
    assert load_state(second, state_dir=tmp_path).last_upload_snapshot_id == "second"
    assert state_path(first, state_dir=tmp_path) != state_path(second, state_dir=tmp_path)


def test_missing_state_loads_empty_but_malformed_state_fails_closed(tmp_path):
    store_id = str(uuid.uuid4())
    assert load_state(store_id, state_dir=tmp_path) == CloudState()
    path = state_path(store_id, state_dir=tmp_path)
    path.write_text("{not-json", encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(CloudStateLoadError, match="is invalid"):
        load_state(store_id, state_dir=tmp_path)
    with pytest.raises(CloudStateLoadError, match="is invalid"):
        update_state(
            store_id,
            memory_dir=tmp_path / "memory",
            state_dir=tmp_path,
            last_upload_error="must not replace corrupt state",
        )
    with pytest.raises(CloudStateLoadError, match="is invalid"):
        save_state(
            store_id,
            CloudState(last_upload_error="must not replace corrupt state"),
            memory_dir=tmp_path / "memory",
            state_dir=tmp_path,
        )

    assert path.read_bytes() == before


def test_one_invalid_field_cannot_erase_durable_protection_state(tmp_path):
    store_id = str(uuid.uuid4())
    path = state_path(store_id, state_dir=tmp_path)
    payload = {
        "schema_version": 2,
        "last_upload_at": 1_753_000_000,
        "protection_enabled_at": "2026-07-13T12:00:00+00:00",
        "protection_state": "protected",
        "pending_protection_intent_id": str(uuid.uuid4()),
        "future_c03_field": {"head": "01HEAD"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(CloudStateLoadError, match="timestamps must be ISO-8601"):
        update_state(
            store_id,
            memory_dir=tmp_path / "memory",
            state_dir=tmp_path,
            last_verify_error="routine writer",
        )

    assert path.read_bytes() == before


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
            protection_state=ProtectionState.ATTENTION_REQUIRED,
        ),
        memory_dir=memory_dir,
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
    assert payload["protection_state"] == ProtectionState.ATTENTION_REQUIRED.value
    assert payload["state_error"] is None
    assert any("stale" in warning for warning in payload["warnings"])
    assert any("bundle truncated" in warning for warning in payload["warnings"])


def test_cloud_status_reports_corrupt_state_without_claiming_protection(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    path = state_path(store_id, state_dir=tmp_path / "state")
    path.parent.mkdir(parents=True)
    path.write_text('{"protection_state":"protected","last_verify_ok":"yes"}', encoding="utf-8")
    settings = Settings(memory_dir=memory_dir, cloud_backup_enabled=True)

    payload = cloud_status_payload(
        settings,
        entitlement="active",
        state_dir=tmp_path / "state",
    )

    assert payload["state_error"] is not None
    assert payload["protection_state"] == ProtectionState.ATTENTION_REQUIRED.value
    assert any("was not overwritten" in warning for warning in payload["warnings"])
    assert json.loads(path.read_text(encoding="utf-8"))["protection_state"] == "protected"


def test_cloud_status_exposes_pollable_intent_without_account_binding(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    intent_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.SUBSCRIPTION_REQUIRED,
            pending_protection_intent_id=intent_id,
            pending_protection_account_id=str(uuid.uuid4()),
            pending_protection_store_id=store_id,
            pending_protection_created_at=now,
            pending_protection_expires_at=now + timedelta(minutes=30),
            pending_protection_status=ProtectionIntentStatus.ACCOUNT_BOUND,
            last_operation_id=intent_id,
            last_operation_kind=ProtectionOperationKind.ENABLE,
            last_operation_phase=ProtectionOperationPhase.PENDING,
        ),
        memory_dir=memory_dir,
        state_dir=tmp_path / "state",
    )

    payload = cloud_status_payload(
        Settings(memory_dir=memory_dir),
        entitlement="expired",
        now=now,
        state_dir=tmp_path / "state",
    )

    assert payload["protection_state"] == ProtectionState.SUBSCRIPTION_REQUIRED.value
    assert payload["protection_intent_id"] == intent_id
    assert payload["protection_intent_status"] == ProtectionIntentStatus.ACCOUNT_BOUND.value
    assert payload["last_operation_id"] == intent_id
    serialized = json.dumps(payload)
    assert "pending_protection_account_id" not in serialized


def test_cloud_status_never_reports_protected_when_invariant_is_incomplete(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    intent_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.PROTECTED,
            pending_protection_intent_id=intent_id,
            pending_protection_status=ProtectionIntentStatus.COMPLETED,
            last_successful_backup_snapshot_id="snapshot-a",
            last_verified_snapshot_id="snapshot-b",
            last_verify_ok=True,
        ),
        memory_dir=memory_dir,
        state_dir=tmp_path / "state",
    )

    payload = cloud_status_payload(
        Settings(memory_dir=memory_dir, cloud_backup_enabled=True),
        entitlement="active",
        state_dir=tmp_path / "state",
    )

    assert payload["protection_state"] == ProtectionState.ATTENTION_REQUIRED.value
    assert any("verification must finish" in item for item in payload["warnings"])


def test_cloud_status_fails_closed_for_future_protection_state(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    path = state_path(store_id, state_dir=tmp_path / "state")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": CURRENT_CLOUD_STATE_SCHEMA_VERSION + 1,
                "protection_state": "future_protected",
            }
        ),
        encoding="utf-8",
    )

    payload = cloud_status_payload(
        Settings(memory_dir=memory_dir, cloud_backup_enabled=True),
        entitlement="active",
        state_dir=tmp_path / "state",
    )

    assert payload["protection_state"] == ProtectionState.ATTENTION_REQUIRED.value
    assert any("newer Ormah version" in item for item in payload["warnings"])
    assert json.loads(path.read_text(encoding="utf-8"))["protection_state"] == (
        "future_protected"
    )


def test_cloud_status_derives_paused_when_upload_entitlement_ends(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    intent_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.PROTECTED,
            pending_protection_intent_id=intent_id,
            pending_protection_status=ProtectionIntentStatus.COMPLETED,
            last_successful_backup_snapshot_id="snapshot-a",
            last_verified_snapshot_id="snapshot-a",
            last_verify_ok=True,
        ),
        memory_dir=memory_dir,
        state_dir=tmp_path / "state",
    )

    payload = cloud_status_payload(
        Settings(memory_dir=memory_dir, cloud_backup_enabled=True),
        entitlement="expired",
        state_dir=tmp_path / "state",
    )

    assert payload["protection_state"] == ProtectionState.PAUSED.value


def test_cloud_status_reports_uploaded_snapshot_awaiting_verification(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.VERIFICATION_PENDING,
            last_successful_backup_snapshot_id="snapshot-new",
            last_verified_snapshot_id="snapshot-old",
            last_verify_ok=True,
        ),
        memory_dir=memory_dir,
        state_dir=tmp_path / "state",
    )

    payload = cloud_status_payload(
        Settings(memory_dir=memory_dir, cloud_backup_enabled=True),
        entitlement="active",
        state_dir=tmp_path / "state",
    )

    assert payload["protection_state"] == ProtectionState.VERIFICATION_PENDING.value
    assert not any("changes pending" in item.lower() for item in payload["warnings"])


def test_cloud_status_requires_sign_in_after_logout_of_protected_store(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.PROTECTED,
            last_successful_backup_snapshot_id="snapshot-a",
            last_verified_snapshot_id="snapshot-a",
            last_verify_ok=True,
        ),
        memory_dir=memory_dir,
        state_dir=tmp_path / "state",
    )

    payload = cloud_status_payload(
        Settings(
            memory_dir=memory_dir,
            cloud_backup_enabled=True,
            account_token=None,
        ),
        entitlement="none",
        state_dir=tmp_path / "state",
    )

    assert payload["protection_state"] == ProtectionState.SIGN_IN_REQUIRED.value


def test_cloud_state_json_contains_only_plain_metadata(tmp_path):
    store_id = str(uuid.uuid4())
    save_state(
        store_id,
        CloudState(last_upload_snapshot_id="01SAFE"),
        memory_dir=tmp_path / "memory",
        state_dir=tmp_path,
    )

    payload = json.loads(state_path(store_id, state_dir=tmp_path).read_text(encoding="utf-8"))

    assert payload["last_upload_snapshot_id"] == "01SAFE"
    assert set(payload) == set(CloudState().to_dict())
    serialized = json.dumps(payload).lower()
    for forbidden in ("account_token", "presigned", "age-secret-key", "recovery_phrase"):
        assert forbidden not in serialized


def test_admin_cloud_status_is_thin_wrapper(monkeypatch):
    from ormah.api import routes_admin

    settings = SimpleNamespace(memory_dir=Path("/tmp/memory"))
    expected = {"last_verify_ok": True, "warnings": []}
    monkeypatch.setattr("ormah.cloud.state.cloud_status_payload", lambda value: expected)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine=SimpleNamespace(settings=settings))))

    assert routes_admin.cloud_status(request) == expected
    assert any(route.path == "/admin/cloud-status" for route in routes_admin.router.routes)
