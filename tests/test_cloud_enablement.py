from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from ormah.cloud import protection, state
from ormah.cloud.client import CloudError
from ormah.cloud.protection import CloudProtectionService
from ormah.cloud.recovery import RecoveryKitError
from ormah.cloud.state import (
    ProtectionIntentStatus,
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    ProtectionReasonCode,
    ProtectionState,
    UploadJournalPhase,
    cloud_status_payload,
    load_state,
    update_state,
)


NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
ACCOUNT_ID = "37b89c91-c3b8-4f64-9c73-d73ea469abc0"
OTHER_ACCOUNT_ID = "bb86b869-6675-4ec5-a6ea-f17571d0ba57"
SNAPSHOT_ID = "01J00000000000000000000000"


class FakeClient:
    def __init__(self, account_id: str = ACCOUNT_ID, *, backup: bool = True) -> None:
        self.response = {
            "account_id": account_id,
            "backup": backup,
            "plan_status": "active" if backup else "none",
        }
        self.calls = 0
        self.closed = False

    def get_entitlements(self):
        self.calls += 1
        return self.response

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def cloud_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "CLOUD_STATE_DIR", tmp_path / "cloud-state")
    monkeypatch.setattr(protection, "_utc_now", lambda: NOW)
    monkeypatch.setattr(protection, "cache_entitlements", lambda *args, **kwargs: None)
    monkeypatch.setattr(protection, "RECOVERY_KIT_PATH", tmp_path / "recovery-kit.md")
    monkeypatch.setattr(
        protection,
        "RecoveryKitService",
        lambda settings: SimpleNamespace(validate_canonical_kit=lambda: None),
    )


def _settings(tmp_path: Path, *, token: str | None = "token"):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return SimpleNamespace(
        memory_dir=memory_dir,
        backup_dir=tmp_path / "backups",
        cloud_backup_enabled=False,
        cloud_backup_interval_hours=24,
        account_token=token,
        account_email="person@example.com" if token else None,
    )


def _store_id(settings) -> str:
    return (settings.memory_dir / ".store_id").read_text(encoding="utf-8").strip()


def _ready_intent(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    assert intent is not None
    client = FakeClient()
    monkeypatch.setattr(protection, "client_from_settings", lambda settings: client)
    bound = service.bind_intent(intent)
    assert bound.reason_code is None
    assert load_state(_store_id(settings)).pending_protection_status is ProtectionIntentStatus.READY
    return settings, service, intent, client


def test_create_intent_is_durable_for_signed_out_store(tmp_path):
    settings = _settings(tmp_path, token=None)

    result = CloudProtectionService(settings).create_intent()

    assert result.phase is ProtectionOperationPhase.PENDING
    assert result.state is ProtectionState.SIGN_IN_REQUIRED
    assert result.reason_code is ProtectionReasonCode.SIGN_IN_REQUIRED
    durable = load_state(_store_id(settings))
    assert durable.pending_protection_intent_id == result.protection_intent_id
    assert durable.pending_protection_status is ProtectionIntentStatus.PENDING
    assert durable.pending_protection_created_store is True
    assert durable.pending_protection_origin_state is ProtectionState.LOCAL_ONLY
    assert durable.pending_protection_expires_at == NOW + timedelta(minutes=30)


def test_repeated_create_and_bind_keep_one_stable_operation_id(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    first = CloudProtectionService(settings).create_intent()
    restarted = CloudProtectionService(settings)
    second = restarted.create_intent()
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: FakeClient(),
    )
    bound = restarted.bind_intent(first.protection_intent_id)

    assert first.operation_id == first.protection_intent_id
    assert second.operation_id == first.operation_id
    assert bound.operation_id == first.operation_id


def test_create_after_completed_protection_is_idempotent(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    first = service.create_intent()
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: protection.EntitlementStatus.ACTIVE,
    )

    repeated = service.create_intent()

    assert repeated.operation_id == first.operation_id
    assert repeated.phase is ProtectionOperationPhase.COMPLETED
    assert repeated.snapshot_id == SNAPSHOT_ID


def test_create_is_idempotent_for_legacy_protected_store(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store_id = protection.get_or_create_store_id(settings.memory_dir)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: protection.EntitlementStatus.ACTIVE,
    )

    result = CloudProtectionService(settings).create_intent()

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.PROTECTED
    assert result.protection_intent_id is None
    assert load_state(store_id).pending_protection_intent_id is None


def test_offline_entitlement_lookup_does_not_replace_completed_intent(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    first = service.create_intent()
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: protection.EntitlementStatus.NONE,
    )

    repeated = service.create_intent()

    assert repeated.operation_id == first.operation_id
    assert repeated.phase is ProtectionOperationPhase.COMPLETED
    assert load_state(store_id).pending_protection_status is ProtectionIntentStatus.COMPLETED


def test_completed_enable_never_returns_false_protected_state(tmp_path):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        last_successful_backup_snapshot_id="snapshot-a",
        last_verified_snapshot_id="snapshot-b",
        last_verify_ok=True,
    )

    result = service.enable(intent)

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.ATTENTION_REQUIRED


def test_running_intent_survives_checkout_expiry_on_restart(tmp_path):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    first = service.create_intent()
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_status=ProtectionIntentStatus.RUNNING,
        pending_protection_expires_at=NOW - timedelta(minutes=1),
    )

    resumed = CloudProtectionService(settings).create_intent()

    assert resumed.operation_id == first.operation_id
    assert resumed.protection_intent_id == first.protection_intent_id
    assert load_state(store_id).pending_protection_status is ProtectionIntentStatus.RUNNING


def test_new_intent_never_reuses_an_old_verified_snapshot(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    old_intent = service.create_intent()
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.STOPPED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        pending_protection_snapshot_id=SNAPSHOT_ID,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verify_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )
    intent = service.create_intent().protection_intent_id
    assert intent != old_intent.protection_intent_id
    monkeypatch.setattr(protection, "client_from_settings", lambda settings: FakeClient())
    service.bind_intent(intent)
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "load_identities", lambda: [object()])
    monkeypatch.setattr(
        protection,
        "write_recovery_kit",
        lambda store_id, **kwargs: Path("kit"),
    )
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )
    upload_called = False

    def fail_new_upload(operation_id, **kwargs):
        nonlocal upload_called
        upload_called = True
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.BACKUP,
            ProtectionOperationPhase.FAILED,
            ProtectionState.ATTENTION_REQUIRED,
            ProtectionReasonCode.UPLOAD_FAILED,
        )

    monkeypatch.setattr(service, "_backup_now", fail_new_upload)
    monkeypatch.setattr(
        service,
        "_verify_now",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an old snapshot must not be verified for the new intent")
        ),
    )

    result = service.enable(intent)

    assert upload_called is True
    assert result.reason_code is ProtectionReasonCode.UPLOAD_FAILED
    assert load_state(store_id).pending_protection_snapshot_id is None


def test_canceled_first_enrollment_can_still_create_its_key_on_retry(tmp_path):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    first = service.create_intent()
    service.cancel_intent(first.protection_intent_id)

    second = service.create_intent()

    assert second.protection_intent_id != first.protection_intent_id
    durable = load_state(_store_id(settings))
    assert durable.pending_protection_created_store is True


def test_bind_uses_live_account_and_is_same_account_idempotent(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    client = FakeClient()
    monkeypatch.setattr(protection, "client_from_settings", lambda settings: client)

    first = service.bind_intent(intent)
    second = service.bind_intent(intent)

    assert first.reason_code is None
    assert second.reason_code is None
    assert client.calls == 2
    durable = load_state(_store_id(settings))
    assert durable.pending_protection_account_id == ACCOUNT_ID
    assert durable.pending_protection_status is ProtectionIntentStatus.READY


def test_bind_requires_subscription_without_canceling_intent(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: FakeClient(backup=False),
    )

    result = service.bind_intent(intent)

    assert result.phase is ProtectionOperationPhase.PENDING
    assert result.state is ProtectionState.SUBSCRIPTION_REQUIRED
    assert result.reason_code is ProtectionReasonCode.SUBSCRIPTION_REQUIRED
    durable = load_state(_store_id(settings))
    assert durable.pending_protection_status is ProtectionIntentStatus.ACCOUNT_BOUND


def test_rebinding_while_checkout_is_pending_preserves_checkout_status(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_status=ProtectionIntentStatus.CHECKOUT_PENDING,
        protection_state=ProtectionState.SUBSCRIPTION_REQUIRED,
    )
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: FakeClient(backup=False),
    )

    service.bind_intent(intent)

    assert (
        load_state(store_id).pending_protection_status
        is ProtectionIntentStatus.CHECKOUT_PENDING
    )


def test_cloud_service_failure_is_not_misreported_as_missing_subscription(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id

    class BrokenClient:
        def get_entitlements(self):
            raise CloudError("service unavailable", status_code=503)

        def close(self):
            pass

    monkeypatch.setattr(protection, "client_from_settings", lambda settings: BrokenClient())

    result = service.enable(intent)

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.OPERATION_INTERRUPTED
    assert result.state is ProtectionState.ATTENTION_REQUIRED


def test_different_live_account_cancels_bound_intent(tmp_path, monkeypatch):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: FakeClient(OTHER_ACCOUNT_ID),
    )

    result = service.bind_intent(intent)

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.ACCOUNT_CHANGED
    durable = load_state(_store_id(settings))
    assert durable.pending_protection_status is ProtectionIntentStatus.CANCELED
    assert durable.pending_protection_account_id == ACCOUNT_ID


def test_enable_fails_closed_on_missing_key_for_preexisting_store(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store_id = str(uuid.uuid4())
    (settings.memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    monkeypatch.setattr(protection, "client_from_settings", lambda settings: FakeClient())
    monkeypatch.setattr(protection, "key_file_exists", lambda: False)
    monkeypatch.setattr(
        protection,
        "init_key",
        lambda: (_ for _ in ()).throw(AssertionError("must not create a replacement key")),
    )

    result = service.enable(intent)

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.KEY_MISSING
    assert settings.cloud_backup_enabled is False
    durable = load_state(store_id)
    assert durable.pending_protection_status is ProtectionIntentStatus.RUNNING
    assert durable.protection_state is ProtectionState.ATTENTION_REQUIRED


def test_enable_only_claims_protected_after_exact_snapshot_verification(
    tmp_path, monkeypatch
):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    order: list[str] = []
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)

    def load_key():
        assert load_state(store_id).protection_state is ProtectionState.INITIALIZING
        return [object()]

    monkeypatch.setattr(protection, "load_identities", load_key)
    kit_args = []

    def write_kit(store_id, **kwargs):
        kit_args.append((store_id, kwargs))
        return Path("kit")

    monkeypatch.setattr(protection, "write_recovery_kit", write_kit)
    monkeypatch.setattr(
        protection,
        "RecoveryKitService",
        lambda settings: SimpleNamespace(
            validate_canonical_kit=lambda: order.append("kit")
        ),
    )

    def enable_setting(settings, enabled):
        assert load_state(store_id).protection_state is ProtectionState.INITIALIZING
        order.append("settings")
        settings.cloud_backup_enabled = enabled

    def upload(
        operation_id,
        *,
        reason,
        only_if_due,
        entitlement_verified,
        protection_intent_id,
    ):
        assert reason == "protection-enable"
        assert entitlement_verified is True
        assert protection_intent_id == intent
        assert load_state(store_id).protection_state is ProtectionState.UPLOADING_FIRST_BACKUP
        assert (
            load_state(store_id).last_operation_phase
            is ProtectionOperationPhase.UPLOADING
        )
        order.append("upload")
        update_state(
            store_id,
            memory_dir=settings.memory_dir,
            last_successful_backup_snapshot_id=SNAPSHOT_ID,
            pending_protection_snapshot_id=SNAPSHOT_ID,
            protection_state=ProtectionState.VERIFYING_FIRST_BACKUP,
        )
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.BACKUP,
            ProtectionOperationPhase.COMPLETED,
            ProtectionState.VERIFYING_FIRST_BACKUP,
            snapshot_id=SNAPSHOT_ID,
        )

    def verify(operation_id, *, requested_snapshot_id):
        assert requested_snapshot_id == SNAPSHOT_ID
        assert load_state(store_id).protection_state is not ProtectionState.PROTECTED
        assert (
            load_state(store_id).last_operation_phase
            is ProtectionOperationPhase.VERIFYING
        )
        order.append("verify")
        update_state(
            store_id,
            memory_dir=settings.memory_dir,
            last_verify_ok=True,
            last_verify_snapshot_id=SNAPSHOT_ID,
            last_verified_snapshot_id=SNAPSHOT_ID,
        )
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.VERIFY,
            ProtectionOperationPhase.COMPLETED,
            ProtectionState.VERIFYING_FIRST_BACKUP,
            snapshot_id=SNAPSHOT_ID,
        )

    monkeypatch.setattr(protection, "set_cloud_backup_enabled", enable_setting)
    monkeypatch.setattr(service, "_backup_now", upload)
    monkeypatch.setattr(service, "_verify_now", verify)

    result = service.enable(intent)

    assert order == ["kit", "settings", "upload", "verify"]
    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.operation_id == intent
    assert result.snapshot_id == SNAPSHOT_ID
    assert kit_args == [(store_id, {"account_email": "person@example.com"})]
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.PROTECTED
    assert durable.pending_protection_status is ProtectionIntentStatus.COMPLETED
    assert durable.recovery_kit_verified_at is None


def test_enable_fails_closed_when_generated_kit_does_not_validate(tmp_path, monkeypatch):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "load_identities", lambda: [object()])
    monkeypatch.setattr(
        protection,
        "write_recovery_kit",
        lambda store_id, **kwargs: Path("kit"),
    )
    monkeypatch.setattr(
        protection,
        "RecoveryKitService",
        lambda settings: SimpleNamespace(
            validate_canonical_kit=lambda: (_ for _ in ()).throw(
                RecoveryKitError("invalid kit")
            )
        ),
    )
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not enable")),
    )

    result = service.enable(intent)

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.KEY_INVALID
    assert settings.cloud_backup_enabled is False


def test_running_intent_retries_verification_without_reupload(tmp_path, monkeypatch):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_status=ProtectionIntentStatus.RUNNING,
        protection_state=ProtectionState.ATTENTION_REQUIRED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        pending_protection_snapshot_id=SNAPSHOT_ID,
        last_verify_snapshot_id=SNAPSHOT_ID,
        last_operation_kind=ProtectionOperationKind.VERIFY,
    )
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "load_identities", lambda: [object()])
    monkeypatch.setattr(
        protection,
        "write_recovery_kit",
        lambda store_id, **kwargs: Path("kit"),
    )
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )
    monkeypatch.setattr(
        service,
        "_backup_now",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not reupload")),
    )

    def verify(operation_id, *, requested_snapshot_id):
        update_state(
            store_id,
            memory_dir=settings.memory_dir,
            last_verify_ok=True,
            last_verified_snapshot_id=requested_snapshot_id,
        )
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.VERIFY,
            ProtectionOperationPhase.COMPLETED,
            ProtectionState.ATTENTION_REQUIRED,
            snapshot_id=requested_snapshot_id,
        )

    monkeypatch.setattr(service, "_verify_now", verify)

    result = service.enable(intent)

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert load_state(store_id).pending_protection_status is ProtectionIntentStatus.COMPLETED


def test_verification_failure_keeps_committed_snapshot_but_never_protected(
    tmp_path, monkeypatch
):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "load_identities", lambda: [object()])
    monkeypatch.setattr(
        protection,
        "write_recovery_kit",
        lambda store_id, **kwargs: Path("kit"),
    )
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )

    def upload(operation_id, **kwargs):
        update_state(
            store_id,
            memory_dir=settings.memory_dir,
            last_successful_backup_snapshot_id=SNAPSHOT_ID,
            pending_protection_snapshot_id=SNAPSHOT_ID,
            protection_state=ProtectionState.VERIFYING_FIRST_BACKUP,
        )
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.BACKUP,
            ProtectionOperationPhase.COMPLETED,
            ProtectionState.VERIFYING_FIRST_BACKUP,
            snapshot_id=SNAPSHOT_ID,
        )

    def fail_verification(operation_id, *, requested_snapshot_id):
        return ProtectionOperation(
            operation_id,
            ProtectionOperationKind.VERIFY,
            ProtectionOperationPhase.FAILED,
            ProtectionState.ATTENTION_REQUIRED,
            ProtectionReasonCode.VERIFICATION_FAILED,
            "scratch rebuild failed",
            requested_snapshot_id,
        )

    monkeypatch.setattr(service, "_backup_now", upload)
    monkeypatch.setattr(service, "_verify_now", fail_verification)

    result = service.enable(intent)

    durable = load_state(store_id)
    assert result.kind is ProtectionOperationKind.ENABLE
    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.operation_id == intent
    assert durable.pending_protection_status is ProtectionIntentStatus.RUNNING
    assert durable.last_successful_backup_snapshot_id == SNAPSHOT_ID
    assert durable.protection_state is not ProtectionState.PROTECTED
    assert durable.last_operation_kind is ProtectionOperationKind.ENABLE
    assert durable.last_operation_phase is ProtectionOperationPhase.FAILED


def test_interruption_before_upload_reservation_resumes_cleanly(tmp_path, monkeypatch):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_status=ProtectionIntentStatus.RUNNING,
        protection_state=ProtectionState.UPLOADING_FIRST_BACKUP,
        last_operation_id=intent,
        last_operation_kind=ProtectionOperationKind.ENABLE,
        last_operation_phase=ProtectionOperationPhase.UPLOADING,
    )
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "load_identities", lambda: [object()])
    monkeypatch.setattr(
        protection,
        "write_recovery_kit",
        lambda store_id, **kwargs: Path("kit"),
    )
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )
    calls = []

    def resumed_backup(*args, **kwargs):
        calls.append(kwargs)
        return ProtectionOperation(
            intent,
            ProtectionOperationKind.BACKUP,
            ProtectionOperationPhase.FAILED,
            ProtectionState.ATTENTION_REQUIRED,
            ProtectionReasonCode.UPLOAD_FAILED,
            "probe stopped after proving retry",
        )

    monkeypatch.setattr(service, "_backup_now", resumed_backup)

    result = service.enable(intent)

    assert result.reason_code is ProtectionReasonCode.UPLOAD_FAILED
    assert len(calls) == 1
    assert load_state(store_id).pending_protection_status is ProtectionIntentStatus.RUNNING


def test_finalizing_upload_journal_is_the_only_ambiguous_marker(tmp_path):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_upload_id="upload-id",
        pending_upload_snapshot_id=SNAPSHOT_ID,
        pending_upload_operation_id=intent,
        pending_upload_protection_intent_id=intent,
        pending_upload_phase=UploadJournalPhase.FINALIZING,
    )

    assert service._upload_status_is_unknown(load_state(store_id)) is True


def test_second_store_cannot_replace_canonical_recovery_kit(tmp_path, monkeypatch):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    other_store_id = str(uuid.uuid4())
    protection.RECOVERY_KIT_PATH.write_text(
        f"# Ormah Recovery Kit\nstore_id: {other_store_id}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "load_identities", lambda: [object()])
    monkeypatch.setattr(
        protection,
        "write_recovery_kit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("canonical kit must not be overwritten")
        ),
    )

    result = service.enable(intent)

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.KEY_INVALID
    assert other_store_id in protection.RECOVERY_KIT_PATH.read_text(encoding="utf-8")


def test_disable_persists_stopped_before_setting_and_preserves_history(
    tmp_path, monkeypatch
):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    verified_at = NOW - timedelta(days=1)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        protection_state=ProtectionState.PROTECTED,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_successful_verify_at=verified_at,
    )
    settings.cloud_backup_enabled = True

    def disable_setting(settings, enabled):
        durable = load_state(store_id)
        assert durable.protection_state is ProtectionState.STOPPED
        assert durable.pending_protection_account_id == ACCOUNT_ID
        assert durable.last_verified_snapshot_id == SNAPSHOT_ID
        settings.cloud_backup_enabled = enabled

    monkeypatch.setattr(protection, "set_cloud_backup_enabled", disable_setting)

    first = service.disable()
    disabled_at = load_state(store_id).protection_disabled_at
    second = service.disable()

    assert first.phase is ProtectionOperationPhase.COMPLETED
    assert second.phase is ProtectionOperationPhase.COMPLETED
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.STOPPED
    assert durable.protection_disabled_at == disabled_at
    assert durable.last_successful_verify_at == verified_at
    assert durable.pending_protection_intent_id == intent


def test_disable_is_idempotent_for_legacy_enrolled_store(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    store_id = protection.get_or_create_store_id(settings.memory_dir)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )
    service = CloudProtectionService(settings)

    first = service.disable()
    second = service.disable()

    assert first.state is ProtectionState.STOPPED
    assert second.state is ProtectionState.STOPPED
    assert load_state(store_id).protection_state is ProtectionState.STOPPED


def test_expired_protected_store_gets_reactivation_intent_and_preserves_origin(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    original = service.create_intent().protection_intent_id
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: protection.EntitlementStatus.EXPIRED,
    )

    replacement = service.create_intent()

    assert replacement.protection_intent_id != original
    assert replacement.state is ProtectionState.PAUSED
    durable = load_state(store_id)
    assert durable.pending_protection_origin_state is ProtectionState.PROTECTED

    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: FakeClient(backup=False),
    )
    bound = service.bind_intent(replacement.protection_intent_id)
    assert bound.state is ProtectionState.SUBSCRIPTION_REQUIRED
    assert (
        load_state(store_id).pending_protection_status
        is ProtectionIntentStatus.ACCOUNT_BOUND
    )

    canceled = service.cancel_intent(replacement.protection_intent_id)
    assert canceled.state is ProtectionState.PROTECTED
    assert load_state(store_id).protection_state is ProtectionState.PROTECTED
    assert cloud_status_payload(settings, entitlement="active")["protection_state"] == (
        ProtectionState.PROTECTED.value
    )


@pytest.mark.parametrize(
    "terminal_status",
    [ProtectionIntentStatus.CANCELED, ProtectionIntentStatus.EXPIRED],
)
def test_terminal_intent_does_not_invalidate_verified_protection(
    tmp_path, terminal_status
):
    settings = _settings(tmp_path)
    store_id = protection.get_or_create_store_id(settings.memory_dir)
    settings.cloud_backup_enabled = True
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_intent_id=str(uuid.uuid4()),
        pending_protection_status=terminal_status,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )

    payload = cloud_status_payload(settings, entitlement="active")

    assert payload["protection_state"] == ProtectionState.PROTECTED.value
    assert not any("metadata is incomplete" in warning for warning in payload["warnings"])


@pytest.mark.parametrize(
    "active_or_unknown_status",
    [
        ProtectionIntentStatus.PENDING,
        ProtectionIntentStatus.ACCOUNT_BOUND,
        ProtectionIntentStatus.CHECKOUT_PENDING,
        ProtectionIntentStatus.READY,
        ProtectionIntentStatus.RUNNING,
        "future_intent_status",
    ],
)
def test_active_or_unknown_intent_blocks_verified_protection_claim(
    tmp_path, active_or_unknown_status
):
    settings = _settings(tmp_path)
    store_id = protection.get_or_create_store_id(settings.memory_dir)
    settings.cloud_backup_enabled = True
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_intent_id=str(uuid.uuid4()),
        pending_protection_status=active_or_unknown_status,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )

    payload = cloud_status_payload(settings, entitlement="active")

    assert payload["protection_state"] == ProtectionState.ATTENTION_REQUIRED.value
    assert any("metadata is incomplete" in warning for warning in payload["warnings"])


def test_replacing_expired_subscription_intent_preserves_protected_origin(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    service.create_intent()
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        last_verify_ok=True,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: protection.EntitlementStatus.EXPIRED,
    )
    first = service.create_intent()
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: FakeClient(backup=False),
    )
    service.bind_intent(first.protection_intent_id)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_expires_at=NOW - timedelta(seconds=1),
    )

    replacement = service.create_intent()

    assert replacement.protection_intent_id != first.protection_intent_id
    assert (
        load_state(store_id).pending_protection_origin_state
        is ProtectionState.PROTECTED
    )
    canceled = service.cancel_intent(replacement.protection_intent_id)
    assert canceled.state is ProtectionState.PROTECTED


def test_disable_before_enrollment_remains_local_only(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )

    result = CloudProtectionService(settings).disable()

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.LOCAL_ONLY
    assert not (settings.memory_dir / ".store_id").exists()


def test_disable_pending_intent_cancels_without_claiming_previously_protected(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    service.create_intent()
    store_id = _store_id(settings)
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: setattr(settings, "cloud_backup_enabled", enabled),
    )

    result = service.disable()

    durable = load_state(store_id)
    assert result.state is ProtectionState.LOCAL_ONLY
    assert durable.protection_state is ProtectionState.LOCAL_ONLY
    assert durable.pending_protection_status is ProtectionIntentStatus.CANCELED
    assert durable.protection_disabled_at is None


def test_disable_setting_failure_preserves_fail_closed_stopped_state(
    tmp_path, monkeypatch
):
    settings, service, _, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
        pending_protection_status=ProtectionIntentStatus.COMPLETED,
    )
    settings.cloud_backup_enabled = True
    monkeypatch.setattr(
        protection,
        "set_cloud_backup_enabled",
        lambda settings, enabled: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = service.disable()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.state is ProtectionState.STOPPED
    assert load_state(store_id).protection_state is ProtectionState.STOPPED


def test_running_intent_does_not_expire_or_allow_pending_cancel(tmp_path, monkeypatch):
    settings, service, intent, _ = _ready_intent(tmp_path, monkeypatch)
    store_id = _store_id(settings)
    update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_protection_status=ProtectionIntentStatus.RUNNING,
        pending_protection_expires_at=NOW - timedelta(seconds=1),
    )

    rebound = service.bind_intent(intent)
    canceled = service.cancel_intent(intent)

    assert rebound.reason_code is None
    assert canceled.phase is ProtectionOperationPhase.FAILED
    assert canceled.reason_code is ProtectionReasonCode.OPERATION_INTERRUPTED
    assert load_state(store_id).pending_protection_status is ProtectionIntentStatus.RUNNING


@pytest.mark.parametrize("method_name", ["bind_intent", "cancel_intent", "enable"])
def test_invalid_intent_does_not_mutate_current_store_health(
    tmp_path, method_name
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    valid = service.create_intent()
    store_id = _store_id(settings)
    before = load_state(store_id)

    result = getattr(service, method_name)("not-an-intent")

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.INTENT_CANCELED
    assert load_state(store_id) == before
    assert valid.protection_intent_id == before.pending_protection_intent_id


@pytest.mark.parametrize("method_name", ["bind_intent", "cancel_intent", "enable"])
def test_intent_methods_report_client_update_required_for_newer_state(
    tmp_path, method_name
):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    intent = service.create_intent().protection_intent_id
    store_id = _store_id(settings)
    path = state.state_path(store_id)
    payload = load_state(store_id).to_dict()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    result = getattr(service, method_name)(intent)

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.CLIENT_UPDATE_REQUIRED
    assert path.read_bytes() == before


def test_disable_reports_client_update_required_for_newer_state(tmp_path):
    settings = _settings(tmp_path)
    service = CloudProtectionService(settings)
    service.create_intent()
    store_id = _store_id(settings)
    path = state.state_path(store_id)
    payload = load_state(store_id).to_dict()
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    result = service.disable()

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.CLIENT_UPDATE_REQUIRED
    assert path.read_bytes() == before


def test_active_intent_blocks_legacy_verification_promotion():
    active = state.CloudState(
        protection_state=ProtectionState.VERIFYING_FIRST_BACKUP,
        pending_protection_intent_id=str(uuid.uuid4()),
        pending_protection_status=ProtectionIntentStatus.RUNNING,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
    )
    legacy = state.CloudState(
        protection_state=ProtectionState.VERIFYING_FIRST_BACKUP,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
    )

    assert protection._state_after_verification(active, SNAPSHOT_ID) is (
        ProtectionState.VERIFYING_FIRST_BACKUP
    )
    assert protection._state_after_verification(legacy, SNAPSHOT_ID) is ProtectionState.PROTECTED
