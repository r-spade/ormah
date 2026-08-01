from __future__ import annotations

import errno
import logging
import threading
from types import SimpleNamespace

import httpx
import pytest

from ormah.cloud import jobs, protection, state
from ormah.cloud.entitlements import EntitlementStatus
from ormah.cloud.protection import CloudProtectionService
from ormah.cloud.state import (
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    ProtectionReasonCode,
    ProtectionState,
    UploadJournalPhase,
    load_state,
    state_path,
)
from ormah.cloud.client import CloudError
from ormah.cloud.store_lock import StoreLock, StoreLockTimeout
from tests.test_cloud_jobs import (
    FakeCloudClient,
    SNAPSHOT_ID,
    _patch_upload_prerequisites,
    _patch_verification,
    _settings,
    _verification_bundle,
)


@pytest.fixture
def cloud_state_dir(tmp_path, monkeypatch):
    path = tmp_path / "cloud-state"
    monkeypatch.setattr(state, "CLOUD_STATE_DIR", path)
    return path


def test_backup_now_returns_typed_completed_operation(tmp_path, monkeypatch, cloud_state_dir):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_upload_at=protection._utc_now(),
        protection_state=ProtectionState.PROTECTED,
    )
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)

    result = CloudProtectionService(settings).backup_now()

    assert isinstance(result, ProtectionOperation)
    assert result.kind is ProtectionOperationKind.BACKUP
    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.VERIFICATION_PENDING
    assert result.snapshot_id == SNAPSHOT_ID
    assert client.finalized == [(store_id, "upload-1")]
    durable = load_state(store_id)
    assert durable.last_operation_id == result.operation_id
    assert durable.last_operation_phase is ProtectionOperationPhase.COMPLETED
    assert durable.pending_upload_phase is None


def test_backup_records_truthful_processing_phases(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, _store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    observed = []
    real_update = state.update_state

    def capture_update(*args, **kwargs):
        phase = kwargs.get("last_operation_phase")
        if phase is not None:
            observed.append(phase)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(protection, "update_state", capture_update)

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert observed == [
        ProtectionOperationPhase.PREPARING,
        ProtectionOperationPhase.ENCRYPTING,
        ProtectionOperationPhase.UPLOADING,
        ProtectionOperationPhase.FINALIZING,
        ProtectionOperationPhase.COMPLETED,
    ]


def test_backup_and_verify_checks_the_exact_uploaded_snapshot(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, _store_id = _settings(tmp_path)
    service = CloudProtectionService(settings)
    calls = []
    backup = ProtectionOperation(
        "backup-op",
        ProtectionOperationKind.BACKUP,
        ProtectionOperationPhase.COMPLETED,
        ProtectionState.VERIFICATION_PENDING,
        snapshot_id=SNAPSHOT_ID,
    )
    verified = ProtectionOperation(
        "backup-op",
        ProtectionOperationKind.VERIFY,
        ProtectionOperationPhase.COMPLETED,
        ProtectionState.PROTECTED,
        snapshot_id=SNAPSHOT_ID,
    )

    monkeypatch.setattr(
        service,
        "_backup_now",
        lambda operation_id, **kwargs: calls.append(("backup", operation_id, kwargs)) or backup,
    )
    monkeypatch.setattr(
        service,
        "_verify_now",
        lambda operation_id, **kwargs: calls.append(("verify", operation_id, kwargs)) or verified,
    )

    result = service.backup_and_verify()

    assert result.kind is ProtectionOperationKind.BACKUP
    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.PROTECTED
    assert result.snapshot_id == SNAPSHOT_ID
    assert calls[0][0] == "backup"
    assert calls[1] == (
        "verify",
        calls[0][1],
        {"requested_snapshot_id": SNAPSHOT_ID},
    )


def test_legacy_opted_in_store_reaches_protected_after_first_backup_and_verification(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    upload_client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, upload_client)

    backup = CloudProtectionService(settings).backup_now()

    assert backup.state is ProtectionState.VERIFYING_FIRST_BACKUP
    assert load_state(store_id).protection_state is ProtectionState.VERIFYING_FIRST_BACKUP

    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    verification_client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, verification_client, bundle, identity)

    verification = CloudProtectionService(settings).verify_now()

    assert verification.phase is ProtectionOperationPhase.COMPLETED
    assert verification.state is ProtectionState.PROTECTED
    assert load_state(store_id).protection_state is ProtectionState.PROTECTED


def test_backup_now_passes_the_operation_reason_into_the_bundle(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, _store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    captured = {}

    def capture_bundle(backup_dir, out_path, recipients, **kwargs):
        captured.update(kwargs)
        out_path.write_bytes(b"age-encrypted-bundle")
        return out_path

    monkeypatch.setattr(protection, "build_bundle", capture_bundle)

    result = CloudProtectionService(settings).backup_now(reason="manual-ui")

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert captured["reason"] == "manual-ui"


def test_failed_put_returns_typed_failure_and_never_finalizes(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, _store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    monkeypatch.setattr(
        protection,
        "put_file",
        lambda url, path, headers: (_ for _ in ()).throw(OSError("PUT failed")),
    )

    result = CloudProtectionService(settings).backup_now()

    assert isinstance(result, ProtectionOperation)
    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.UPLOAD_FAILED
    assert client.finalized == []


def test_quota_and_disk_failures_have_actionable_reason_codes(
    tmp_path, monkeypatch, cloud_state_dir
):
    quota_settings, _ = _settings(tmp_path / "quota")
    quota_client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, quota_client)
    quota_client.create_upload = lambda *args: (_ for _ in ()).throw(
        CloudError(
            "Account storage quota exceeded.",
            status_code=413,
            payload={"detail": "Account storage quota exceeded."},
        )
    )

    quota = CloudProtectionService(quota_settings).backup_now()

    disk_settings, _ = _settings(tmp_path / "disk")
    disk_client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, disk_client)
    monkeypatch.setattr(
        protection,
        "build_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.ENOSPC, "No space left on device")
        ),
    )

    disk = CloudProtectionService(disk_settings).backup_now()

    assert quota.reason_code is ProtectionReasonCode.QUOTA_EXCEEDED
    assert disk.reason_code is ProtectionReasonCode.DISK_SPACE_INSUFFICIENT


def test_uncertain_finalize_retries_the_same_upload_without_new_reservation(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    real_finalize = client.finalize_upload

    def timeout_after_commit(store_id, upload_id):
        real_finalize(store_id, upload_id)
        raise CloudError("finalize response lost")

    client.finalize_upload = timeout_after_commit

    first = CloudProtectionService(settings).backup_now()
    client.finalize_upload = real_finalize
    monkeypatch.setattr(
        client,
        "create_upload",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unknown finalize must block a new upload")
        ),
    )
    second = CloudProtectionService(settings).backup_now()

    assert first.reason_code is ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN
    assert second.phase is ProtectionOperationPhase.COMPLETED
    assert client.finalized == [(store_id, "upload-1"), (store_id, "upload-1")]
    durable = load_state(store_id)
    assert durable.last_error_code is None
    assert durable.pending_upload_phase is None


def test_definitively_expired_finalize_is_replaced_after_local_expiry(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_upload_id="expired-upload",
        pending_upload_snapshot_id="01J11111111111111111111111",
        pending_upload_operation_id="expired-operation",
        pending_upload_phase=UploadJournalPhase.FINALIZING,
        pending_upload_expires_at=protection._utc_now() - protection.timedelta(seconds=1),
        last_error_code=ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN,
    )
    real_finalize = client.finalize_upload

    def finalize(store_id, upload_id):
        if upload_id == "expired-upload":
            raise CloudError(
                "Upload expired.",
                status_code=409,
                payload={"detail": {"code": "upload_expired"}},
            )
        return real_finalize(store_id, upload_id)

    client.finalize_upload = finalize

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert len(client.created) == 1
    assert client.finalized == [(store_id, "upload-1")]
    durable = load_state(store_id)
    assert durable.pending_upload_phase is None
    assert durable.last_error_code is None


def test_server_expiry_before_local_expiry_keeps_ambiguous_upload(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_upload_id="ambiguous-upload",
        pending_upload_snapshot_id="01J11111111111111111111111",
        pending_upload_operation_id="ambiguous-operation",
        pending_upload_phase=UploadJournalPhase.FINALIZING,
        pending_upload_expires_at=protection._utc_now() + protection.timedelta(minutes=1),
    )
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    client.create_upload = lambda *args: (_ for _ in ()).throw(
        AssertionError("an ambiguous upload must not be replaced")
    )
    client.finalize_upload = lambda *args: (_ for _ in ()).throw(
        CloudError(
            "Upload expired.",
            status_code=409,
            payload={"detail": {"code": "upload_expired"}},
        )
    )

    result = CloudProtectionService(settings).backup_now()

    assert result.reason_code is ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN
    assert load_state(store_id).pending_upload_phase is UploadJournalPhase.FINALIZING


def test_reserved_upload_from_interrupted_put_is_safe_to_replace(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        pending_upload_id="abandoned-upload",
        pending_upload_snapshot_id="01J11111111111111111111111",
        pending_upload_operation_id="abandoned-operation",
        pending_upload_phase=UploadJournalPhase.RESERVED,
        pending_upload_expires_at=protection._utc_now() + protection.timedelta(minutes=10),
    )
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert len(client.created) == 1
    assert client.finalized == [(store_id, "upload-1")]
    durable = load_state(store_id)
    assert durable.pending_upload_phase is None
    assert durable.last_successful_backup_snapshot_id == SNAPSHOT_ID


def test_successful_verification_does_not_clear_ambiguous_upload_journal(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.ATTENTION_REQUIRED,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        last_error_code=ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN,
        last_error_message="finalize response lost",
        pending_upload_id="upload-1",
        pending_upload_snapshot_id=SNAPSHOT_ID,
        pending_upload_operation_id="operation-1",
        pending_upload_phase=UploadJournalPhase.FINALIZING,
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)

    result = CloudProtectionService(settings).verify_now(SNAPSHOT_ID)

    assert result.phase is ProtectionOperationPhase.COMPLETED
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.ATTENTION_REQUIRED
    assert durable.last_error_code is ProtectionReasonCode.UPLOAD_STATUS_UNKNOWN
    assert durable.pending_upload_phase is UploadJournalPhase.FINALIZING


def test_upload_rejects_invalid_server_snapshot_id_before_put(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, _store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    original_create = client.create_upload

    def invalid_snapshot(*args):
        response = original_create(*args)
        response["snapshot_id"] = "../../escape"
        return response

    client.create_upload = invalid_snapshot
    monkeypatch.setattr(
        protection,
        "put_file",
        lambda *args: (_ for _ in ()).throw(AssertionError("PUT should not run")),
    )

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.UPLOAD_FAILED
    assert client.finalized == []


def test_verification_rejects_invalid_server_snapshot_id_before_tempfile(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, _store_id = _settings(tmp_path)
    client = FakeCloudClient()
    client.list_blobs = lambda store_id: {
        "blobs": [{"snapshot_id": "../../escape", "size_bytes": 1}]
    }
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "client_from_settings", lambda settings: client)
    monkeypatch.setattr(
        protection.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tempfile should not be created")
        ),
    )

    result = CloudProtectionService(settings).verify_now()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.VERIFICATION_FAILED


def test_index_rebuild_failure_is_not_reported_as_backup_corruption(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        protection_state=ProtectionState.VERIFICATION_PENDING,
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    monkeypatch.setattr(
        protection,
        "IndexBuilder",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("embedding model is unavailable")
        ),
    )

    result = CloudProtectionService(settings).verify_now(SNAPSHOT_ID)

    assert result.phase is ProtectionOperationPhase.FAILED
    assert (
        result.reason_code
        is ProtectionReasonCode.INDEX_ENVIRONMENT_UNAVAILABLE
    )


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("download", ProtectionReasonCode.DOWNLOAD_FAILED),
        ("ciphertext", ProtectionReasonCode.CIPHERTEXT_HASH_MISMATCH),
        ("decrypt", ProtectionReasonCode.DECRYPT_FAILED),
        ("manifest", ProtectionReasonCode.MANIFEST_VERIFICATION_FAILED),
        ("node", ProtectionReasonCode.NODE_PARSE_FAILED),
        ("search", ProtectionReasonCode.SEARCH_PROBE_FAILED),
    ],
)
def test_verification_reports_the_failed_restore_stage(
    tmp_path, monkeypatch, cloud_state_dir, failure, reason_code
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        protection_state=ProtectionState.VERIFICATION_PENDING,
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    secret = "private-frontmatter-must-not-escape"

    if failure == "download":
        monkeypatch.setattr(
            protection,
            "download_file",
            lambda *args: (_ for _ in ()).throw(RuntimeError("object unavailable")),
        )
    elif failure == "ciphertext":
        client.list_blobs = lambda store_id: {
            "blobs": [
                {
                    "snapshot_id": SNAPSHOT_ID,
                    "size_bytes": bundle.stat().st_size,
                    "sha256": "0" * 64,
                }
            ]
        }
    elif failure == "decrypt":
        monkeypatch.setattr(
            protection,
            "open_bundle",
            lambda *args: (_ for _ in ()).throw(
                protection.CloudCryptoError("wrong recovery key")
            ),
        )
    elif failure == "manifest":
        monkeypatch.setattr(
            protection,
            "open_bundle",
            lambda *args: (_ for _ in ()).throw(
                TypeError(f"nodes\\{secret}.md manifest entry is malformed")
            ),
        )
    elif failure == "node":
        monkeypatch.setattr(
            protection,
            "parse_node",
            lambda *args: (_ for _ in ()).throw(ValueError(secret)),
        )
    else:
        monkeypatch.setattr(
            protection,
            "_probe_search",
            lambda *args: (_ for _ in ()).throw(RuntimeError("probe failed")),
        )

    result = CloudProtectionService(settings).verify_now(SNAPSHOT_ID)

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is reason_code
    durable = load_state(store_id)
    assert durable.last_error_code is reason_code
    assert secret not in (durable.last_verify_error or "")


def test_verification_distinguishes_old_service_from_unverified_legacy_blob(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    client.list_blobs = lambda store_id: {
        "blobs": [{"snapshot_id": SNAPSHOT_ID, "size_bytes": bundle.stat().st_size}]
    }

    old_service = CloudProtectionService(settings).verify_now()
    client.list_blobs = lambda store_id: {
        "blobs": [
            {
                "snapshot_id": SNAPSHOT_ID,
                "size_bytes": bundle.stat().st_size,
                "sha256": None,
            }
        ]
    }
    unverified_legacy = CloudProtectionService(settings).verify_now()

    assert old_service.reason_code is ProtectionReasonCode.SERVICE_UPDATE_REQUIRED
    assert (
        unverified_legacy.reason_code
        is ProtectionReasonCode.CIPHERTEXT_HASH_UNAVAILABLE
    )


def test_verification_reports_local_processing_limit_as_local_failure(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    client.processing_limit = lambda **kwargs: 0

    result = CloudProtectionService(settings).verify_now()

    assert result.reason_code is ProtectionReasonCode.PROCESSING_LIMIT_EXCEEDED
    assert result.state is ProtectionState.ATTENTION_REQUIRED
    assert load_state(store_id).last_verify_ok is False


def test_safe_error_message_redacts_plaintext_node_slugs():
    message = protection.safe_error_message(
        "Hash mismatch for 'nodes/fact_my-therapist-said-secret_a1b2.md' "
        "and deleted/event_private-diagnosis_c3d4.md"
    )

    assert "therapist" not in message
    assert "diagnosis" not in message
    assert message == (
        "Hash mismatch for 'nodes/<redacted>.md' and deleted/<redacted>.md"
    )


def test_safe_error_message_redacts_windows_node_paths():
    message = protection.safe_error_message(
        r"Hash mismatch for nodes\fact_private-diagnosis_a1b2.md"
    )

    assert message == "Hash mismatch for nodes/<redacted>.md"


def test_persisted_error_keeps_nonsecret_path_for_cli_diagnostics():
    message = protection.safe_error_message(
        "Could not write /home/person/.local/share/ormah/cloud/state.json"
    )

    assert "/home/person/.local/share/ormah/cloud/state.json" in message


def test_product_error_redacts_nonsecret_local_path():
    message = protection.safe_product_error_message(
        "Could not write /home/person/.local/share/ormah/cloud/state.json"
    )

    assert message == "Could not write <redacted-path>"


def test_offline_upload_preserves_verification_health_and_redacts_persisted_error(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    verified_at = protection._utc_now()
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_verify_at=verified_at,
        last_verify_ok=True,
        last_verify_snapshot_id="01J11111111111111111111111",
        last_successful_verify_at=verified_at,
        last_verified_snapshot_id="01J11111111111111111111111",
        protection_state=ProtectionState.PROTECTED,
    )
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    secret = "do-not-persist"
    client.create_upload = lambda *args: (_ for _ in ()).throw(
        CloudError(f"request failed X-Amz-Signature={secret}&token=also-secret")
    )

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.OFFLINE
    assert result.state is ProtectionState.OFFLINE
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.OFFLINE
    assert durable.last_verify_ok is True
    assert durable.last_verified_snapshot_id == "01J11111111111111111111111"
    assert secret not in durable.last_upload_error
    assert "also-secret" not in durable.last_upload_error
    assert durable.last_upload_error.count("<redacted>") == 2


def test_first_backup_offline_moves_local_only_store_to_offline(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    client.create_upload = lambda *args: (_ for _ in ()).throw(CloudError("offline"))

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.OFFLINE
    assert result.state is ProtectionState.OFFLINE
    assert load_state(store_id).protection_state is ProtectionState.OFFLINE


def test_first_backup_missing_key_moves_local_only_store_to_attention_required(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    monkeypatch.setattr(protection, "key_file_exists", lambda: False)

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.KEY_MISSING
    assert result.state is ProtectionState.ATTENTION_REQUIRED
    assert load_state(store_id).protection_state is ProtectionState.ATTENTION_REQUIRED


def test_offline_verification_preserves_known_good_verification_health(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    verified_at = protection._utc_now()
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_verify_at=verified_at,
        last_verify_ok=True,
        last_verify_snapshot_id=SNAPSHOT_ID,
        last_successful_verify_at=verified_at,
        last_verified_snapshot_id=SNAPSHOT_ID,
        protection_state=ProtectionState.PROTECTED,
    )
    client = FakeCloudClient()
    request = httpx.Request("GET", "https://cloud.example/blobs")
    client.list_blobs = lambda *args: (_ for _ in ()).throw(
        httpx.ConnectError("offline", request=request)
    )
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "client_from_settings", lambda settings: client)

    result = CloudProtectionService(settings).verify_now()

    assert result.phase is ProtectionOperationPhase.FAILED
    assert result.reason_code is ProtectionReasonCode.OFFLINE
    assert result.state is ProtectionState.OFFLINE
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.OFFLINE
    assert durable.last_verify_at == verified_at
    assert durable.last_verify_ok is True
    assert durable.last_verify_snapshot_id == SNAPSHOT_ID
    assert durable.last_successful_verify_at == verified_at
    assert durable.last_verified_snapshot_id == SNAPSHOT_ID


def test_verify_now_accepts_snapshot_id_after_entitlement_lapse(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    settings.cloud_backup_enabled = False
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_successful_backup_snapshot_id=SNAPSHOT_ID,
        protection_state=ProtectionState.CHANGES_PENDING,
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: (_ for _ in ()).throw(
            AssertionError("downloads must not check upload entitlement")
        ),
    )
    observed = []
    real_update = state.update_state

    def capture_update(*args, **kwargs):
        phase = kwargs.get("last_operation_phase")
        if phase is not None:
            observed.append(phase)
        return real_update(*args, **kwargs)

    monkeypatch.setattr(protection, "update_state", capture_update)

    result = CloudProtectionService(settings).verify_now(SNAPSHOT_ID)

    assert isinstance(result, ProtectionOperation)
    assert result.kind is ProtectionOperationKind.VERIFY
    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.PROTECTED
    assert result.snapshot_id == SNAPSHOT_ID
    assert load_state(store_id).last_verified_snapshot_id == SNAPSHOT_ID
    assert observed == [
        ProtectionOperationPhase.DOWNLOADING,
        ProtectionOperationPhase.VERIFYING,
        ProtectionOperationPhase.REBUILDING,
        ProtectionOperationPhase.COMPLETED,
    ]


def test_verifying_an_older_snapshot_does_not_claim_latest_backup_is_protected(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    previous_verified_at = protection._utc_now()
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_successful_backup_snapshot_id="01NEWER",
        last_verify_at=previous_verified_at,
        last_verify_ok=True,
        last_verify_snapshot_id="01NEWER",
        last_successful_verify_at=previous_verified_at,
        last_verified_snapshot_id="01NEWER",
        protection_state=ProtectionState.CHANGES_PENDING,
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)

    result = CloudProtectionService(settings).verify_now(SNAPSHOT_ID)

    assert result.phase is ProtectionOperationPhase.COMPLETED
    assert result.state is ProtectionState.CHANGES_PENDING
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.CHANGES_PENDING
    assert durable.last_verify_at == previous_verified_at
    assert durable.last_verify_ok is True
    assert durable.last_verify_snapshot_id == "01NEWER"
    assert durable.last_successful_verify_at == previous_verified_at
    assert durable.last_verified_snapshot_id == "01NEWER"


def test_failure_verifying_an_older_snapshot_preserves_latest_verification_health(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    previous_verified_at = protection._utc_now()
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_successful_backup_snapshot_id="01NEWER",
        last_verify_at=previous_verified_at,
        last_verify_ok=True,
        last_verify_snapshot_id="01NEWER",
        last_successful_verify_at=previous_verified_at,
        last_verified_snapshot_id="01NEWER",
        protection_state=ProtectionState.PROTECTED,
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    monkeypatch.setattr(
        protection,
        "open_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(protection.BundleError("corrupt")),
    )

    result = CloudProtectionService(settings).verify_now(SNAPSHOT_ID)

    assert result.phase is ProtectionOperationPhase.FAILED
    durable = load_state(store_id)
    assert durable.protection_state is ProtectionState.PROTECTED
    assert durable.last_verify_at == previous_verified_at
    assert durable.last_verify_ok is True
    assert durable.last_verify_snapshot_id == "01NEWER"
    assert durable.last_successful_verify_at == previous_verified_at
    assert durable.last_verified_snapshot_id == "01NEWER"


def test_expired_entitlement_pauses_only_direct_upload(tmp_path, monkeypatch, cloud_state_dir):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
    )
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(protection, "check_entitlement", lambda settings: EntitlementStatus.EXPIRED)
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: (_ for _ in ()).throw(AssertionError("upload should not start")),
    )

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.state is ProtectionState.PAUSED
    assert result.reason_code is ProtectionReasonCode.ENTITLEMENT_EXPIRED


def test_stopped_protection_blocks_direct_upload_before_entitlement(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.STOPPED,
    )
    monkeypatch.setattr(
        protection,
        "key_file_exists",
        lambda: (_ for _ in ()).throw(AssertionError("key guard should not run")),
    )
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: (_ for _ in ()).throw(AssertionError("entitlement should not run")),
    )

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.state is ProtectionState.STOPPED
    assert result.reason_code is ProtectionReasonCode.PROTECTION_STOPPED


def test_disabled_backup_reports_the_actual_durable_state(tmp_path, monkeypatch, cloud_state_dir):
    settings, store_id = _settings(tmp_path)
    settings.cloud_backup_enabled = False
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state=ProtectionState.PROTECTED,
    )
    monkeypatch.setattr(
        protection,
        "key_file_exists",
        lambda: (_ for _ in ()).throw(AssertionError("key guard should not run")),
    )

    result = CloudProtectionService(settings).backup_now()

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.NOT_ENABLED
    assert result.state is ProtectionState.PROTECTED


def test_empty_and_not_due_cancellations_have_stable_reason_codes(
    tmp_path, monkeypatch, cloud_state_dir
):
    empty_settings, _empty_store_id = _settings(tmp_path / "empty", with_node=False)
    empty_client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, empty_client)

    empty = CloudProtectionService(empty_settings).backup_now()

    assert empty.phase is ProtectionOperationPhase.CANCELED
    assert empty.reason_code is ProtectionReasonCode.NO_BACKUPABLE_MEMORY

    due_settings, due_store_id = _settings(tmp_path / "due")
    state.update_state(
        due_store_id,
        memory_dir=due_settings.memory_dir,
        last_upload_at=protection._utc_now(),
        protection_state=ProtectionState.PROTECTED,
    )
    due_client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, due_client)

    not_due = CloudProtectionService(due_settings).backup_now(only_if_due=True)

    assert not_due.phase is ProtectionOperationPhase.CANCELED
    assert not_due.reason_code is ProtectionReasonCode.NOT_DUE


@pytest.mark.parametrize("method_name", ["backup_now", "verify_now"])
def test_unknown_protection_state_requires_update_without_remote_work_or_write(
    tmp_path, monkeypatch, cloud_state_dir, method_name
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        protection_state="future_protection_state",
    )
    path = state_path(store_id)
    before = path.read_bytes()
    monkeypatch.setattr(
        protection,
        "key_file_exists",
        lambda: (_ for _ in ()).throw(AssertionError("key guard should not run")),
    )
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: (_ for _ in ()).throw(AssertionError("network should not run")),
    )

    result = getattr(CloudProtectionService(settings), method_name)()

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.CLIENT_UPDATE_REQUIRED
    assert result.state is ProtectionState.ATTENTION_REQUIRED
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("method_name", "kind"),
    [
        ("backup_now", ProtectionOperationKind.BACKUP),
        ("verify_now", ProtectionOperationKind.VERIFY),
    ],
)
def test_store_busy_is_transient_and_preserves_durable_health(
    tmp_path, monkeypatch, cloud_state_dir, method_name, kind
):
    settings, store_id = _settings(tmp_path)
    verified_at = protection._utc_now()
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_verify_at=verified_at,
        last_verify_ok=True,
        last_verify_snapshot_id="01VERIFIED",
        last_successful_verify_at=verified_at,
        last_verified_snapshot_id="01VERIFIED",
        protection_state=ProtectionState.PROTECTED,
    )
    before = load_state(store_id).to_dict()
    monkeypatch.setattr(
        protection,
        "StoreLock",
        lambda *args, **kwargs: (_ for _ in ()).throw(StoreLockTimeout("busy")),
    )

    result = getattr(CloudProtectionService(settings), method_name)()

    assert result.kind is kind
    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.STORE_BUSY
    assert result.state is ProtectionState.PROTECTED
    assert load_state(store_id).to_dict() == before


def test_real_store_lock_contention_is_transient_through_the_service(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    state.update_state(
        store_id,
        memory_dir=settings.memory_dir,
        last_verify_ok=True,
        last_verify_snapshot_id=SNAPSHOT_ID,
        last_verified_snapshot_id=SNAPSHOT_ID,
        protection_state=ProtectionState.PROTECTED,
    )
    before = load_state(store_id).to_dict()
    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with StoreLock(settings.memory_dir, timeout=1):
            acquired.set()
            release.wait(2)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert acquired.wait(1)
        monkeypatch.setattr(
            protection,
            "StoreLock",
            lambda memory_dir: StoreLock(memory_dir, timeout=0.1),
        )
        result = CloudProtectionService(settings).verify_now()
    finally:
        release.set()
        holder.join(2)

    assert result.phase is ProtectionOperationPhase.CANCELED
    assert result.reason_code is ProtectionReasonCode.STORE_BUSY
    assert load_state(store_id).to_dict() == before


def test_corrupt_state_stops_backup_and_verification_before_network(
    tmp_path, monkeypatch, cloud_state_dir
):
    settings, store_id = _settings(tmp_path)
    path = state_path(store_id)
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(protection, "key_file_exists", lambda: True)
    monkeypatch.setattr(
        protection,
        "check_entitlement",
        lambda settings: (_ for _ in ()).throw(AssertionError("entitlement should not run")),
    )
    monkeypatch.setattr(
        protection,
        "client_from_settings",
        lambda settings: (_ for _ in ()).throw(AssertionError("network should not run")),
    )

    backup = CloudProtectionService(settings).backup_now()
    verification = CloudProtectionService(settings).verify_now()

    assert backup.phase is ProtectionOperationPhase.FAILED
    assert verification.phase is ProtectionOperationPhase.FAILED
    assert path.read_bytes() == before


def test_scheduler_adapters_delegate_and_keep_legacy_results(monkeypatch):
    calls = []
    backup = ProtectionOperation(
        "backup-op",
        ProtectionOperationKind.BACKUP,
        ProtectionOperationPhase.COMPLETED,
        ProtectionState.CHANGES_PENDING,
        snapshot_id="01BACKUP",
    )
    verify = ProtectionOperation(
        "verify-op",
        ProtectionOperationKind.VERIFY,
        ProtectionOperationPhase.COMPLETED,
        ProtectionState.PROTECTED,
        snapshot_id="01BACKUP",
    )

    class FakeService:
        @classmethod
        def from_engine(cls, engine):
            calls.append(("engine", engine))
            return cls()

        def backup_now(self, reason="manual", *, only_if_due=False):
            calls.append(("backup", reason, only_if_due))
            return backup

        def verify_now(self, snapshot_id=None):
            calls.append(("verify", snapshot_id))
            return verify

    engine = SimpleNamespace(
        settings=SimpleNamespace(cloud_backup_enabled=True, account_token="test-token")
    )
    monkeypatch.setattr(jobs, "CloudProtectionService", FakeService)

    assert jobs.run_cloud_backup(engine) == "01BACKUP"
    assert jobs.run_restore_verification(engine) is True
    assert calls == [
        ("engine", engine),
        ("backup", "scheduled", True),
        ("engine", engine),
        ("verify", None),
    ]


@pytest.mark.parametrize(
    ("adapter", "fallback"),
    [(jobs.run_cloud_backup, None), (jobs.run_restore_verification, False)],
)
def test_scheduler_adapters_swallow_unexpected_exceptions(monkeypatch, caplog, adapter, fallback):
    class BrokenService:
        @classmethod
        def from_engine(cls, engine):
            raise RuntimeError(
                "upstream failed with test-token at "
                "https://objects.example/secret?token=do-not-log"
            )

    monkeypatch.setattr(jobs, "CloudProtectionService", BrokenService)

    engine = SimpleNamespace(
        settings=SimpleNamespace(cloud_backup_enabled=True, account_token="test-token")
    )
    with caplog.at_level(logging.WARNING):
        assert adapter(engine) is fallback

    assert "RuntimeError" in caplog.text
    assert "upstream failed" in caplog.text
    assert "<redacted-url>" in caplog.text
    assert "test-token" not in caplog.text
    assert "objects.example" not in caplog.text
    assert "do-not-log" not in caplog.text


def test_service_can_be_constructed_from_engine(tmp_path):
    settings, _store_id = _settings(tmp_path)

    service = CloudProtectionService.from_engine(SimpleNamespace(settings=settings))

    assert service.settings is settings
