from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ormah.api import routes_protection
from ormah.api.local_auth import LOCAL_ADMIN_HEADER, require_loopback
from ormah.cloud.operations import ProtectionOperationCoordinator
from ormah.cloud.recovery import RecoveryKitError, RecoveryReadiness
from ormah.cloud.state import (
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
    ProtectionReasonCode,
    ProtectionState,
)

LOCAL_TOKEN = "a" * 64
HEADERS = {LOCAL_ADMIN_HEADER: LOCAL_TOKEN}
INTENT_ID = "3f1a6c4e-4b0a-4d5f-9a2b-8c7d6e5f4a3b"
SNAPSHOT_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def _operation(
    kind: ProtectionOperationKind,
    *,
    operation_id: str = "durable-operation",
    intent_id: str | None = None,
) -> ProtectionOperation:
    return ProtectionOperation(
        operation_id=operation_id,
        kind=kind,
        phase=ProtectionOperationPhase.COMPLETED,
        state=ProtectionState.PROTECTED,
        reason_code=None,
        snapshot_id=SNAPSHOT_ID,
        protection_intent_id=intent_id,
    )


@dataclass
class FakeProtectionService:
    release_backup: threading.Event | None = None
    restore_busy_once: bool = False

    def __post_init__(self):
        self.calls: list[tuple] = []

    def create_intent(self):
        self.calls.append(("create_intent",))
        return _operation(ProtectionOperationKind.ENABLE, intent_id=INTENT_ID)

    def bind_intent(self, intent_id):
        self.calls.append(("bind_intent", intent_id))
        return _operation(ProtectionOperationKind.ENABLE, intent_id=intent_id)

    def cancel_intent(self, intent_id):
        self.calls.append(("cancel_intent", intent_id))
        return _operation(ProtectionOperationKind.ENABLE, intent_id=intent_id)

    def enable(self, intent_id):
        self.calls.append(("enable", intent_id))
        return _operation(ProtectionOperationKind.ENABLE, intent_id=intent_id)

    def disable(self):
        self.calls.append(("disable",))
        return _operation(ProtectionOperationKind.DISABLE)

    def backup_and_verify(self, *, reason):
        self.calls.append(("backup_and_verify", reason))
        if self.release_backup is not None:
            assert self.release_backup.wait(timeout=2)
        return _operation(ProtectionOperationKind.BACKUP)

    def verify_now(self, snapshot_id=None):
        self.calls.append(("verify_now", snapshot_id))
        return _operation(ProtectionOperationKind.VERIFY)

    def prepare_restore(self):
        self.calls.append(("prepare_restore",))
        return ProtectionOperation(
            operation_id="durable-restore-preparation",
            kind=ProtectionOperationKind.RESTORE,
            phase=ProtectionOperationPhase.READY,
            state=ProtectionState.PROTECTED,
            snapshot_id=SNAPSHOT_ID,
            verified_node_count=1817,
            snapshot_created_at="2026-08-09T10:00:00+00:00",
            prepared_backup_name="memory_private_prepared_name",
        )

    def restore_prepared(self, prepared):
        self.calls.append(("restore_prepared", prepared.prepared_backup_name))
        if self.restore_busy_once:
            self.restore_busy_once = False
            return ProtectionOperation(
                operation_id="durable-restore-busy",
                kind=ProtectionOperationKind.RESTORE,
                phase=ProtectionOperationPhase.FAILED,
                state=ProtectionState.PROTECTED,
                reason_code=ProtectionReasonCode.STORE_BUSY,
                message="Memory is busy.",
                snapshot_id=prepared.snapshot_id,
            )
        return ProtectionOperation(
            operation_id="durable-restore",
            kind=ProtectionOperationKind.RESTORE,
            phase=ProtectionOperationPhase.COMPLETED,
            state=ProtectionState.PROTECTED,
            snapshot_id=prepared.snapshot_id,
            verified_node_count=prepared.verified_node_count,
            snapshot_created_at=prepared.snapshot_created_at,
            safety_backup_name="memory_safety_backup",
        )

    def discard_prepared_restore(self, prepared):
        self.calls.append(("discard_prepared_restore", prepared.prepared_backup_name))
        return True


class FakeRecoveryKitService:
    def __init__(self):
        self.calls: list[str] = []
        self.fail = False
        self.ready = True

    def ensure_current_kit(self):
        self.calls.append("prepare")
        if self.fail:
            raise RecoveryKitError("secret path and AGE-SECRET-KEY-private")
        return True

    def confirm_saved_digest(self, digest):
        self.calls.append(digest)
        if self.fail:
            raise RecoveryKitError("secret path and AGE-SECRET-KEY-private")
        return RecoveryReadiness(
            self.ready,
            datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        )


@pytest.fixture
def protection_app(tmp_path: Path, monkeypatch):
    async def allow_test_client():
        return None

    service = FakeProtectionService()
    recovery_service = FakeRecoveryKitService()
    coordinator = ProtectionOperationCoordinator(max_workers=4)
    app = FastAPI()
    app.dependency_overrides[require_loopback] = allow_test_client
    app.state.local_admin_token = LOCAL_TOKEN
    app.state.engine = SimpleNamespace(
        settings=SimpleNamespace(memory_dir=tmp_path, account_token="never-return-this-token")
    )
    app.state.protection_service = service
    app.state.protection_operations = coordinator
    app.state.recovery_kit_service = recovery_service
    app.include_router(routes_protection.router)
    monkeypatch.setattr(
        routes_protection,
        "cloud_status_payload",
        lambda settings, **kwargs: {
            "protection_state": "local_only",
            "last_operation_id": None,
            "warnings": [],
        },
    )
    try:
        with TestClient(app) as client:
            yield client, service, coordinator, recovery_service
    finally:
        if service.release_backup is not None:
            service.release_backup.set()
        coordinator.shutdown()


def _poll(client: TestClient, operation_id: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(
            f"/admin/cloud/protection/operations/{operation_id}",
            headers=HEADERS,
        )
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.005)
    raise AssertionError("operation did not finish")


def test_all_protection_routes_require_local_capability(protection_app):
    client, _, _, _ = protection_app
    requests = [
        ("GET", "/admin/cloud/protection", None),
        ("POST", "/admin/cloud/protection/intents", {}),
        ("POST", f"/admin/cloud/protection/intents/{INTENT_ID}/bind", {}),
        ("POST", f"/admin/cloud/protection/intents/{INTENT_ID}/cancel", {}),
        ("POST", f"/admin/cloud/protection/intents/{INTENT_ID}/enable", {}),
        ("POST", "/admin/cloud/protection/disable", {}),
        ("POST", "/admin/cloud/protection/backup", {}),
        ("POST", "/admin/cloud/protection/verify", {}),
        ("POST", "/admin/cloud/protection/restore/prepare", {}),
        (
            "POST",
            f"/admin/cloud/protection/restore/{INTENT_ID}/confirm",
            {},
        ),
        ("POST", "/admin/cloud/protection/recovery-kit/prepare", {}),
        (
            "POST",
            "/admin/cloud/protection/recovery-kit/confirm",
            {"sha256_digest": "a" * 64},
        ),
        ("GET", f"/admin/cloud/protection/operations/{INTENT_ID}", None),
    ]

    for method, path, body in requests:
        response = client.request(method, path, json=body)
        assert response.status_code == 401, path


def test_status_and_intent_adapters_are_thin_and_token_free(protection_app):
    client, service, _, _ = protection_app

    status_response = client.get("/admin/cloud/protection", headers=HEADERS)
    create_response = client.post(
        "/admin/cloud/protection/intents", headers=HEADERS, json={}
    )
    bind_response = client.post(
        f"/admin/cloud/protection/intents/{INTENT_ID}/bind", headers=HEADERS, json={}
    )
    cancel_response = client.post(
        f"/admin/cloud/protection/intents/{INTENT_ID}/cancel", headers=HEADERS, json={}
    )

    assert status_response.json()["protection_state"] == "local_only"
    assert create_response.json()["protection_intent_id"] == INTENT_ID
    assert bind_response.json()["protection_intent_id"] == INTENT_ID
    assert cancel_response.json()["protection_intent_id"] == INTENT_ID
    assert service.calls == [
        ("create_intent",),
        ("bind_intent", INTENT_ID),
        ("cancel_intent", INTENT_ID),
    ]
    assert "never-return-this-token" not in str(
        [
            status_response.json(),
            create_response.json(),
            bind_response.json(),
            cancel_response.json(),
        ]
    )


def test_product_status_redacts_paths_credentials_and_secret_material(
    protection_app,
    monkeypatch,
):
    client, _, _, _ = protection_app
    monkeypatch.setattr(
        routes_protection,
        "cloud_status_payload",
        lambda settings, **kwargs: {
            "protection_state": "attention_required",
            "state_error": "failed at /home/person/cloud-state.json",
            "last_upload_error": "token never-return-this-token",
            "last_verify_error": "AGE-SECRET-KEY-private",
            "last_error_message": "https://example.test/private",
            "warnings": ["bad file /tmp/recovery/kit.md"],
        },
    )

    response = client.get("/admin/cloud/protection", headers=HEADERS)

    assert response.status_code == 200
    serialized = str(response.json()).lower()
    assert "state_error" not in response.json()
    for forbidden in [
        "/home/person",
        "/tmp/recovery",
        "never-return-this-token",
        "age-secret-key",
        "https://",
    ]:
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/admin/cloud/protection/intents", {"email": "person@example.com"}),
        (f"/admin/cloud/protection/intents/{INTENT_ID}/bind", {"account_id": "secret"}),
        (f"/admin/cloud/protection/intents/{INTENT_ID}/cancel", {"force": True}),
        (f"/admin/cloud/protection/intents/{INTENT_ID}/enable", {"price_id": "price_x"}),
        ("/admin/cloud/protection/disable", {"delete_remote": True}),
        ("/admin/cloud/protection/backup", {"advance_head": True}),
        ("/admin/cloud/protection/verify", {"presigned_url": "https://example.test"}),
        ("/admin/cloud/protection/restore/prepare", {"snapshot_id": SNAPSHOT_ID}),
        (
            "/admin/cloud/protection/recovery-kit/prepare",
            {"path": "/secret"},
        ),
        (
            "/admin/cloud/protection/recovery-kit/confirm",
            {"sha256_digest": "a" * 64, "path": "/secret"},
        ),
    ],
)
def test_mutation_routes_forbid_extra_fields(protection_app, path, body):
    client, _, _, _ = protection_app
    response = client.post(path, headers=HEADERS, json=body)
    assert response.status_code == 422


def test_long_operations_return_202_and_poll_safe_results(protection_app):
    client, service, _, _ = protection_app
    requests = [
        (
            f"/admin/cloud/protection/intents/{INTENT_ID}/enable",
            {},
            ("enable", INTENT_ID),
        ),
        ("/admin/cloud/protection/disable", {}, ("disable",)),
        ("/admin/cloud/protection/backup", {}, ("backup_and_verify", "manual-ui")),
        (
            "/admin/cloud/protection/verify",
            {"snapshot_id": SNAPSHOT_ID},
            ("verify_now", SNAPSHOT_ID),
        ),
    ]

    for path, body, expected_call in requests:
        response = client.post(path, headers=HEADERS, json=body)
        assert response.status_code == 202
        payload = _poll(client, response.json()["operation_id"])
        assert payload["status"] == "completed"
        assert payload["durable_operation_id"] == "durable-operation"
        assert payload["protection_state"] == "protected"
        assert "never-return-this-token" not in str(payload)
        assert expected_call in service.calls


def test_repeated_active_backup_joins_one_operation(protection_app):
    client, service, _, _ = protection_app
    service.release_backup = threading.Event()

    first = client.post("/admin/cloud/protection/backup", headers=HEADERS, json={})
    second = client.post("/admin/cloud/protection/backup", headers=HEADERS, json={})

    assert first.status_code == second.status_code == 202
    assert first.json()["operation_id"] == second.json()["operation_id"]
    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True
    service.release_backup.set()
    assert _poll(client, first.json()["operation_id"])["status"] == "completed"
    assert service.calls.count(("backup_and_verify", "manual-ui")) == 1


def test_restore_preparation_is_verified_then_claimed_once(protection_app):
    client, service, _, _ = protection_app

    response = client.post(
        "/admin/cloud/protection/restore/prepare",
        headers=HEADERS,
        json={},
    )
    assert response.status_code == 202
    preparation_id = response.json()["operation_id"]
    prepared = _poll(client, preparation_id)

    assert prepared["phase"] == "ready"
    assert prepared["verified_node_count"] == 1817
    assert prepared["snapshot_created_at"] == "2026-08-09T10:00:00+00:00"
    assert "memory_private_prepared_name" not in str(prepared)

    confirmed = client.post(
        f"/admin/cloud/protection/restore/{preparation_id}/confirm",
        headers=HEADERS,
        json={},
    )
    assert confirmed.status_code == 202
    restored = _poll(client, confirmed.json()["operation_id"])
    assert restored["phase"] == "completed"
    assert restored["safety_backup_name"] == "memory_safety_backup"
    assert ("restore_prepared", "memory_private_prepared_name") in service.calls

    repeated = client.post(
        f"/admin/cloud/protection/restore/{preparation_id}/confirm",
        headers=HEADERS,
        json={},
    )
    assert repeated.status_code == 409


def test_restore_preparation_cancel_discards_private_copy(protection_app):
    client, service, _, _ = protection_app
    response = client.post(
        "/admin/cloud/protection/restore/prepare", headers=HEADERS, json={}
    )
    preparation_id = response.json()["operation_id"]
    assert _poll(client, preparation_id)["phase"] == "ready"

    canceled = client.post(
        f"/admin/cloud/protection/restore/{preparation_id}/cancel",
        headers=HEADERS,
        json={},
    )

    assert canceled.status_code == 200
    assert canceled.json() == {"status": "discarded"}
    assert ("discard_prepared_restore", "memory_private_prepared_name") in service.calls
    repeated = client.post(
        f"/admin/cloud/protection/restore/{preparation_id}/cancel",
        headers=HEADERS,
        json={},
    )
    assert repeated.status_code == 409


def test_store_busy_restore_can_retry_same_verified_preparation(protection_app):
    client, service, _, _ = protection_app
    service.restore_busy_once = True
    response = client.post(
        "/admin/cloud/protection/restore/prepare", headers=HEADERS, json={}
    )
    preparation_id = response.json()["operation_id"]
    assert _poll(client, preparation_id)["phase"] == "ready"

    first = client.post(
        f"/admin/cloud/protection/restore/{preparation_id}/confirm",
        headers=HEADERS,
        json={},
    )
    assert _poll(client, first.json()["operation_id"])["reason_code"] == "store_busy"
    second = client.post(
        f"/admin/cloud/protection/restore/{preparation_id}/confirm",
        headers=HEADERS,
        json={},
    )
    assert second.status_code == 202
    assert _poll(client, second.json()["operation_id"])["phase"] == "completed"


def test_unknown_operation_returns_404(protection_app):
    client, _, _, _ = protection_app
    response = client.get(
        "/admin/cloud/protection/operations/unknown",
        headers=HEADERS,
    )
    assert response.status_code == 404


def test_polling_entitlement_is_cache_only(monkeypatch):
    settings = SimpleNamespace(account_token="private-token")
    cached = object()
    calls = []
    monkeypatch.setattr(routes_protection, "load_entitlement_cache", lambda: cached)
    monkeypatch.setattr(
        routes_protection,
        "status_from_cache",
        lambda value: calls.append(value) or SimpleNamespace(value="grace"),
    )

    assert routes_protection._cached_entitlement(settings) == "grace"
    assert calls == [cached]

    settings.account_token = None
    assert routes_protection._cached_entitlement(settings) == "none"
    assert calls == [cached]


def test_invalid_intent_and_snapshot_ids_fail_before_service_work(protection_app):
    client, service, _, _ = protection_app

    invalid_intent = client.post(
        "/admin/cloud/protection/intents/not-a-uuid/enable",
        headers=HEADERS,
        json={},
    )
    invalid_snapshot = client.post(
        "/admin/cloud/protection/verify",
        headers=HEADERS,
        json={"snapshot_id": "not-a-snapshot"},
    )

    assert invalid_intent.status_code == 422
    assert invalid_snapshot.status_code == 422
    assert service.calls == []


def test_recovery_confirmation_is_typed_and_secret_free(protection_app):
    client, _, _, recovery_service = protection_app

    response = client.post(
        "/admin/cloud/protection/recovery-kit/confirm",
        headers=HEADERS,
        json={"sha256_digest": "a" * 64},
    )

    assert response.status_code == 200
    assert response.json() == {
        "device_loss_recovery_ready": True,
        "recovery_kit_verified_at": "2026-07-31T12:00:00+00:00",
    }
    assert recovery_service.calls == ["a" * 64]
    serialized = str(response.json()).lower()
    for forbidden in ["age-secret-key", "path", "token", "digest", "identity"]:
        assert forbidden not in serialized


def test_recovery_kit_prepare_repairs_server_side_without_returning_material(
    protection_app,
):
    client, _, _, recovery_service = protection_app

    response = client.post(
        "/admin/cloud/protection/recovery-kit/prepare",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ready_to_save", "regenerated": True}
    assert recovery_service.calls == ["prepare"]
    serialized = str(response.json()).lower()
    for forbidden in ["age-secret-key", "path", "token", "identity", "store_id"]:
        assert forbidden not in serialized


def test_recovery_kit_prepare_failure_is_generic(protection_app):
    client, _, _, recovery_service = protection_app
    recovery_service.fail = True

    response = client.post(
        "/admin/cloud/protection/recovery-kit/prepare",
        headers=HEADERS,
        json={},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "The recovery kit could not be prepared for saving."}
    assert "secret" not in response.text.lower()


def test_recovery_confirmation_does_not_invent_readiness(protection_app):
    client, _, _, recovery_service = protection_app
    recovery_service.ready = False

    response = client.post(
        "/admin/cloud/protection/recovery-kit/confirm",
        headers=HEADERS,
        json={"sha256_digest": "a" * 64},
    )

    assert response.status_code == 200
    assert response.json()["device_loss_recovery_ready"] is False


@pytest.mark.parametrize("digest", ["ABC", "A" * 64, "a" * 63, "a" * 65])
def test_malformed_recovery_digest_is_rejected_before_service_work(
    protection_app,
    digest,
):
    client, _, _, recovery_service = protection_app

    response = client.post(
        "/admin/cloud/protection/recovery-kit/confirm",
        headers=HEADERS,
        json={"sha256_digest": digest},
    )

    assert response.status_code == 422
    assert recovery_service.calls == []


def test_recovery_confirmation_failure_does_not_leak_service_error(protection_app):
    client, _, _, recovery_service = protection_app
    recovery_service.fail = True

    response = client.post(
        "/admin/cloud/protection/recovery-kit/confirm",
        headers=HEADERS,
        json={"sha256_digest": "a" * 64},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "The saved recovery kit could not be verified."}
    assert "secret" not in response.text.lower()
