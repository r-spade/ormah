from __future__ import annotations

from dataclasses import dataclass
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
from ormah.cloud.state import (
    ProtectionOperation,
    ProtectionOperationKind,
    ProtectionOperationPhase,
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

    def backup_now(self, *, reason):
        self.calls.append(("backup_now", reason))
        if self.release_backup is not None:
            assert self.release_backup.wait(timeout=2)
        return _operation(ProtectionOperationKind.BACKUP)

    def verify_now(self, snapshot_id=None):
        self.calls.append(("verify_now", snapshot_id))
        return _operation(ProtectionOperationKind.VERIFY)


@pytest.fixture
def protection_app(tmp_path: Path, monkeypatch):
    async def allow_test_client():
        return None

    service = FakeProtectionService()
    coordinator = ProtectionOperationCoordinator(max_workers=4)
    app = FastAPI()
    app.dependency_overrides[require_loopback] = allow_test_client
    app.state.local_admin_token = LOCAL_TOKEN
    app.state.engine = SimpleNamespace(
        settings=SimpleNamespace(memory_dir=tmp_path, account_token="never-return-this-token")
    )
    app.state.protection_service = service
    app.state.protection_operations = coordinator
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
            yield client, service, coordinator
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
    client, _, _ = protection_app
    requests = [
        ("GET", "/admin/cloud/protection", None),
        ("POST", "/admin/cloud/protection/intents", {}),
        ("POST", f"/admin/cloud/protection/intents/{INTENT_ID}/bind", {}),
        ("POST", f"/admin/cloud/protection/intents/{INTENT_ID}/cancel", {}),
        ("POST", f"/admin/cloud/protection/intents/{INTENT_ID}/enable", {}),
        ("POST", "/admin/cloud/protection/disable", {}),
        ("POST", "/admin/cloud/protection/backup", {}),
        ("POST", "/admin/cloud/protection/verify", {}),
        ("GET", f"/admin/cloud/protection/operations/{INTENT_ID}", None),
    ]

    for method, path, body in requests:
        response = client.request(method, path, json=body)
        assert response.status_code == 401, path


def test_status_and_intent_adapters_are_thin_and_token_free(protection_app):
    client, service, _ = protection_app

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
    ],
)
def test_mutation_routes_forbid_extra_fields(protection_app, path, body):
    client, _, _ = protection_app
    response = client.post(path, headers=HEADERS, json=body)
    assert response.status_code == 422


def test_long_operations_return_202_and_poll_safe_results(protection_app):
    client, service, _ = protection_app
    requests = [
        (
            f"/admin/cloud/protection/intents/{INTENT_ID}/enable",
            {},
            ("enable", INTENT_ID),
        ),
        ("/admin/cloud/protection/disable", {}, ("disable",)),
        ("/admin/cloud/protection/backup", {}, ("backup_now", "manual-ui")),
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
    client, service, _ = protection_app
    service.release_backup = threading.Event()

    first = client.post("/admin/cloud/protection/backup", headers=HEADERS, json={})
    second = client.post("/admin/cloud/protection/backup", headers=HEADERS, json={})

    assert first.status_code == second.status_code == 202
    assert first.json()["operation_id"] == second.json()["operation_id"]
    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True
    service.release_backup.set()
    assert _poll(client, first.json()["operation_id"])["status"] == "completed"
    assert service.calls.count(("backup_now", "manual-ui")) == 1


def test_unknown_operation_returns_404(protection_app):
    client, _, _ = protection_app
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
    client, service, _ = protection_app

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
