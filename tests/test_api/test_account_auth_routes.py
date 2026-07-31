"""Tests for token-free local account authentication adapters."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ormah import setup
from ormah.api.local_auth import LOCAL_ADMIN_HEADER, require_loopback
from ormah.api.routes_account import router as account_router
from ormah.cloud import client as cloud_client
from ormah.cloud import entitlements
from ormah.cloud.client import CloudError
from ormah.config import Settings


LOCAL_TOKEN = "a" * 64
HEADERS = {LOCAL_ADMIN_HEADER: LOCAL_TOKEN}
BEARER_TOKEN = "account-bearer-token-must-never-leak"


class FakeCloudClient:
    def __init__(self, *, verify_error=None, entitlement_error=None, revoke_error=None):
        self.verify_error = verify_error
        self.entitlement_error = entitlement_error
        self.revoke_error = revoke_error
        self.calls: list[tuple] = []
        self.before_revoke = None

    def request_code(self, email):
        self.calls.append(("request_code", email))
        return {"message": "generic"}

    def verify_code(self, email, code, device_id, device_name):
        self.calls.append(("verify_code", email, code, device_id, device_name))
        if self.verify_error:
            raise self.verify_error
        return BEARER_TOKEN

    def get_entitlements(self):
        self.calls.append(("get_entitlements",))
        if self.entitlement_error:
            raise self.entitlement_error
        return {"backup": True, "plan_status": "active", "account_id": "ignored"}

    def revoke_token(self):
        self.calls.append(("revoke_token",))
        if self.before_revoke:
            self.before_revoke()
        if self.revoke_error:
            raise self.revoke_error
        return {"revoked": True}


@pytest.fixture
def account_paths(tmp_path, monkeypatch):
    env_path = tmp_path / "config" / ".env"
    device_path = tmp_path / "data" / "device_id"
    cache_path = tmp_path / "data" / "entitlements.json"
    monkeypatch.setattr(setup, "ENV_PATH", env_path)
    monkeypatch.setattr(setup, "ENV_DIR", env_path.parent)
    monkeypatch.setattr(cloud_client, "DEVICE_ID_PATH", device_path)
    monkeypatch.setattr(entitlements, "ENTITLEMENT_CACHE_PATH", cache_path)
    return env_path, device_path, cache_path


def build_client(tmp_path, fake, *, token=None, email=None):
    async def allow_test_client():
        return None

    settings = Settings(
        memory_dir=tmp_path / "memory",
        cloud_api_url="https://cloud.test",
        account_token=token,
        account_email=email,
    )
    app = FastAPI()
    app.include_router(account_router)
    app.dependency_overrides[require_loopback] = allow_test_client
    app.state.engine = SimpleNamespace(settings=settings)
    app.state.local_admin_token = LOCAL_TOKEN
    app.state.cloud_client = fake
    return settings, app


def test_otp_routes_require_local_capability(tmp_path, account_paths):
    fake = FakeCloudClient()
    _, app = build_client(tmp_path, fake)

    with TestClient(app) as http:
        assert http.post(
            "/admin/account/request-code", json={"email": "person@example.com"}
        ).status_code == 401
        assert http.post(
            "/admin/account/verify",
            json={"email": "person@example.com", "code": "123456"},
        ).status_code == 401

    assert fake.calls == []


def test_request_code_is_generic_normalized_and_rejects_extra_fields(
    tmp_path, account_paths
):
    fake = FakeCloudClient()
    _, app = build_client(tmp_path, fake)

    with TestClient(app, headers=HEADERS) as http:
        response = http.post(
            "/admin/account/request-code", json={"email": " Person@Example.com "}
        )
        rejected = http.post(
            "/admin/account/request-code",
            json={"email": "person@example.com", "device_id": "caller-controlled"},
        )

    assert response.status_code == 202
    assert response.json() == {"status": "code_sent"}
    assert rejected.status_code == 422
    assert fake.calls == [("request_code", "person@example.com")]


def test_verify_persists_privately_updates_live_settings_and_returns_no_token(
    tmp_path, account_paths
):
    env_path, device_path, cache_path = account_paths
    env_path.parent.mkdir(parents=True)
    env_path.write_text("KEEP=exact\n", encoding="utf-8")
    fake = FakeCloudClient()
    settings, app = build_client(tmp_path, fake)

    with TestClient(app, headers=HEADERS) as http:
        response = http.post(
            "/admin/account/verify",
            json={"email": "Person@Example.com", "code": "123456"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "signed_in": True,
        "email": "person@example.com",
        "device_name": payload["device_name"],
        "entitlement": "active",
        "plan_status": "active",
        "cache_age_seconds": payload["cache_age_seconds"],
        "entitlement_available": True,
    }
    assert BEARER_TOKEN not in response.text
    assert "token" not in payload
    assert settings.account_token == BEARER_TOKEN
    assert settings.account_email == "person@example.com"
    assert f"ORMAH_ACCOUNT_TOKEN={BEARER_TOKEN}\n" in env_path.read_text()
    assert "ORMAH_ACCOUNT_EMAIL=person@example.com\n" in env_path.read_text()
    assert device_path.is_file()
    assert cache_path.is_file()
    assert [call[0] for call in fake.calls] == ["verify_code", "get_entitlements"]


def test_verify_error_is_static_and_cannot_echo_remote_token(tmp_path, account_paths):
    fake = FakeCloudClient(
        verify_error=CloudError(
            f"invalid {BEARER_TOKEN}",
            status_code=401,
            payload={"token": BEARER_TOKEN},
        )
    )
    _, app = build_client(tmp_path, fake)

    with TestClient(app, headers=HEADERS) as http:
        response = http.post(
            "/admin/account/verify",
            json={"email": "person@example.com", "code": "123456"},
        )

    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "invalid_or_expired_code"
    assert BEARER_TOKEN not in response.text


def test_verify_keeps_login_when_entitlement_refresh_is_offline(
    tmp_path, account_paths
):
    fake = FakeCloudClient(entitlement_error=CloudError("offline"))
    settings, app = build_client(tmp_path, fake)

    with TestClient(app, headers=HEADERS) as http:
        response = http.post(
            "/admin/account/verify",
            json={"email": "person@example.com", "code": "123456"},
        )

    assert response.status_code == 200
    assert response.json()["signed_in"] is True
    assert response.json()["entitlement_available"] is False
    assert response.json()["entitlement"] == "none"
    assert settings.account_token == BEARER_TOKEN


def test_logout_revokes_first_then_clears_locally_even_offline(
    tmp_path, account_paths
):
    env_path, _, cache_path = account_paths
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "KEEP=exact\n"
        f"ORMAH_ACCOUNT_TOKEN={BEARER_TOKEN}\n"
        "ORMAH_ACCOUNT_EMAIL=person@example.com\n",
        encoding="utf-8",
    )
    entitlements.cache_entitlements({"backup": True, "plan_status": "active"})
    fake = FakeCloudClient(revoke_error=CloudError("offline"))
    settings, app = build_client(
        tmp_path,
        fake,
        token=BEARER_TOKEN,
        email="person@example.com",
    )

    def assert_local_state_exists():
        assert "ORMAH_ACCOUNT_TOKEN" in env_path.read_text()
        assert cache_path.is_file()

    fake.before_revoke = assert_local_state_exists
    with TestClient(app, headers=HEADERS) as http:
        response = http.post("/admin/account/logout", json={})

    assert response.status_code == 200
    assert response.json() == {
        "signed_in": False,
        "revoked_remotely": False,
        "warning": "Could not revoke this device while offline.",
    }
    assert fake.calls == [("revoke_token",)]
    assert env_path.read_text() == "KEEP=exact\n"
    assert not cache_path.exists()
    assert settings.account_token is None
    assert settings.account_email is None


def test_status_is_token_free_and_logout_requires_explicit_json(
    tmp_path, account_paths
):
    entitlements.cache_entitlements({"backup": True, "plan_status": "trialing"})
    fake = FakeCloudClient()
    _, app = build_client(
        tmp_path,
        fake,
        token=BEARER_TOKEN,
        email="person@example.com",
    )

    with TestClient(app, headers=HEADERS) as http:
        status = http.get("/admin/account/status")
        no_body = http.post("/admin/account/logout")
        extra = http.post("/admin/account/logout", json={"token": BEARER_TOKEN})

    assert status.status_code == 200
    assert status.json()["email"] == "person@example.com"
    assert status.json()["entitlement"] == "active"
    assert BEARER_TOKEN not in status.text
    assert "token" not in status.text.lower()
    assert no_body.status_code == 422
    assert extra.status_code == 422
    assert fake.calls == []
