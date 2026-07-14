from __future__ import annotations

import stat
import uuid
from types import SimpleNamespace

import httpx
import pytest
import respx

from ormah.cloud.client import (
    CloudClient,
    CloudError,
    client_from_settings,
    get_or_create_device_id,
)

BASE_URL = "https://cloud.test"
TOKEN = "opaque-account-token"


@respx.mock
def test_auth_and_entitlement_requests_match_service_shapes():
    request_route = respx.post(f"{BASE_URL}/auth/request-code").mock(
        return_value=httpx.Response(202, json={"message": "sent"})
    )
    verify_route = respx.post(f"{BASE_URL}/auth/verify").mock(
        return_value=httpx.Response(200, json={"token": TOKEN, "token_type": "bearer"})
    )
    entitlement_route = respx.get(f"{BASE_URL}/me/entitlements").mock(
        return_value=httpx.Response(
            200,
            json={"backup": True, "founding": False, "plan_status": "active"},
        )
    )
    device_id = str(uuid.uuid4())

    with CloudClient(BASE_URL) as client:
        client.request_code("Person@Example.com")
        assert client.verify_code(
            "Person@Example.com", "123456", device_id, "Test laptop"
        ) == TOKEN
        assert client.get_entitlements()["plan_status"] == "active"

    assert request_route.calls[0].request.content == b'{"email":"person@example.com"}'
    assert "authorization" not in request_route.calls[0].request.headers
    assert b'"device_id"' in verify_route.calls[0].request.content
    assert "authorization" not in verify_route.calls[0].request.headers
    assert entitlement_route.calls[0].request.headers["authorization"] == f"Bearer {TOKEN}"


@respx.mock
def test_revoke_resolves_server_token_id_from_device():
    device_id = str(uuid.uuid4())
    respx.get(f"{BASE_URL}/me/tokens").mock(
        return_value=httpx.Response(
            200,
            json={
                "tokens": [
                    {
                        "token_id": "server-side-hmac-id",
                        "device_id": device_id,
                        "device_name": "Laptop",
                    }
                ]
            },
        )
    )
    revoke_route = respx.post(f"{BASE_URL}/me/tokens/revoke").mock(
        return_value=httpx.Response(200, json={"revoked": True})
    )

    with CloudClient(BASE_URL, TOKEN, device_id=device_id) as client:
        assert client.revoke_token() == {"revoked": True}

    assert revoke_route.calls[0].request.content == b'{"token_id":"server-side-hmac-id"}'
    assert TOKEN.encode() not in revoke_route.calls[0].request.content


@respx.mock
def test_upload_blob_and_head_methods_match_service_shapes():
    store_id = str(uuid.uuid4())
    create_route = respx.post(f"{BASE_URL}/stores/{store_id}/uploads").mock(
        return_value=httpx.Response(
            201,
            json={"upload_id": "upload", "snapshot_id": "snapshot", "put_url": "https://put"},
        )
    )
    finalize_route = respx.post(
        f"{BASE_URL}/stores/{store_id}/uploads/upload/finalize"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "snapshot_id": "snapshot",
                "status": "committed",
                "head": {"snapshot_id": "snapshot", "seq": 4},
            },
        )
    )
    respx.get(f"{BASE_URL}/stores/{store_id}/blobs").mock(
        return_value=httpx.Response(200, json={"blobs": [{"snapshot_id": "snapshot"}]})
    )
    download_route = respx.post(f"{BASE_URL}/stores/{store_id}/presign-download").mock(
        return_value=httpx.Response(200, json={"get_url": "https://get"})
    )
    respx.get(f"{BASE_URL}/stores/{store_id}/head").mock(
        return_value=httpx.Response(200, json={"snapshot_id": "snapshot", "seq": 4})
    )

    with CloudClient(BASE_URL, TOKEN) as client:
        client.create_upload(store_id, 42, "a" * 64)
        client.finalize_upload(store_id, "upload", {"expected_seq": 3})
        assert client.list_blobs(store_id)["blobs"][0]["snapshot_id"] == "snapshot"
        assert client.presign_download(store_id, "snapshot") == {"get_url": "https://get"}
        assert client.get_head(store_id) == {"snapshot_id": "snapshot", "seq": 4}

    assert create_route.calls[0].request.content == (
        b'{"size_bytes":42,"sha256":"' + b"a" * 64 + b'"}'
    )
    assert finalize_route.calls[0].request.content == b'{"advance_head":{"expected_seq":3}}'
    assert download_route.calls[0].request.content == b'{"snapshot_id":"snapshot"}'


@respx.mock
def test_finalize_cas_conflict_returns_current_head_payload():
    store_id = str(uuid.uuid4())
    respx.post(f"{BASE_URL}/stores/{store_id}/uploads/upload/finalize").mock(
        return_value=httpx.Response(
            409,
            json={
                "detail": {
                    "message": "Sync head changed.",
                    "snapshot_id": "loser",
                    "status": "committed",
                    "head": {"snapshot_id": "winner", "seq": 2},
                }
            },
        )
    )

    with CloudClient(BASE_URL, TOKEN) as client:
        result = client.finalize_upload(store_id, "upload", {"expected_seq": 1})

    assert result["status"] == "committed"
    assert result["head"] == {"snapshot_id": "winner", "seq": 2}


@pytest.mark.parametrize("status_code", [401, 403, 500])
@respx.mock
def test_non_success_responses_raise_cloud_error(status_code):
    respx.get(f"{BASE_URL}/me/entitlements").mock(
        return_value=httpx.Response(status_code, json={"detail": "denied"})
    )

    with CloudClient(BASE_URL, TOKEN) as client:
        with pytest.raises(CloudError) as exc_info:
            client.get_entitlements()

    assert exc_info.value.status_code == status_code
    assert TOKEN not in str(exc_info.value)


@respx.mock
def test_non_cas_conflict_still_raises():
    store_id = str(uuid.uuid4())
    respx.post(f"{BASE_URL}/stores/{store_id}/uploads/upload/finalize").mock(
        return_value=httpx.Response(409, json={"detail": "Uploaded object is not available."})
    )

    with CloudClient(BASE_URL, TOKEN) as client:
        with pytest.raises(CloudError, match="not available"):
            client.finalize_upload(store_id, "upload")


@respx.mock
def test_network_errors_are_wrapped():
    respx.get(f"{BASE_URL}/me/entitlements").mock(
        side_effect=httpx.ConnectError("offline")
    )
    with CloudClient(BASE_URL, TOKEN) as client:
        with pytest.raises(CloudError, match="Could not reach"):
            client.get_entitlements()


def test_device_id_is_stable_uuid4_and_mode_0600(tmp_path):
    path = tmp_path / "device_id"
    first = get_or_create_device_id(path)
    second = get_or_create_device_id(path)

    assert first == second
    assert uuid.UUID(first).version == 4
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_device_id_fails_closed(tmp_path):
    path = tmp_path / "device_id"
    path.write_text("not-a-uuid\n")

    with pytest.raises(CloudError, match="expected UUIDv4"):
        get_or_create_device_id(path)


def test_client_factory_reads_settings():
    client = client_from_settings(
        SimpleNamespace(cloud_api_url=BASE_URL, account_token=TOKEN)
    )
    try:
        assert client.base_url == f"{BASE_URL}/"
        assert client.token == TOKEN
    finally:
        client.close()
