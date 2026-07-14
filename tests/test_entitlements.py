from __future__ import annotations

import stat
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ormah.cloud import entitlements
from ormah.cloud.entitlements import EntitlementStatus

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.closed = False

    def get_entitlements(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / "entitlements.json"
    monkeypatch.setattr(entitlements, "ENTITLEMENT_CACHE_PATH", path)
    return path


def _settings(token="opaque-token"):
    return SimpleNamespace(
        account_token=token,
        cloud_api_url="https://cloud.test",
    )


@pytest.mark.parametrize(
    ("age", "payload", "expected"),
    [
        (timedelta(hours=1), {"backup": True, "plan_status": "active"}, EntitlementStatus.ACTIVE),
        (timedelta(hours=24), {"backup": True}, EntitlementStatus.ACTIVE),
        (timedelta(hours=24, microseconds=1), {"backup": True}, EntitlementStatus.GRACE),
        (timedelta(days=2), {"backup": True, "plan_status": "active"}, EntitlementStatus.GRACE),
        (timedelta(days=7), {"backup": True}, EntitlementStatus.GRACE),
        (timedelta(days=7, microseconds=1), {"backup": True}, EntitlementStatus.EXPIRED),
        (timedelta(days=8), {"backup": True, "plan_status": "active"}, EntitlementStatus.EXPIRED),
        (timedelta(hours=1), {"backup": False, "plan_status": "none"}, EntitlementStatus.EXPIRED),
        (timedelta(hours=1), {"plan_status": "active"}, EntitlementStatus.EXPIRED),
        (timedelta(hours=1), {"backup": "true"}, EntitlementStatus.EXPIRED),
    ],
)
def test_cached_entitlement_states(age, payload, expected, monkeypatch):
    entitlements.cache_entitlements(payload, fetched_at=NOW - age)
    offline = FakeClient(error=OSError("offline"))
    monkeypatch.setattr(entitlements, "client_from_settings", lambda settings: offline)

    assert entitlements.check_entitlement(_settings(), now=NOW) is expected
    if age <= timedelta(hours=24):
        assert offline.calls == 0
    else:
        assert offline.calls == 1


def test_no_token_returns_none_without_refresh(monkeypatch):
    client = FakeClient(response={"backup": True})
    monkeypatch.setattr(entitlements, "client_from_settings", lambda settings: client)

    assert entitlements.check_entitlement(_settings(token=None), now=NOW) is EntitlementStatus.NONE
    assert client.calls == 0


def test_no_cache_offline_returns_none_and_never_raises(monkeypatch):
    offline = FakeClient(error=OSError("offline"))
    monkeypatch.setattr(entitlements, "client_from_settings", lambda settings: offline)

    assert entitlements.check_entitlement(_settings(), now=NOW) is EntitlementStatus.NONE
    assert offline.closed is True


def test_missing_cache_refreshes_to_active(monkeypatch):
    client = FakeClient(
        response={"backup": True, "founding": False, "plan_status": "past_due"}
    )
    monkeypatch.setattr(entitlements, "client_from_settings", lambda settings: client)

    assert entitlements.check_entitlement(_settings(), now=NOW) is EntitlementStatus.ACTIVE
    cache = entitlements.load_entitlement_cache()
    assert cache is not None
    assert cache.plan_status == "past_due"
    assert cache.fetched_at == NOW


def test_successful_malformed_refresh_can_never_be_active(monkeypatch):
    entitlements.cache_entitlements(
        {"backup": True, "plan_status": "active"},
        fetched_at=NOW - timedelta(days=2),
    )
    client = FakeClient(response={"plan_status": "active"})
    monkeypatch.setattr(entitlements, "client_from_settings", lambda settings: client)

    assert entitlements.check_entitlement(_settings(), now=NOW) is EntitlementStatus.EXPIRED
    assert entitlements.load_entitlement_cache().entitlements.get("backup") is None


def test_cache_is_raw_response_plus_timestamp_and_mode_0600(cache_path):
    raw = {"backup": True, "founding": True, "plan_status": "trialing", "future": "kept"}
    entitlements.cache_entitlements(raw, fetched_at=NOW)

    text = cache_path.read_text()
    assert '"future": "kept"' in text
    assert '"fetched_at": "2026-07-13T12:00:00+00:00"' in text
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600


def test_corrupt_cache_is_ignored(monkeypatch, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not-json")
    offline = FakeClient(error=OSError("offline"))
    monkeypatch.setattr(entitlements, "client_from_settings", lambda settings: offline)

    assert entitlements.check_entitlement(_settings(), now=NOW) is EntitlementStatus.NONE


def test_clear_cache_is_idempotent(cache_path):
    entitlements.cache_entitlements({"backup": True}, fetched_at=NOW)
    entitlements.clear_entitlement_cache()
    entitlements.clear_entitlement_cache()
    assert not cache_path.exists()
