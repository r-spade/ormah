from __future__ import annotations

import json
import stat
import uuid
from unittest.mock import patch

import pytest

from ormah import cli
from ormah.cloud import client as cloud_client
from ormah.cloud import entitlements
from ormah.cloud.client import CloudError
from ormah.config import Settings, settings


class FakeClient:
    def __init__(self, *, entitlement=None, revoke_error=None, before_revoke=None):
        self.entitlement = entitlement or {
            "backup": True,
            "founding": False,
            "plan_status": "active",
        }
        self.revoke_error = revoke_error
        self.before_revoke = before_revoke
        self.calls = []
        self.closed = False

    def request_code(self, email):
        self.calls.append(("request_code", email))

    def verify_code(self, email, code, device_id, device_name):
        self.calls.append(("verify_code", email, code, device_id, device_name))
        return "secret-account-token"

    def get_entitlements(self):
        self.calls.append(("get_entitlements",))
        if isinstance(self.entitlement, Exception):
            raise self.entitlement
        return self.entitlement

    def revoke_token(self):
        self.calls.append(("revoke_token",))
        if self.before_revoke:
            self.before_revoke()
        if self.revoke_error:
            raise self.revoke_error
        return {"revoked": True}

    def close(self):
        self.closed = True


@pytest.fixture
def account_paths(tmp_path, monkeypatch):
    env_path = tmp_path / "config" / ".env"
    device_path = tmp_path / "data" / "device_id"
    cache_path = tmp_path / "data" / "entitlements.json"
    monkeypatch.setattr("ormah.setup.ENV_PATH", env_path)
    monkeypatch.setattr("ormah.setup.ENV_DIR", env_path.parent)
    monkeypatch.setattr(cloud_client, "DEVICE_ID_PATH", device_path)
    monkeypatch.setattr(entitlements, "ENTITLEMENT_CACHE_PATH", cache_path)
    monkeypatch.setattr(settings, "account_token", None)
    monkeypatch.setattr(settings, "account_email", None)
    monkeypatch.setattr(settings, "cloud_api_url", "https://cloud.test")
    return env_path, device_path, cache_path


def _run(argv, client):
    with (
        patch("sys.argv", ["ormah", *argv]),
        patch("ormah.cli._cloud_client", return_value=client),
    ):
        cli.main()


def test_login_persists_credentials_without_rewriting_unrelated_lines(
    account_paths, capsys
):
    env_path, device_path, cache_path = account_paths
    original = "# user comment\nMANUAL = spaced  # keep exactly\n\nORMAH_LLM_PROVIDER=none\n"
    env_path.parent.mkdir(parents=True)
    env_path.write_text(original)
    fake = FakeClient()

    _run(
        ["account", "login", "--email", "Person@Example.com", "--code", "123456"],
        fake,
    )

    text = env_path.read_text()
    assert text.startswith(original)
    assert "ORMAH_ACCOUNT_TOKEN=secret-account-token\n" in text
    assert "ORMAH_ACCOUNT_EMAIL=person@example.com\n" in text
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(device_path.stat().st_mode) == 0o600
    assert cache_path.is_file()
    assert [call[0] for call in fake.calls] == [
        "request_code",
        "verify_code",
        "get_entitlements",
    ]
    verify_call = fake.calls[1]
    assert uuid.UUID(verify_call[3]).version == 4
    assert verify_call[4]
    output = capsys.readouterr().out
    assert "Plan status: active" in output
    assert "secret-account-token" not in output


def test_login_keeps_credentials_when_entitlement_refresh_is_offline(
    account_paths, capsys
):
    env_path, _, cache_path = account_paths
    fake = FakeClient(entitlement=CloudError("offline"))

    _run(
        ["account", "login", "--email", "person@example.com", "--code", "123456"],
        fake,
    )

    assert "ORMAH_ACCOUNT_TOKEN=secret-account-token" in env_path.read_text()
    assert not cache_path.exists()
    assert "unavailable while offline" in capsys.readouterr().out


@pytest.mark.parametrize("offline", [False, True])
def test_logout_revokes_before_local_deletion_and_preserves_other_keys(
    account_paths, capsys, monkeypatch, offline
):
    env_path, _, cache_path = account_paths
    original = (
        "# header\n"
        "MANUAL = spaced  # keep exactly\n"
        "ORMAH_ACCOUNT_TOKEN=secret-account-token\n"
        "ORMAH_X=1\n"
        "ORMAH_ACCOUNT_EMAIL=person@example.com\n"
        "# footer\n"
    )
    env_path.parent.mkdir(parents=True)
    env_path.write_text(original)
    entitlements.cache_entitlements(
        {"backup": True, "plan_status": "active"},
    )
    monkeypatch.setattr(settings, "account_token", "secret-account-token")
    monkeypatch.setattr(settings, "account_email", "person@example.com")

    def assert_local_state_still_exists():
        assert "ORMAH_ACCOUNT_TOKEN" in env_path.read_text()
        assert cache_path.is_file()

    fake = FakeClient(
        revoke_error=CloudError("offline") if offline else None,
        before_revoke=assert_local_state_still_exists,
    )

    _run(["account", "logout", "--yes"], fake)

    assert fake.calls[0] == ("revoke_token",)
    assert env_path.read_text() == (
        "# header\n"
        "MANUAL = spaced  # keep exactly\n"
        "ORMAH_X=1\n"
        "# footer\n"
    )
    assert not cache_path.exists()
    output = capsys.readouterr().out
    assert "Signed out locally" in output
    assert ("Could not revoke" in output) is offline


def test_status_json_contains_no_token(account_paths, capsys, monkeypatch):
    _, _, _ = account_paths
    entitlements.cache_entitlements(
        {"backup": True, "founding": True, "plan_status": "trialing"}
    )
    monkeypatch.setattr(settings, "account_token", "secret-account-token")
    monkeypatch.setattr(settings, "account_email", "person@example.com")

    _run(["account", "status", "--json"], FakeClient())

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["email"] == "person@example.com"
    assert payload["entitlement"] == "active"
    assert payload["plan_status"] == "trialing"
    assert payload["device_name"]
    assert "token" not in output.lower()


def test_logout_requires_confirmation_in_noninteractive_shell(account_paths, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        _run(["account", "logout"], FakeClient())
    assert "--yes" in capsys.readouterr().err


def test_logout_still_clears_locally_when_client_construction_fails(
    account_paths, monkeypatch, capsys
):
    env_path, _, cache_path = account_paths
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "KEEP=exact\n"
        "ORMAH_ACCOUNT_TOKEN=secret-account-token\n"
        "ORMAH_ACCOUNT_EMAIL=person@example.com\n"
    )
    entitlements.cache_entitlements({"backup": True})
    monkeypatch.setattr(settings, "account_token", "secret-account-token")
    monkeypatch.setattr(settings, "account_email", "person@example.com")

    with (
        patch("sys.argv", ["ormah", "account", "logout", "--yes"]),
        patch("ormah.cli._cloud_client", side_effect=ValueError("bad URL")),
    ):
        cli.main()

    assert env_path.read_text() == "KEEP=exact\n"
    assert not cache_path.exists()
    output = capsys.readouterr().out
    assert "Could not revoke" in output
    assert "Signed out locally" in output


def test_account_settings_are_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("ORMAH_ACCOUNT_TOKEN", "configured-token")
    monkeypatch.setenv("ORMAH_ACCOUNT_EMAIL", "person@example.com")
    configured = Settings(memory_dir="/tmp/ormah-e04-config-test")
    assert configured.account_token == "configured-token"
    assert configured.account_email == "person@example.com"
