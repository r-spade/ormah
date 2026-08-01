"""CLI tests for the `ormah cloud` group."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import patch

import pytest

from ormah import cli
from ormah.cloud import keys as cloud_keys
from ormah.cloud import state as cloud_state
from ormah.cloud.state import CloudState, load_state, save_state


@pytest.fixture
def cloud_paths(tmp_path, monkeypatch):
    """Point every cloud path at tmp and return the key path."""
    key_path = tmp_path / "config" / "cloud.key"
    kit_path = tmp_path / "config" / "ormah-recovery-kit.md"
    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(cloud_keys, "KEY_PATH", key_path)
    monkeypatch.setattr(cloud_keys, "RECOVERY_KIT_PATH", kit_path)
    monkeypatch.setattr(cloud_state, "CLOUD_STATE_DIR", tmp_path / "cloud-state")
    from ormah.config import settings

    monkeypatch.setattr(settings, "memory_dir", memory_dir)
    monkeypatch.setattr(settings, "account_email", None)
    return key_path, kit_path, memory_dir


def _run(argv):
    with patch("sys.argv", ["ormah", *argv]):
        cli.main()


def test_cloud_init_json(cloud_paths, capsys):
    key_path, kit_path, memory_dir = cloud_paths

    _run(["cloud", "init", "--json"])

    out = json.loads(capsys.readouterr().out)
    assert out["key_path"] == str(key_path)
    assert out["identity_count"] == 1
    assert out["imported"] is False
    assert out["recovery_kit"] == str(kit_path)
    assert key_path.is_file()
    assert kit_path.is_file()
    assert (memory_dir / ".store_id").is_file()
    assert out["store_id"] == (memory_dir / ".store_id").read_text().strip()


def test_cloud_init_writes_signed_in_email_to_recovery_kit(
    cloud_paths, capsys, monkeypatch
):
    _, kit_path, _ = cloud_paths
    from ormah.config import settings

    monkeypatch.setattr(settings, "account_email", "person@example.com")

    _run(["cloud", "init", "--json"])

    assert "Email: person@example.com" in kit_path.read_text(encoding="utf-8")


def test_cloud_init_refuses_second_run(cloud_paths, capsys):
    _run(["cloud", "init", "--json"])
    with pytest.raises(SystemExit):
        _run(["cloud", "init", "--json"])
    assert "rotate-key" in capsys.readouterr().err


def test_cloud_init_import_key(cloud_paths, tmp_path, capsys):
    key_path, kit_path, _ = cloud_paths
    _run(["cloud", "init", "--json"])
    original = cloud_keys.load_identity_strings(key_path)
    kit_copy = tmp_path / "kit-copy.md"
    kit_copy.write_text(kit_path.read_text())

    # fresh machine: move real key aside
    key_path.rename(key_path.with_suffix(".bak"))
    capsys.readouterr()

    _run(["cloud", "init", "--import-key", str(kit_copy), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["imported"] is True
    assert cloud_keys.load_identity_strings(key_path) == original


def test_cloud_rotate_key_json(cloud_paths, capsys):
    key_path, kit_path, _ = cloud_paths
    _run(["cloud", "init", "--json"])
    first = cloud_keys.load_identity_strings(key_path)
    capsys.readouterr()

    _run(["cloud", "rotate-key", "--yes", "--json"])

    out = json.loads(capsys.readouterr().out)
    assert out["rotated"] is True
    assert out["identity_count"] == 2
    strings = cloud_keys.load_identity_strings(key_path)
    assert strings[1:] == first  # old identity retained
    assert strings[0] in kit_path.read_text()  # kit regenerated with new key


def test_cloud_rotate_key_clears_recovery_readiness(cloud_paths, capsys):
    _, _, memory_dir = cloud_paths
    _run(["cloud", "init", "--json"])
    store_id = (memory_dir / ".store_id").read_text(encoding="utf-8").strip()
    save_state(
        store_id,
        CloudState(recovery_kit_verified_at=datetime.now(timezone.utc)),
        memory_dir=memory_dir,
    )
    capsys.readouterr()

    _run(["cloud", "rotate-key", "--yes", "--json"])

    assert load_state(store_id).recovery_kit_verified_at is None


def test_cloud_rotate_key_requires_confirmation_non_tty(cloud_paths, capsys, monkeypatch):
    _run(["cloud", "init", "--json"])
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit):
        _run(["cloud", "rotate-key"])
    assert "--yes" in capsys.readouterr().err


def test_cloud_rotate_key_without_init_fails(cloud_paths, capsys):
    with pytest.raises(SystemExit):
        _run(["cloud", "rotate-key", "--yes"])
    assert "cloud init" in capsys.readouterr().err


def test_cloud_kit_regenerates_after_loss(cloud_paths, capsys):
    """`ormah cloud kit` is the recovery path when init/rotate is interrupted
    between key commit and kit generation."""
    key_path, kit_path, memory_dir = cloud_paths
    _run(["cloud", "init", "--json"])
    kit_path.unlink()  # simulate the stranded state
    capsys.readouterr()

    _run(["cloud", "kit", "--json"])

    out = json.loads(capsys.readouterr().out)
    assert out["recovery_kit"] == str(kit_path)
    assert out["identity_count"] == 1
    assert kit_path.is_file()
    current = cloud_keys.load_identity_strings(key_path)[0]
    assert current in kit_path.read_text()


def test_cloud_kit_writes_signed_in_email(cloud_paths, capsys, monkeypatch):
    _, kit_path, _ = cloud_paths
    from ormah.config import settings

    _run(["cloud", "init", "--json"])
    capsys.readouterr()
    monkeypatch.setattr(settings, "account_email", "person@example.com")

    _run(["cloud", "kit", "--json"])

    assert "Email: person@example.com" in kit_path.read_text(encoding="utf-8")


def test_cloud_kit_without_key_fails(cloud_paths, capsys):
    with pytest.raises(SystemExit):
        _run(["cloud", "kit", "--json"])
    assert "cloud init" in capsys.readouterr().err


def test_import_key_preserves_store_id(cloud_paths, tmp_path, capsys):
    """Fresh-machine import must adopt the kit's store id, not mint a new one
    — the store id is the remote namespace for all existing backups."""
    key_path, kit_path, memory_dir = cloud_paths
    _run(["cloud", "init", "--json"])
    original_store_id = json.loads(capsys.readouterr().out)["store_id"]
    kit_copy = tmp_path / "kit-copy.md"
    kit_copy.write_text(kit_path.read_text())

    # Fresh machine: no key, no store id
    key_path.rename(key_path.with_suffix(".bak"))
    (memory_dir / ".store_id").unlink()

    _run(["cloud", "init", "--import-key", str(kit_copy), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["store_id"] == original_store_id
    assert (memory_dir / ".store_id").read_text().strip() == original_store_id


def test_import_key_refuses_store_id_conflict(cloud_paths, tmp_path, capsys):
    key_path, kit_path, memory_dir = cloud_paths
    _run(["cloud", "init", "--json"])
    kit_copy = tmp_path / "kit-copy.md"
    kit_copy.write_text(kit_path.read_text())

    key_path.rename(key_path.with_suffix(".bak"))
    (memory_dir / ".store_id").write_text("99999999-8888-7777-6666-555555555555\n")
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _run(["cloud", "init", "--import-key", str(kit_copy)])
    err = capsys.readouterr().err
    assert "orphan" in err
    # Key material untouched by the aborted import
    assert not key_path.exists()


def test_import_key_aborts_on_damaged_kit_store_id(cloud_paths, tmp_path, capsys):
    """A damaged store_id line must abort the whole import before any key
    material is written — never silently mint a new namespace."""
    key_path, kit_path, memory_dir = cloud_paths
    _run(["cloud", "init", "--json"])
    damaged = tmp_path / "damaged-kit.md"
    damaged.write_text(kit_path.read_text().replace(
        "store_id: ", "store_id: corrupted-"
    ))
    key_path.rename(key_path.with_suffix(".bak"))
    (memory_dir / ".store_id").unlink()
    capsys.readouterr()

    with pytest.raises(SystemExit):
        _run(["cloud", "init", "--import-key", str(damaged)])

    assert "malformed store id" in capsys.readouterr().err
    assert not key_path.exists()  # no key material written
    assert not (memory_dir / ".store_id").exists()  # no new namespace minted
