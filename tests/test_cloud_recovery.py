from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from ormah.cloud import recovery
from ormah.cloud.keys import (
    get_or_create_store_id,
    init_key,
    load_identity_strings,
    write_recovery_kit,
)
from ormah.cloud.recovery import RecoveryKitError, RecoveryKitService
from ormah.cloud.state import CloudState, ProtectionState, load_state, save_state


NOW = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def recovery_store(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    key_path = tmp_path / "config" / "cloud.key"
    kit_path = tmp_path / "config" / "ormah-recovery-kit.md"
    state_dir = tmp_path / "state"
    settings = SimpleNamespace(memory_dir=memory_dir, cloud_backup_enabled=True)
    store_id = get_or_create_store_id(memory_dir)
    init_key(key_path)
    write_recovery_kit(store_id, key_path=key_path, kit_path=kit_path)
    save_state(
        store_id,
        CloudState(
            protection_state=ProtectionState.PROTECTED,
            last_successful_backup_snapshot_id="snapshot-a",
            last_verified_snapshot_id="snapshot-a",
            last_verify_ok=True,
        ),
        memory_dir=memory_dir,
        state_dir=state_dir,
    )
    service = RecoveryKitService(
        settings,
        key_path=key_path,
        kit_path=kit_path,
        state_dir=state_dir,
        now=lambda: NOW,
    )
    return settings, store_id, key_path, kit_path, state_dir, service


def test_current_reopened_digest_updates_state_only_after_success(recovery_store):
    settings, store_id, _, kit_path, state_dir, service = recovery_store
    digest = hashlib.sha256(kit_path.read_bytes()).hexdigest()

    result = service.confirm_saved_digest(digest)

    assert result.device_loss_recovery_ready is True
    assert result.recovery_kit_verified_at == NOW
    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at == NOW
    assert settings.memory_dir.is_dir()


def test_wrong_digest_fails_without_updating_state(recovery_store):
    _, store_id, _, _, state_dir, service = recovery_store

    with pytest.raises(RecoveryKitError, match="could not be verified"):
        service.confirm_saved_digest("0" * 64)

    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at is None


def test_ensure_current_kit_keeps_valid_material_and_existing_readiness(recovery_store):
    _, store_id, _, kit_path, state_dir, service = recovery_store
    original = kit_path.read_bytes()
    service.confirm_saved_digest(hashlib.sha256(original).hexdigest())

    assert service.ensure_current_kit() is False

    assert kit_path.read_bytes() == original
    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at == NOW


def test_ensure_current_kit_repairs_stale_material_before_native_save(recovery_store):
    settings, store_id, key_path, kit_path, state_dir, service = recovery_store
    service.confirm_saved_digest(hashlib.sha256(kit_path.read_bytes()).hexdigest())
    replacement_key = key_path.with_suffix(".replacement")
    init_key(replacement_key)
    key_path.write_bytes(replacement_key.read_bytes())
    settings.account_email = "person@example.com"

    assert service.ensure_current_kit() is True

    service.validate_canonical_kit()
    assert "person@example.com" in kit_path.read_text(encoding="utf-8")
    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at is None


def test_saved_copy_proof_does_not_claim_ready_without_current_protection(
    recovery_store,
):
    settings, store_id, _, kit_path, state_dir, service = recovery_store
    settings.cloud_backup_enabled = False

    result = service.confirm_saved_digest(hashlib.sha256(kit_path.read_bytes()).hexdigest())

    assert result.device_loss_recovery_ready is False
    assert result.recovery_kit_verified_at == NOW
    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at == NOW


@pytest.mark.parametrize("damage", ["store", "key", "kit"])
def test_wrong_store_key_or_kit_fails_closed(recovery_store, damage):
    _, store_id, key_path, kit_path, state_dir, service = recovery_store
    if damage == "store":
        text = kit_path.read_text(encoding="utf-8")
        kit_path.write_text(
            text.replace(store_id, str(uuid.uuid4())),
            encoding="utf-8",
        )
    elif damage == "key":
        init_key(key_path.with_suffix(".new"))
        key_path.write_bytes(key_path.with_suffix(".new").read_bytes())
    else:
        kit_path.write_text("not a recovery kit\n", encoding="utf-8")
    digest = hashlib.sha256(kit_path.read_bytes()).hexdigest()

    with pytest.raises(RecoveryKitError):
        service.confirm_saved_digest(digest)

    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at is None


@pytest.mark.parametrize(
    "replacement",
    ["", "format_version: 2"],
    ids=["missing", "unsupported"],
)
def test_unversioned_or_unsupported_kit_cannot_be_confirmed(
    recovery_store,
    replacement,
):
    _, store_id, _, kit_path, state_dir, service = recovery_store
    kit_text = kit_path.read_text(encoding="utf-8")
    kit_path.write_text(
        kit_text.replace("format_version: 1", replacement),
        encoding="utf-8",
    )

    with pytest.raises(RecoveryKitError, match="not current"):
        service.confirm_saved_digest(hashlib.sha256(kit_path.read_bytes()).hexdigest())

    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at is None


def test_malformed_digest_is_rejected_before_kit_work(recovery_store, monkeypatch):
    *_, service = recovery_store
    calls = []
    monkeypatch.setattr(
        service,
        "_validate_canonical_kit_locked",
        lambda: calls.append(True),
    )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        service.confirm_saved_digest("ABC")

    assert calls == []


def test_confirmation_never_mints_a_missing_store_identity(recovery_store):
    settings, _, _, kit_path, _, service = recovery_store
    store_marker = settings.memory_dir / ".store_id"
    store_marker.unlink()

    with pytest.raises(RecoveryKitError):
        service.confirm_saved_digest(hashlib.sha256(kit_path.read_bytes()).hexdigest())

    assert not store_marker.exists()


def test_rotation_clears_readiness_and_regenerates_current_kit(recovery_store):
    _, store_id, key_path, kit_path, state_dir, service = recovery_store
    digest = hashlib.sha256(kit_path.read_bytes()).hexdigest()
    service.confirm_saved_digest(digest)
    identities_before = load_identity_strings(key_path)

    service.rotate_current_key()

    state = load_state(store_id, state_dir=state_dir)
    identities_after = load_identity_strings(key_path)
    assert state.recovery_kit_verified_at is None
    assert identities_after[1:] == identities_before
    assert identities_after[0] in kit_path.read_text(encoding="utf-8")


def test_rotation_state_failure_leaves_key_and_kit_untouched(
    recovery_store,
    monkeypatch,
):
    _, _, key_path, kit_path, _, service = recovery_store
    key_before = key_path.read_bytes()
    kit_before = kit_path.read_bytes()

    def fail_state(*args, **kwargs):
        raise RuntimeError("state unavailable")

    monkeypatch.setattr(recovery, "update_state", fail_state)

    with pytest.raises(RuntimeError, match="state unavailable"):
        service.rotate_current_key()

    assert key_path.read_bytes() == key_before
    assert kit_path.read_bytes() == kit_before


def test_partially_applied_rotation_is_not_confirmable(
    recovery_store,
    monkeypatch,
):
    _, store_id, key_path, kit_path, state_dir, service = recovery_store
    service.confirm_saved_digest(hashlib.sha256(kit_path.read_bytes()).hexdigest())
    active_identity = load_identity_strings(key_path)[0]
    atomic_write = recovery.cloud_keys._atomic_write_0600

    def fail_key(path, text):
        if path == key_path:
            raise OSError("key unavailable")
        atomic_write(path, text)

    monkeypatch.setattr(recovery.cloud_keys, "_atomic_write_0600", fail_key)

    with pytest.raises(OSError, match="key unavailable"):
        service.rotate_current_key()

    assert active_identity in kit_path.read_text(encoding="utf-8")
    assert load_state(store_id, state_dir=state_dir).recovery_kit_verified_at is None
    with pytest.raises(RecoveryKitError, match="not current"):
        service.validate_canonical_kit()


def test_canonical_read_is_bounded(recovery_store):
    _, _, _, kit_path, _, service = recovery_store
    kit_path.write_bytes(b"x" * (recovery.MAX_RECOVERY_KIT_BYTES + 1))

    with pytest.raises(RecoveryKitError, match="invalid"):
        service.validate_canonical_kit()
