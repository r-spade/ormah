"""Tests for cloud key lifecycle, store_id, and the recovery kit."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import uuid

import pytest

from ormah.cloud import keys
from ormah.cloud.bundle import build_bundle, open_bundle
from ormah.cloud.crypto import encrypt_bytes, decrypt_bytes
from ormah.cloud.keys import (
    CloudKeyError,
    _rotate_key_without_recovery_kit,
    apply_recovery_import,
    current_recipient,
    get_or_create_store_id,
    import_key,
    init_key,
    key_file_exists,
    load_identities,
    load_identity_strings,
    plan_recovery_import,
    rotate_key_and_recovery_kit,
    write_recovery_kit,
)


@pytest.fixture
def key_path(tmp_path):
    return tmp_path / "config" / "cloud.key"


# --- init ---


def test_init_creates_key_with_0600(key_path):
    identity_str = init_key(key_path)
    assert identity_str.startswith("AGE-SECRET-KEY-")
    assert key_file_exists(key_path)
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert load_identity_strings(key_path) == [identity_str]


def test_init_refuses_overwrite_and_names_rotate_key(key_path):
    init_key(key_path)
    with pytest.raises(CloudKeyError, match="rotate-key"):
        init_key(key_path)


# --- rotation ---


def test_rotate_retains_old_identities_and_old_bundles_decrypt(key_path):
    init_key(key_path)
    old_recipient = current_recipient(key_path)
    old_bundle = encrypt_bytes(b"pre-rotation", [old_recipient])

    new_str = _rotate_key_without_recovery_kit(key_path)

    strings = load_identity_strings(key_path)
    assert strings[0] == new_str
    assert len(strings) == 2
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600

    # Pre-rotation data decrypts with the full identity list
    assert decrypt_bytes(old_bundle, load_identities(key_path)) == b"pre-rotation"
    # New encryptions use the new current identity
    new_bundle = encrypt_bytes(b"post-rotation", [current_recipient(key_path)])
    assert decrypt_bytes(new_bundle, load_identities(key_path)) == b"post-rotation"


def test_double_rotation_keeps_all_three(key_path):
    first = init_key(key_path)
    second = _rotate_key_without_recovery_kit(key_path)
    third = _rotate_key_without_recovery_kit(key_path)
    assert load_identity_strings(key_path) == [third, second, first]


def test_rotate_without_key_fails(key_path):
    with pytest.raises(CloudKeyError, match="cloud init"):
        _rotate_key_without_recovery_kit(key_path)


def test_recovery_first_rotation_leaves_key_untouched_when_kit_write_fails(
    tmp_path, key_path, monkeypatch
):
    original_identity = init_key(key_path)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        key_path,
        tmp_path / "kit.md",
    )
    key_before = key_path.read_bytes()
    kit_before = kit_path.read_bytes()
    atomic_write = keys._atomic_write_0600

    def fail_kit(path, text):
        if path == kit_path:
            raise OSError("kit unavailable")
        atomic_write(path, text)

    monkeypatch.setattr(keys, "_atomic_write_0600", fail_kit)

    with pytest.raises(OSError, match="kit unavailable"):
        rotate_key_and_recovery_kit(
            "22222222-3333-4444-9555-666666666666",
            key_path=key_path,
            kit_path=kit_path,
        )

    assert key_path.read_bytes() == key_before
    assert kit_path.read_bytes() == kit_before
    assert load_identity_strings(key_path) == [original_identity]


def test_recovery_first_rotation_keeps_active_key_recoverable_when_key_write_fails(
    tmp_path, key_path, monkeypatch
):
    original_identity = init_key(key_path)
    kit_path = tmp_path / "kit.md"
    key_before = key_path.read_bytes()
    atomic_write = keys._atomic_write_0600

    def fail_key(path, text):
        if path == key_path:
            raise OSError("key unavailable")
        atomic_write(path, text)

    monkeypatch.setattr(keys, "_atomic_write_0600", fail_key)

    with pytest.raises(OSError, match="key unavailable"):
        rotate_key_and_recovery_kit(
            "22222222-3333-4444-9555-666666666666",
            key_path=key_path,
            kit_path=kit_path,
        )

    assert key_path.read_bytes() == key_before
    kit_identities = [
        line.strip()
        for line in kit_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("AGE-SECRET-KEY-")
    ]
    assert kit_identities[1:] == [original_identity]


def test_older_client_never_overwrites_newer_recovery_kit(tmp_path, key_path):
    original_identity = init_key(key_path)
    store_id = "22222222-3333-4444-9555-666666666666"
    kit_path = write_recovery_kit(store_id, key_path, tmp_path / "kit.md")
    newer_kit = kit_path.read_text(encoding="utf-8").replace(
        "format_version: 1",
        "format_version: 2\nfuture_recovery_token: opaque",
    )
    kit_path.write_text(newer_kit, encoding="utf-8")
    key_before = key_path.read_bytes()

    with pytest.raises(CloudKeyError, match="newer Ormah version"):
        write_recovery_kit(store_id, key_path, kit_path)
    with pytest.raises(CloudKeyError, match="newer Ormah version"):
        rotate_key_and_recovery_kit(store_id, key_path=key_path, kit_path=kit_path)

    assert kit_path.read_text(encoding="utf-8") == newer_kit
    assert key_path.read_bytes() == key_before
    assert load_identity_strings(key_path) == [original_identity]


@pytest.mark.parametrize(
    "format_text",
    [
        "format_version: invalid",
        "format_version: 0",
        "format_version: 1\nformat_version: 1",
    ],
)
def test_malformed_existing_kit_is_never_overwritten(
    tmp_path,
    key_path,
    format_text,
):
    init_key(key_path)
    kit_path = tmp_path / "kit.md"
    damaged = f"# Ormah Recovery Kit\n{format_text}\nfuture_field: preserve-me\n"
    kit_path.write_text(damaged, encoding="utf-8")

    with pytest.raises(CloudKeyError, match="format version"):
        write_recovery_kit(
            "22222222-3333-4444-9555-666666666666",
            key_path,
            kit_path,
        )

    assert kit_path.read_text(encoding="utf-8") == damaged


def test_non_utf8_oversized_or_non_file_kit_is_never_overwritten(tmp_path, key_path):
    init_key(key_path)
    store_id = "22222222-3333-4444-9555-666666666666"

    non_utf8 = tmp_path / "non-utf8.md"
    non_utf8.write_bytes(b"format_version: 1\n\xff")
    with pytest.raises(CloudKeyError, match="invalid"):
        write_recovery_kit(store_id, key_path, non_utf8)

    oversized = tmp_path / "oversized.md"
    oversized.write_bytes(b"x" * (keys.MAX_RECOVERY_KIT_BYTES + 1))
    with pytest.raises(CloudKeyError, match="invalid"):
        write_recovery_kit(store_id, key_path, oversized)

    directory = tmp_path / "directory.md"
    directory.mkdir()
    with pytest.raises(CloudKeyError, match="invalid"):
        write_recovery_kit(store_id, key_path, directory)


@pytest.mark.parametrize("separator", ["\n", "\x85", "\u2028", "\u2029"])
def test_recovery_kit_refuses_multiline_account_email(
    tmp_path,
    key_path,
    separator,
):
    init_key(key_path)

    with pytest.raises(CloudKeyError, match="account email"):
        write_recovery_kit(
            "22222222-3333-4444-9555-666666666666",
            key_path,
            tmp_path / "kit.md",
            account_email=f"person@example.com{separator}format_version: 99",
        )


# --- import ---


def test_import_key_roundtrip_from_kit(tmp_path, key_path):
    init_key(key_path)
    _rotate_key_without_recovery_kit(key_path)
    originals = load_identity_strings(key_path)
    kit = write_recovery_kit("store-1", key_path, tmp_path / "kit.md")

    fresh_key_path = tmp_path / "fresh" / "cloud.key"
    imported = import_key(str(kit), fresh_key_path)

    assert imported == originals
    assert load_identity_strings(fresh_key_path) == originals
    assert stat.S_IMODE(fresh_key_path.stat().st_mode) == 0o600


def test_import_key_from_pasted_text(tmp_path, key_path):
    original = init_key(key_path)
    fresh = tmp_path / "fresh.key"
    assert import_key(f"junk\n{original}\nmore junk", fresh) == [original]


def test_import_refuses_existing_key(tmp_path, key_path):
    init_key(key_path)
    with pytest.raises(CloudKeyError, match="refusing to overwrite"):
        import_key("AGE-SECRET-KEY-1WHATEVER", key_path)


def test_import_rejects_material_without_identities(tmp_path):
    with pytest.raises(CloudKeyError, match="No age identities"):
        import_key("nothing to see here", tmp_path / "fresh.key")


def test_import_validates_before_writing(tmp_path):
    fresh = tmp_path / "fresh.key"
    with pytest.raises(Exception):
        import_key("AGE-SECRET-KEY-1NOTREAL", fresh)
    assert not fresh.exists()


def test_recovery_import_rejects_duplicate_kit_identities_before_writing(key_path):
    identity = init_key(key_path)
    fresh_key = key_path.parent / "fresh.key"
    memory_dir = key_path.parent / "memory"

    with pytest.raises(CloudKeyError, match="duplicate age identities"):
        plan_recovery_import(
            f"{identity}\n{identity}\nstore_id: 22222222-3333-4444-9555-666666666666\n",
            memory_dir,
            fresh_key,
        )

    assert not fresh_key.exists()
    assert not (memory_dir / ".store_id").exists()


def test_recovery_import_reads_source_file_once(tmp_path, key_path, monkeypatch):
    init_key(key_path)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        key_path,
        tmp_path / "kit.md",
    )
    original_read_text = Path.read_text
    reads = 0

    def count_source_reads(path, *args, **kwargs):
        nonlocal reads
        if path == kit_path:
            reads += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", count_source_reads)

    plan_recovery_import(str(kit_path), tmp_path / "memory", key_path)

    assert reads == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior")
def test_recovery_import_rejects_symlinked_existing_key(tmp_path, key_path):
    init_key(key_path)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        key_path,
        tmp_path / "kit.md",
    )
    linked_key = tmp_path / "linked.key"
    linked_key.symlink_to(key_path)

    with pytest.raises(CloudKeyError, match="must not be a symlink"):
        plan_recovery_import(str(kit_path), tmp_path / "memory", linked_key)

    assert linked_key.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink behavior")
def test_recovery_import_rejects_symlinked_existing_store(tmp_path, key_path):
    init_key(key_path)
    store_id = "22222222-3333-4444-9555-666666666666"
    kit_path = write_recovery_kit(store_id, key_path, tmp_path / "kit.md")
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    target = tmp_path / "outside-store-id"
    target.write_text(store_id + "\n", encoding="utf-8")
    (memory_dir / ".store_id").symlink_to(target)

    with pytest.raises(CloudKeyError, match="must not be a symlink"):
        plan_recovery_import(str(kit_path), memory_dir, tmp_path / "fresh.key")

    assert target.read_text(encoding="utf-8") == store_id + "\n"


def test_recovery_import_rejects_malformed_existing_store(tmp_path, key_path):
    init_key(key_path)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        key_path,
        tmp_path / "kit.md",
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    store_path = memory_dir / ".store_id"
    store_path.write_text("not-a-uuid\n", encoding="utf-8")
    key_before = key_path.read_bytes()

    with pytest.raises(CloudKeyError, match="Corrupt store id"):
        plan_recovery_import(str(kit_path), memory_dir, key_path)

    assert store_path.read_text(encoding="utf-8") == "not-a-uuid\n"
    assert key_path.read_bytes() == key_before


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission behavior")
def test_recovery_import_rejects_broad_existing_key_permissions(tmp_path, key_path):
    init_key(key_path)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        key_path,
        tmp_path / "kit.md",
    )
    key_path.chmod(0o644)

    with pytest.raises(CloudKeyError, match="chmod 600"):
        plan_recovery_import(str(kit_path), tmp_path / "memory", key_path)

    assert stat.S_IMODE(key_path.stat().st_mode) == 0o644


def test_recovery_import_wraps_malformed_existing_key(tmp_path, key_path):
    source_key = tmp_path / "source.key"
    init_key(source_key)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        source_key,
        tmp_path / "kit.md",
    )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text("AGE-SECRET-KEY-1NOTREAL\n", encoding="utf-8")
    key_path.chmod(0o600)

    with pytest.raises(CloudKeyError, match="existing cloud key is invalid"):
        plan_recovery_import(str(kit_path), tmp_path / "memory", key_path)

    assert key_path.read_text(encoding="utf-8") == "AGE-SECRET-KEY-1NOTREAL\n"


def test_recovery_import_rejects_duplicate_existing_key(tmp_path, key_path):
    source_key = tmp_path / "source.key"
    identity = init_key(source_key)
    kit_path = write_recovery_kit(
        "22222222-3333-4444-9555-666666666666",
        source_key,
        tmp_path / "kit.md",
    )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(f"{identity}\n{identity}\n", encoding="utf-8")
    key_path.chmod(0o600)

    with pytest.raises(CloudKeyError, match="duplicate identities"):
        plan_recovery_import(str(kit_path), tmp_path / "memory", key_path)

    assert key_path.read_text(encoding="utf-8") == f"{identity}\n{identity}\n"


def test_apply_recovery_import_installs_store_and_key(tmp_path, key_path):
    source_key = tmp_path / "source.key"
    identities = [init_key(source_key)]
    store_id = "22222222-3333-4444-9555-666666666666"
    kit_path = write_recovery_kit(store_id, source_key, tmp_path / "kit.md")
    memory_dir = tmp_path / "memory"

    result = apply_recovery_import(
        plan_recovery_import(str(kit_path), memory_dir, key_path)
    )

    assert result.store_id_status == "installed"
    assert result.key_status == "installed"
    assert load_identity_strings(key_path) == identities
    assert (memory_dir / ".store_id").read_text(encoding="utf-8").strip() == store_id
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((memory_dir / ".store_id").stat().st_mode) == 0o600


# --- store_id ---


def test_store_id_created_and_stable(tmp_path):
    memory_dir = tmp_path / "memory"
    first = get_or_create_store_id(memory_dir)
    assert uuid.UUID(first)
    assert (memory_dir / ".store_id").read_text().strip() == first
    assert get_or_create_store_id(memory_dir) == first


def test_store_id_corrupt_raises(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / ".store_id").write_text("not-a-uuid\n")
    with pytest.raises(CloudKeyError, match="Corrupt store id"):
        get_or_create_store_id(memory_dir)


# --- recovery kit ---


def test_recovery_kit_contains_everything_and_no_passphrase(tmp_path, key_path):
    init_key(key_path)
    _rotate_key_without_recovery_kit(key_path)
    strings = load_identity_strings(key_path)

    kit_path = write_recovery_kit(
        "22222222-3333-4444-5555-666666666666", key_path, tmp_path / "kit.md",
        account_email="rishi@example.com",
    )

    text = kit_path.read_text()
    for s in strings:
        assert s in text
    assert "22222222-3333-4444-5555-666666666666" in text
    assert "rishi@example.com" in text
    assert "format_version: 1" in text
    assert "ormah cloud init --import-key" in text
    assert "ormah backup restore --cloud" in text
    assert "including us" in text
    assert "passphrase" not in text.lower()
    assert stat.S_IMODE(kit_path.stat().st_mode) == 0o600


def test_kit_keys_open_real_bundle(tmp_path, key_path):
    """End-to-end: a bundle encrypted to the current key opens with identities
    re-imported from the recovery kit on a 'fresh machine'."""
    init_key(key_path)
    store_id = get_or_create_store_id(tmp_path / "memory")

    backup = tmp_path / "backup"
    (backup / "nodes").mkdir(parents=True)
    (backup / "nodes" / "fact_x_aaa111.md").write_text("---\nid: aaa111\n---\nX.\n")

    bundle = build_bundle(
        backup, tmp_path / "b.age", [current_recipient(key_path)], store_id=store_id
    )

    kit = write_recovery_kit(store_id, key_path, tmp_path / "kit.md")
    fresh_key = tmp_path / "fresh" / "cloud.key"
    import_key(str(kit), fresh_key)

    info = open_bundle(bundle, tmp_path / "restored", load_identities(fresh_key))
    assert info.store_id == store_id
    assert (tmp_path / "restored" / "nodes" / "fact_x_aaa111.md").read_text().endswith("X.\n")


# --- store_id preservation on import (Codex re-review) ---


def test_extract_store_id_from_kit(tmp_path, key_path):
    init_key(key_path)
    from ormah.cloud.keys import extract_store_id

    kit = write_recovery_kit("33333333-4444-4555-8666-777777777777", key_path, tmp_path / "kit.md")
    assert extract_store_id(str(kit)) == "33333333-4444-4555-8666-777777777777"


def test_extract_store_id_legacy_bare_uuid(tmp_path):
    from ormah.cloud.keys import extract_store_id

    legacy = "## Your store id\n\n```\n44444444-5555-4666-8777-888888888888\n```\n"
    assert extract_store_id(legacy) == "44444444-5555-4666-8777-888888888888"


def test_extract_store_id_absent(tmp_path):
    from ormah.cloud.keys import extract_store_id

    assert extract_store_id("AGE-SECRET-KEY-1FOO\nno uuid here") is None


def test_install_store_id_fresh_and_idempotent(tmp_path):
    from ormah.cloud.keys import install_store_id

    memory_dir = tmp_path / "memory"
    sid = "55555555-6666-4777-8888-999999999999"
    assert install_store_id(memory_dir, sid) == sid
    assert (memory_dir / ".store_id").read_text().strip() == sid
    assert stat.S_IMODE((memory_dir / ".store_id").stat().st_mode) == 0o600
    assert install_store_id(memory_dir, sid) == sid  # same id: no-op


def test_install_store_id_refuses_mismatch(tmp_path):
    from ormah.cloud.keys import install_store_id

    memory_dir = tmp_path / "memory"
    install_store_id(memory_dir, "55555555-6666-4777-8888-999999999999")
    with pytest.raises(CloudKeyError, match="orphan"):
        install_store_id(memory_dir, "66666666-7777-4888-9999-aaaaaaaaaaaa")


def test_extract_store_id_fails_closed_on_malformed(tmp_path):
    """A kit whose store_id line is damaged must abort, not fall through to
    minting a fresh namespace."""
    from ormah.cloud.keys import extract_store_id

    with pytest.raises(CloudKeyError, match="malformed store id"):
        extract_store_id("store_id: not-a-uuid-at-all")


def test_extract_store_id_fails_closed_on_non_v4(tmp_path):
    from ormah.cloud.keys import extract_store_id

    # Valid UUID, wrong version (v7) — E08 requires UUIDv4
    with pytest.raises(CloudKeyError, match="malformed store id"):
        extract_store_id("store_id: 018f4b2c-7a00-7000-8000-000000000000")


def test_install_store_id_rejects_non_v4(tmp_path):
    from ormah.cloud.keys import install_store_id

    with pytest.raises(CloudKeyError, match="UUIDv4"):
        install_store_id(tmp_path / "memory", "018f4b2c-7a00-7000-8000-000000000000")
