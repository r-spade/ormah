"""Tests for encrypted snapshot bundles: round-trip, tamper detection,
and the malicious-tar hardening suite."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from ormah.cloud.bundle import (
    MANIFEST_NAME,
    BundleError,
    build_bundle,
    open_bundle,
)
from ormah.cloud.crypto import (
    decrypt_bytes,
    encrypt_bytes,
    generate_identity,
    recipient_for,
)

STORE_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def keypair():
    identity = generate_identity()
    return identity, recipient_for(identity)


@pytest.fixture
def backup_dir(tmp_path):
    """A minimal finished local backup: nodes/, deleted/, backup.json."""
    d = tmp_path / "memory_2026-07-12_12-00-00"
    (d / "nodes").mkdir(parents=True)
    (d / "deleted").mkdir()
    (d / "nodes" / "fact_alpha_abc123.md").write_text(
        "---\nid: abc123\ntype: fact\n---\nAlpha content.\n"
    )
    (d / "nodes" / "fact_beta_def456.md").write_text(
        "---\nid: def456\ntype: fact\n---\nBeta content.\n"
    )
    (d / "deleted" / "fact_gone_9f9f9f.md").write_text(
        "---\nid: 9f9f9f\ntype: fact\ndeleted_at: '2026-07-12T10:00:00Z'\n---\nGone.\n"
    )
    (d / "backup.json").write_text(json.dumps({"version": 1, "reason": "test"}) + "\n")
    return d


def _dir_hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --- Round trip ---


def test_roundtrip_byte_identical(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(
        backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID, reason="test"
    )
    assert bundle.read_bytes().startswith(b"age-encryption.org/v1")

    dest = tmp_path / "restored"
    info = open_bundle(bundle, dest, [identity])

    source = _dir_hashes(backup_dir)
    restored = _dir_hashes(dest)
    restored.pop(MANIFEST_NAME)  # manifest is bundle-only, not part of the backup
    assert restored == source

    assert info.store_id == STORE_ID
    assert info.reason == "test"
    assert info.node_count == 2
    assert info.deleted_count == 1
    assert info.file_count == 4
    assert info.sync_base_snapshot_id is None
    assert info.device_id is None


def test_sync_fields_carried(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(
        backup_dir,
        tmp_path / "out.age",
        [recipient],
        store_id=STORE_ID,
        sync_base_snapshot_id="01J0SNAPSHOT",
        device_id="dev-42",
    )
    info = open_bundle(bundle, tmp_path / "restored", [identity])
    assert info.sync_base_snapshot_id == "01J0SNAPSHOT"
    assert info.device_id == "dev-42"


def test_missing_backup_dir(tmp_path, keypair):
    _, recipient = keypair
    with pytest.raises(BundleError, match="not found"):
        build_bundle(tmp_path / "nope", tmp_path / "out.age", [recipient], store_id=STORE_ID)


# --- Tamper detection ---


def _rebuild_tampered(bundle_path, identity, recipient, mutate):
    """Decrypt a bundle, let `mutate` alter the member dict, re-encrypt."""
    plaintext = decrypt_bytes(bundle_path.read_bytes(), [identity])
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as tar:
        for member in tar:
            members[member.name] = tar.extractfile(member).read()

    mutate(members)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tampered = bundle_path.parent / "tampered.age"
    tampered.write_bytes(encrypt_bytes(buffer.getvalue(), [recipient]))
    return tampered


def test_flipped_byte_fails_loudly(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    def flip(members):
        name = "nodes/fact_alpha_abc123.md"
        data = bytearray(members[name])
        data[-2] ^= 0xFF
        members[name] = bytes(data)

    tampered = _rebuild_tampered(bundle, identity, recipient, flip)
    with pytest.raises(BundleError, match="Hash mismatch"):
        open_bundle(tampered, tmp_path / "restored", [identity])


def test_missing_file_fails(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    tampered = _rebuild_tampered(
        bundle, identity, recipient, lambda m: m.pop("nodes/fact_beta_def456.md")
    )
    with pytest.raises(BundleError, match="missing manifest-listed"):
        open_bundle(tampered, tmp_path / "restored", [identity])


def test_extra_file_fails(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    tampered = _rebuild_tampered(
        bundle,
        identity,
        recipient,
        lambda m: m.__setitem__("nodes/smuggled_ffffff.md", b"not in manifest"),
    )
    with pytest.raises(BundleError, match="not in the manifest"):
        open_bundle(tampered, tmp_path / "restored", [identity])


def test_missing_manifest_fails(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    tampered = _rebuild_tampered(bundle, identity, recipient, lambda m: m.pop(MANIFEST_NAME))
    with pytest.raises(BundleError, match=f"missing {MANIFEST_NAME}"):
        open_bundle(tampered, tmp_path / "restored", [identity])


def test_wrong_identity_fails_cleanly(tmp_path, backup_dir, keypair):
    from ormah.cloud.crypto import CloudCryptoError

    _, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)
    with pytest.raises(CloudCryptoError, match="no matching key"):
        open_bundle(bundle, tmp_path / "restored", [generate_identity()])


# --- Malicious tar suite ---


def _encrypted_tar(tmp_path, recipient, add_members):
    """Build an encrypted tar from scratch via `add_members(tar)`."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        add_members(tar)
    path = tmp_path / "evil.age"
    path.write_bytes(encrypt_bytes(buffer.getvalue(), [recipient]))
    return path


def _reg(tar, name, data=b"x"):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize(
    "name",
    [
        "../escape.md",
        "nodes/../../escape.md",
        "/etc/passwd",
        "nodes/sub/dir.md",  # nesting not allowed
        "nodes/not-markdown.txt",
        "secrets.json",  # not in the allowlist
    ],
)
def test_rejects_unsafe_or_disallowed_paths(tmp_path, keypair, name):
    identity, recipient = keypair
    evil = _encrypted_tar(tmp_path, recipient, lambda tar: _reg(tar, name))
    with pytest.raises(BundleError, match="unsafe path|disallowed member"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_symlink(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        info = tarfile.TarInfo(name="nodes/link_abc.md")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="non-regular"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_hardlink(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        info = tarfile.TarInfo(name="nodes/hard_abc.md")
        info.type = tarfile.LNKTYPE
        info.linkname = "backup.json"
        tar.addfile(info)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="non-regular"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_device_node(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        info = tarfile.TarInfo(name="nodes/dev_abc.md")
        info.type = tarfile.CHRTYPE
        info.devmajor, info.devminor = 1, 3
        tar.addfile(info)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="non-regular"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_directory_member(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        info = tarfile.TarInfo(name="nodes/")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="non-regular"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_duplicate_names(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        _reg(tar, "nodes/dup_abc.md", b"first")
        _reg(tar, "nodes/dup_abc.md", b"second")

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="duplicate member"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_case_colliding_names(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        _reg(tar, "nodes/case_abc.md", b"lower")
        _reg(tar, "nodes/CASE_abc.md", b"upper")

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="case-colliding"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_rejects_too_many_members(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        for i in range(5):
            _reg(tar, f"nodes/n{i}_aaaa.md")

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="member limit"):
        open_bundle(evil, tmp_path / "restored", [identity], max_members=3)


def test_rejects_oversized_expansion(tmp_path, keypair):
    identity, recipient = keypair

    def add(tar):
        _reg(tar, "nodes/big_abc.md", b"a" * 1024)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="expansion limit"):
        open_bundle(evil, tmp_path / "restored", [identity], max_expanded_bytes=512)


def test_rejects_non_tar_payload(tmp_path, keypair):
    identity, recipient = keypair
    garbage = tmp_path / "garbage.age"
    garbage.write_bytes(encrypt_bytes(b"this is not a tar archive", [recipient]))
    with pytest.raises(BundleError, match="not a valid tar.gz"):
        open_bundle(garbage, tmp_path / "restored", [identity])


def test_rejects_unknown_format_version(tmp_path, keypair):
    identity, recipient = keypair

    manifest = json.dumps({"format_version": 99, "files": []}).encode()

    def add(tar):
        _reg(tar, MANIFEST_NAME, manifest)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="format_version"):
        open_bundle(evil, tmp_path / "restored", [identity])


# --- Review fixes (Codex review of PR #108) ---


def test_rejects_nonempty_destination(tmp_path, backup_dir, keypair):
    """A pre-populated destination (e.g. containing a planted symlink) is
    refused outright — extraction only ever targets fresh directories."""
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    victim = tmp_path / "victim.txt"
    victim.write_text("do not touch")
    dest = tmp_path / "restored"
    (dest / "nodes").mkdir(parents=True)
    (dest / "nodes" / "fact_alpha_abc123.md").symlink_to(victim)

    with pytest.raises(BundleError, match="new or empty"):
        open_bundle(bundle, dest, [identity])
    assert victim.read_text() == "do not touch"


def test_rejects_destination_that_is_a_file(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)
    dest = tmp_path / "not-a-dir"
    dest.write_text("occupied")
    with pytest.raises(BundleError, match="new or empty"):
        open_bundle(bundle, dest, [identity])


def test_empty_existing_destination_is_fine(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)
    dest = tmp_path / "restored"
    dest.mkdir()
    info = open_bundle(bundle, dest, [identity])
    assert info.file_count == 4


def test_rejects_unicode_normalization_collision(tmp_path, keypair):
    """Composed vs decomposed forms of the same name must collide."""
    identity, recipient = keypair
    composed = "nodes/café_abc.md"          # é as single codepoint
    decomposed = "nodes/café_abc.md"       # e + combining acute

    def add(tar):
        _reg(tar, composed, b"one")
        _reg(tar, decomposed, b"two")

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="case-colliding"):
        open_bundle(evil, tmp_path / "restored", [identity])


def test_expansion_limit_enforced_on_actual_bytes(tmp_path, keypair):
    """A member lying about its size cannot bypass the streamed-bytes cap."""
    identity, recipient = keypair

    def add(tar):
        _reg(tar, "nodes/big_abc.md", b"a" * 2048)

    evil = _encrypted_tar(tmp_path, recipient, add)
    with pytest.raises(BundleError, match="expansion limit"):
        open_bundle(evil, tmp_path / "restored", [identity], max_expanded_bytes=1024)


def test_failed_open_leaves_no_staging_debris(tmp_path, backup_dir, keypair):
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    tampered = _rebuild_tampered(
        bundle, identity, recipient, lambda m: m.pop("nodes/fact_beta_def456.md")
    )
    dest = tmp_path / "cleanup-check" / "restored"
    with pytest.raises(BundleError):
        open_bundle(tampered, dest, [identity])
    parent = dest.parent
    leftovers = [p for p in parent.iterdir() if p.name.startswith(".ormah_extract_")]
    assert leftovers == []
    assert not dest.exists() or not any(dest.iterdir())


def test_rejects_symlinked_destination(tmp_path, backup_dir, keypair):
    """An empty dir behind a symlink must be rejected, not followed."""
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)

    real_target = tmp_path / "elsewhere"
    real_target.mkdir()
    dest = tmp_path / "restored"
    dest.symlink_to(real_target)

    with pytest.raises(BundleError, match="symlink"):
        open_bundle(bundle, dest, [identity])
    assert not any(real_target.iterdir())


def test_rejects_oversized_encrypted_bundle(tmp_path, backup_dir, keypair):
    """The encrypted input itself is capped before being read into memory."""
    identity, recipient = keypair
    bundle = build_bundle(backup_dir, tmp_path / "out.age", [recipient], store_id=STORE_ID)
    with pytest.raises(BundleError, match="size limit"):
        open_bundle(bundle, tmp_path / "restored", [identity], max_bundle_bytes=64)
