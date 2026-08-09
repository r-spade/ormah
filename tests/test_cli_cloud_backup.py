from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import pytest

from ormah.backup import BackupInfo, RestoreResult, service_from_settings
from ormah.cloud.bundle import build_bundle
from ormah.cloud.crypto import generate_identity
from ormah.cloud.restore import (
    CloudRestoreResult,
    prepare_cloud_restore,
    restore_cloud_snapshot,
)
from ormah.cloud.transfer import sha256_file
from ormah.config import Settings
from ormah.models.node import MemoryNode, NodeType
from ormah.store.file_store import FileStore


def _save_node(memory_dir: Path, title: str, content: str) -> MemoryNode:
    node = MemoryNode(type=NodeType.fact, title=title, content=content)
    FileStore(memory_dir / "nodes").save(node)
    return node


class RestoreClient:
    def __init__(self, *, size_bytes=100, sha256="0" * 64):
        self.closed = False
        self.size_bytes = size_bytes
        self.sha256 = sha256

    def list_blobs(self, store_id):
        return {
            "blobs": [
                {
                    "snapshot_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": self.size_bytes,
                    "sha256": self.sha256,
                }
            ]
        }

    def presign_download(self, store_id, snapshot_id):
        return {"get_url": "https://objects.example/restore"}

    def processing_limit(self, *, require_hardened_write):
        return 512 * 1024 * 1024

    def close(self):
        self.closed = True


def test_restore_cloud_snapshot_verifies_then_uses_backup_service(tmp_path, monkeypatch):
    from ormah.cloud import restore

    store_id = str(uuid.uuid4())
    source_memory = tmp_path / "source-memory"
    expected = _save_node(source_memory, "Recovered node", "Recovered cloud content")
    source_settings = Settings(
        memory_dir=source_memory,
        backup_dir=tmp_path / "source-backups",
    )
    backup = service_from_settings(source_settings).create(reason="cloud-fixture")
    identity = generate_identity()
    bundle = tmp_path / "snapshot.age"
    build_bundle(
        backup.path,
        bundle,
        [identity.to_public()],
        store_id=store_id,
        reason="cloud-backup",
    )

    target_memory = tmp_path / "target-memory"
    target_memory.mkdir()
    (target_memory / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    replaced = _save_node(target_memory, "Old local node", "This should be replaced")
    settings = Settings(
        memory_dir=target_memory,
        backup_dir=tmp_path / "target-backups",
        account_token="test-token",
    )
    client = RestoreClient(size_bytes=bundle.stat().st_size, sha256=sha256_file(bundle))
    monkeypatch.setattr(restore, "key_file_exists", lambda: True)
    monkeypatch.setattr(restore, "load_identities", lambda: [identity])
    monkeypatch.setattr(restore, "client_from_settings", lambda settings: client)
    monkeypatch.setattr(restore, "download_file", lambda url, path: shutil.copyfile(bundle, path))

    result = restore_cloud_snapshot(settings)

    assert result.snapshot_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert result.restore.rebuilt_nodes == 1
    assert list((target_memory / "nodes").glob(f"*_{expected.short_id}.md"))
    assert not list((target_memory / "nodes").glob(f"*_{replaced.short_id}.md"))
    assert result.restore.safety_backup is not None
    assert client.closed is True


def test_prepare_restore_skips_corrupt_newest_without_touching_live_memory(
    tmp_path, monkeypatch
):
    from ormah.cloud import restore

    store_id = str(uuid.uuid4())
    source_memory = tmp_path / "source-memory"
    _save_node(source_memory, "Recovered node", "Recovered cloud content")
    source_settings = Settings(
        memory_dir=source_memory,
        backup_dir=tmp_path / "source-backups",
    )
    backup = service_from_settings(source_settings).create(reason="cloud-fixture")
    identity = generate_identity()
    valid_bundle = tmp_path / "valid.age"
    build_bundle(
        backup.path,
        valid_bundle,
        [identity.to_public()],
        store_id=store_id,
        reason="cloud-backup",
    )
    corrupt_bundle = tmp_path / "corrupt.age"
    corrupt_bundle.write_bytes(b"not-an-age-bundle")

    newest_id = "01ARZ3NDEKTSV4RRFFQ69G5FAW"
    safe_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    class FallbackClient(RestoreClient):
        def list_blobs(self, store_id):
            return {
                "blobs": [
                    {
                        "snapshot_id": newest_id,
                        "created_at": "2026-08-09T11:00:00+00:00",
                        "size_bytes": corrupt_bundle.stat().st_size,
                        "sha256": sha256_file(corrupt_bundle),
                    },
                    {
                        "snapshot_id": safe_id,
                        "created_at": "2026-08-08T11:00:00+00:00",
                        "size_bytes": valid_bundle.stat().st_size,
                        "sha256": sha256_file(valid_bundle),
                    },
                ]
            }

        def presign_download(self, store_id, snapshot_id):
            return {"get_url": f"https://objects.example/{snapshot_id}"}

    target_memory = tmp_path / "target-memory"
    target_memory.mkdir()
    (target_memory / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    live_node = _save_node(target_memory, "Live local node", "Must remain unchanged")
    live_path = next((target_memory / "nodes").glob(f"*_{live_node.short_id}.md"))
    before = live_path.read_bytes()
    before_mtime = live_path.stat().st_mtime_ns
    settings = Settings(
        memory_dir=target_memory,
        backup_dir=tmp_path / "target-backups",
        account_token="test-token",
    )
    client = FallbackClient()
    monkeypatch.setattr(restore, "key_file_exists", lambda: True)
    monkeypatch.setattr(restore, "load_identities", lambda: [identity])
    monkeypatch.setattr(restore, "client_from_settings", lambda settings: client)

    def download(url, path):
        source = corrupt_bundle if url.endswith(newest_id) else valid_bundle
        shutil.copyfile(source, path)

    monkeypatch.setattr(restore, "download_file", download)

    prepared = prepare_cloud_restore(settings)

    assert prepared.snapshot_id == safe_id
    assert prepared.skipped_newer_snapshots == 1
    assert prepared.verified_node_count == 1
    assert live_path.read_bytes() == before
    assert live_path.stat().st_mtime_ns == before_mtime
    assert (settings.backup_dir / prepared.backup_name).is_dir()


def test_cloud_restore_does_not_check_entitlement(tmp_path, monkeypatch):
    from ormah.cloud import restore
    from ormah.cloud import entitlements

    store_id = str(uuid.uuid4())
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    settings = Settings(
        memory_dir=memory_dir,
        backup_dir=tmp_path / "backups",
        account_token="test-token",
    )
    monkeypatch.setattr(restore, "key_file_exists", lambda: True)
    monkeypatch.setattr(
        entitlements,
        "check_entitlement",
        lambda settings: (_ for _ in ()).throw(AssertionError("restore gated entitlement")),
    )
    monkeypatch.setattr(
        restore,
        "client_from_settings",
        lambda settings: SimpleNamespace(
            list_blobs=lambda store_id: {"blobs": []},
            close=lambda: None,
        ),
    )

    with pytest.raises(restore.CloudRestoreError, match="No committed"):
        restore_cloud_snapshot(settings)


def test_cloud_restore_rejects_oversized_snapshot_before_download(tmp_path, monkeypatch):
    from ormah.cloud import restore

    store_id = str(uuid.uuid4())
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    settings = Settings(
        memory_dir=memory_dir,
        backup_dir=tmp_path / "backups",
        account_token="test-token",
    )
    client = RestoreClient()
    client.processing_limit = lambda **kwargs: 50
    monkeypatch.setattr(restore, "key_file_exists", lambda: True)
    monkeypatch.setattr(restore, "client_from_settings", lambda settings: client)
    monkeypatch.setattr(
        restore,
        "download_file",
        lambda *args: (_ for _ in ()).throw(AssertionError("download must not start")),
    )

    with pytest.raises(restore.CloudRestoreError, match="safe processing limit"):
        restore_cloud_snapshot(settings, "01ARZ3NDEKTSV4RRFFQ69G5FAV")


def _local_status_service():
    return SimpleNamespace(
        latest=lambda: None,
        has_backupable_memory=lambda: False,
        backup_due=lambda interval_hours: True,
    )


def _cloud_status(*, warning=False):
    return {
        "enabled": True,
        "store_id": str(uuid.uuid4()),
        "interval_hours": 24,
        "entitlement": "active",
        "last_upload_at": "2026-07-13T10:00:00+00:00",
        "last_upload_snapshot_id": "01LATEST",
        "last_upload_error": None,
        "last_upload_age_seconds": 7200,
        "last_verify_at": "2026-07-13T11:00:00+00:00",
        "last_verify_ok": not warning,
        "last_verify_snapshot_id": "01LATEST",
        "last_verify_error": "hash mismatch" if warning else None,
        "warnings": ["Cloud restore verification failed: hash mismatch"] if warning else [],
    }


def test_backup_status_json_contains_cloud_section(capsys):
    from ormah.cli import main

    with (
        patch("sys.argv", ["ormah", "backup", "status", "--json"]),
        patch("ormah.cli._backup_service", return_value=_local_status_service()),
        patch("ormah.cloud.state.cloud_status_payload", return_value=_cloud_status()),
    ):
        main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["cloud"]["last_upload_snapshot_id"] == "01LATEST"
    assert payload["cloud"]["last_verify_ok"] is True
    assert "account_token" not in json.dumps(payload)


def test_backup_status_prints_loud_verification_warning(capsys):
    from ormah.cli import main

    with (
        patch("sys.argv", ["ormah", "backup", "status"]),
        patch("ormah.cli._backup_service", return_value=_local_status_service()),
        patch("ormah.cloud.state.cloud_status_payload", return_value=_cloud_status(warning=True)),
    ):
        main()

    output = capsys.readouterr().out
    assert "Last verified restorable: ✗" in output
    assert "WARNING: Cloud restore verification failed" in output


@pytest.mark.parametrize(
    ("argv", "expected_snapshot"),
    [
        (["ormah", "backup", "restore", "--cloud", "--yes"], None),
        (["ormah", "backup", "restore", "--cloud", "01CHOSEN", "--yes"], "01CHOSEN"),
    ],
)
def test_cli_cloud_restore_delegates_to_reusable_workflow(
    tmp_path, capsys, argv, expected_snapshot
):
    from ormah.cli import main

    backup = BackupInfo(
        name="memory_2026-07-13_12-00-00",
        path=tmp_path / "backups" / "memory_2026-07-13_12-00-00",
        created_at=datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
        node_count=3,
        deleted_count=1,
        size_bytes=100,
    )
    result = CloudRestoreResult(
        snapshot_id=expected_snapshot or "01LATEST",
        restore=RestoreResult(restored=backup, safety_backup=None, rebuilt_nodes=3),
    )

    with (
        patch("sys.argv", argv),
        patch("ormah.server_manager.is_server_running", return_value=False),
        patch("ormah.cloud.restore.restore_cloud_snapshot", return_value=result) as restore_call,
    ):
        main()

    assert restore_call.call_args.args[1] == expected_snapshot
    assert "Restored backup: memory_2026-07-13_12-00-00" in capsys.readouterr().out


def test_cli_restore_rejects_missing_source(capsys):
    from ormah.cli import main

    with (
        patch("sys.argv", ["ormah", "backup", "restore", "--yes"]),
        patch("ormah.server_manager.is_server_running", return_value=False),
        pytest.raises(SystemExit),
    ):
        main()

    assert "Provide a local backup name or use --cloud" in capsys.readouterr().err
