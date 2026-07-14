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
from ormah.cloud.restore import CloudRestoreResult, restore_cloud_snapshot
from ormah.config import Settings
from ormah.models.node import MemoryNode, NodeType
from ormah.store.file_store import FileStore


def _save_node(memory_dir: Path, title: str, content: str) -> MemoryNode:
    node = MemoryNode(type=NodeType.fact, title=title, content=content)
    FileStore(memory_dir / "nodes").save(node)
    return node


class RestoreClient:
    def __init__(self):
        self.closed = False

    def list_blobs(self, store_id):
        return {
            "blobs": [
                {
                    "snapshot_id": "01RESTORE",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": 100,
                }
            ]
        }

    def presign_download(self, store_id, snapshot_id):
        return {"get_url": "https://objects.example/restore"}

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
    client = RestoreClient()
    monkeypatch.setattr(restore, "key_file_exists", lambda: True)
    monkeypatch.setattr(restore, "load_identities", lambda: [identity])
    monkeypatch.setattr(restore, "client_from_settings", lambda settings: client)
    monkeypatch.setattr(restore, "download_file", lambda url, path: shutil.copyfile(bundle, path))

    result = restore_cloud_snapshot(settings)

    assert result.snapshot_id == "01RESTORE"
    assert result.restore.rebuilt_nodes == 1
    assert list((target_memory / "nodes").glob(f"*_{expected.short_id}.md"))
    assert not list((target_memory / "nodes").glob(f"*_{replaced.short_id}.md"))
    assert result.restore.safety_backup is not None
    assert client.closed is True


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
