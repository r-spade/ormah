from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest

from ormah.backup import run_auto_backup, service_from_settings
from ormah.cloud.bundle import build_bundle
from ormah.cloud.crypto import generate_identity
from ormah.cloud.entitlements import EntitlementStatus
from ormah.cloud.state import load_state, save_state, CloudState
from ormah.config import Settings
from ormah.models.node import MemoryNode, NodeType
from ormah.store.file_store import FileStore


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


class FakeCloudClient:
    def __init__(self, *, bundle: Path | None = None):
        self.bundle = bundle
        self.created: list[tuple[str, int, str]] = []
        self.finalized: list[tuple] = []
        self.closed = False

    def create_upload(self, store_id, size, sha256):
        self.created.append((store_id, size, sha256))
        return {
            "upload_id": "upload-1",
            "snapshot_id": "01SNAPSHOT",
            "put_url": "https://objects.example/put",
        }

    def finalize_upload(self, *args):
        self.finalized.append(args)
        return {"snapshot_id": "01SNAPSHOT", "status": "committed", "head": {"seq": 0}}

    def list_blobs(self, store_id):
        return {
            "blobs": [
                {"snapshot_id": "01SNAPSHOT", "created_at": NOW.isoformat(), "size_bytes": 1}
            ]
        }

    def presign_download(self, store_id, snapshot_id):
        return {"get_url": "https://objects.example/get"}

    def close(self):
        self.closed = True


def _save_node(memory_dir: Path, title: str = "Cloud proof") -> MemoryNode:
    node = MemoryNode(
        type=NodeType.fact,
        title=title,
        content="Cloud restore verification finds this durable memory.",
    )
    FileStore(memory_dir / "nodes").save(node)
    return node


def _settings(tmp_path: Path, *, with_node: bool = True) -> tuple[Settings, str]:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    store_id = str(uuid.uuid4())
    (memory_dir / ".store_id").write_text(store_id + "\n", encoding="utf-8")
    if with_node:
        _save_node(memory_dir)
    return (
        Settings(
            memory_dir=memory_dir,
            backup_dir=tmp_path / "backups",
            cloud_backup_enabled=True,
            cloud_backup_interval_hours=24,
            account_token="test-token",
        ),
        store_id,
    )


@pytest.fixture
def cloud_state_dir(tmp_path, monkeypatch):
    from ormah.cloud import state

    path = tmp_path / "cloud-state"
    monkeypatch.setattr(state, "CLOUD_STATE_DIR", path)
    return path


def _patch_upload_prerequisites(monkeypatch, client: FakeCloudClient):
    from ormah.cloud import jobs

    monkeypatch.setattr(jobs, "key_file_exists", lambda: True)
    monkeypatch.setattr(jobs, "check_entitlement", lambda settings: EntitlementStatus.ACTIVE)
    monkeypatch.setattr(jobs, "current_recipient", lambda: object())
    monkeypatch.setattr(jobs, "client_from_settings", lambda settings: client)
    monkeypatch.setattr(jobs, "_utc_now", lambda: NOW)

    def fake_bundle(backup_dir, out_path, recipients, **kwargs):
        out_path.write_bytes(b"age-encrypted-bundle")
        return out_path

    monkeypatch.setattr(jobs, "build_bundle", fake_bundle)
    monkeypatch.setattr(jobs, "put_file", lambda url, path: None)


def test_cloud_backup_disabled_is_first_guard(tmp_path, monkeypatch):
    from ormah.cloud import jobs

    settings, _ = _settings(tmp_path)
    settings.cloud_backup_enabled = False
    monkeypatch.setattr(
        jobs,
        "key_file_exists",
        lambda: (_ for _ in ()).throw(AssertionError("key guard should not run")),
    )

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None


def test_cloud_backup_missing_key_records_error(tmp_path, monkeypatch, cloud_state_dir):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    monkeypatch.setattr(jobs, "key_file_exists", lambda: False)

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None
    assert "key is missing" in load_state(store_id).last_upload_error


def test_cloud_backup_missing_store_short_circuits_entitlement(tmp_path, monkeypatch):
    from ormah.cloud import jobs

    settings, _ = _settings(tmp_path)
    (settings.memory_dir / ".store_id").unlink()
    monkeypatch.setattr(jobs, "key_file_exists", lambda: True)
    monkeypatch.setattr(
        jobs,
        "check_entitlement",
        lambda settings: (_ for _ in ()).throw(AssertionError("entitlement should not run")),
    )

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None
    assert not (settings.memory_dir / ".store_id").exists()


def test_expired_entitlement_skips_cloud_but_not_local_backup(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    monkeypatch.setattr(jobs, "key_file_exists", lambda: True)
    monkeypatch.setattr(jobs, "check_entitlement", lambda settings: EntitlementStatus.EXPIRED)
    monkeypatch.setattr(
        jobs,
        "build_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("upload should not run")),
    )

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None
    assert "expired" in load_state(store_id).last_upload_error
    assert run_auto_backup(SimpleNamespace(settings=settings)) is not None


def test_cloud_backup_empty_store_short_circuits_bundle(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, _ = _settings(tmp_path, with_node=False)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    monkeypatch.setattr(
        jobs,
        "build_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bundle should not build")),
    )

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None
    assert client.created == []


def test_cloud_backup_fresh_upload_short_circuits_bundle(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    save_state(store_id, CloudState(last_upload_at=NOW - timedelta(hours=23)))
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    monkeypatch.setattr(
        jobs,
        "build_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bundle should not build")),
    )

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None
    assert client.created == []


def test_failed_put_records_error_and_never_finalizes(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)
    monkeypatch.setattr(jobs, "put_file", lambda url, path: (_ for _ in ()).throw(OSError("PUT failed")))

    assert jobs.run_cloud_backup(SimpleNamespace(settings=settings)) is None
    assert client.finalized == []
    assert "PUT failed" in load_state(store_id).last_upload_error
    assert client.closed is True


def test_successful_upload_finalizes_without_advancing_sync_head(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)

    result = jobs.run_cloud_backup(SimpleNamespace(settings=settings))

    assert result == "01SNAPSHOT"
    assert client.finalized == [(store_id, "upload-1")]
    state = load_state(store_id)
    assert state.last_upload_at == NOW
    assert state.last_upload_snapshot_id == "01SNAPSHOT"
    assert state.last_upload_error is None


def test_cloud_backup_never_reuses_newer_safety_backup_with_different_contents(tmp_path):
    from ormah.cloud import jobs

    settings, _ = _settings(tmp_path)
    service = service_from_settings(settings)
    stale = service.create(reason="pre-restore", prune=False)
    for path in (stale.path / "nodes").glob("*.md"):
        path.unlink()

    selected = jobs._backup_for_upload(service)

    assert selected.name != stale.name
    assert selected.node_count == 1
    assert len(list((selected.path / "nodes").glob("*.md"))) == 1


def test_two_stores_upload_under_their_own_ids(tmp_path, monkeypatch, cloud_state_dir):
    from ormah.cloud import jobs

    first_settings, first_id = _settings(tmp_path / "first")
    second_settings, second_id = _settings(tmp_path / "second")
    client = FakeCloudClient()
    _patch_upload_prerequisites(monkeypatch, client)

    jobs.run_cloud_backup(SimpleNamespace(settings=first_settings))
    jobs.run_cloud_backup(SimpleNamespace(settings=second_settings))

    assert [call[0] for call in client.created] == [first_id, second_id]
    assert load_state(first_id).last_upload_snapshot_id == "01SNAPSHOT"
    assert load_state(second_id).last_upload_snapshot_id == "01SNAPSHOT"


def _verification_bundle(tmp_path: Path, settings: Settings, store_id: str):
    service = service_from_settings(settings)
    backup = service.create(reason="verification-fixture")
    identity = generate_identity()
    bundle = tmp_path / "snapshot.age"
    build_bundle(
        backup.path,
        bundle,
        [identity.to_public()],
        store_id=store_id,
        reason="cloud-backup",
    )
    return bundle, identity


def _patch_verification(monkeypatch, client, bundle: Path, identity):
    from ormah.cloud import jobs

    monkeypatch.setattr(jobs, "key_file_exists", lambda: True)
    monkeypatch.setattr(jobs, "client_from_settings", lambda settings: client)
    monkeypatch.setattr(jobs, "load_identities", lambda: [identity])
    monkeypatch.setattr(jobs, "_utc_now", lambda: NOW)
    monkeypatch.setattr(jobs, "download_file", lambda url, path: shutil.copyfile(bundle, path))


def test_restore_verification_rebuilds_search_and_leaves_live_store_untouched(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    live_files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in settings.memory_dir.rglob("*")
        if path.is_file()
    }

    assert jobs.run_restore_verification(SimpleNamespace(settings=settings)) is True

    state = load_state(store_id)
    assert state.last_verify_ok is True
    assert state.last_verify_snapshot_id == "01SNAPSHOT"
    assert state.last_verify_error is None
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in live_files
    } == live_files


def test_restore_verification_search_probe_supports_unicode_nodes(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path, with_node=False)
    _save_node(settings.memory_dir, title="Mémoire récupérée")
    node_path = next((settings.memory_dir / "nodes").glob("*.md"))
    text = node_path.read_text(encoding="utf-8")
    node_path.write_text(
        text.replace(
            "Cloud restore verification finds this durable memory.",
            "Récupération chiffrée réussie.",
        ),
        encoding="utf-8",
    )
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)

    assert jobs.run_restore_verification(SimpleNamespace(settings=settings)) is True
    assert load_state(store_id).last_verify_ok is True


@pytest.mark.parametrize("failure", ["truncated", "hash-mismatch"])
def test_restore_verification_records_bundle_failures(
    tmp_path, monkeypatch, cloud_state_dir, failure
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    if failure == "truncated":
        bundle.write_bytes(bundle.read_bytes()[:20])
    else:
        monkeypatch.setattr(
            jobs,
            "open_bundle",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Hash mismatch")),
        )

    assert jobs.run_restore_verification(SimpleNamespace(settings=settings)) is False

    state = load_state(store_id)
    assert state.last_verify_ok is False
    assert state.last_verify_snapshot_id == "01SNAPSHOT"
    expected = "Hash mismatch" if failure == "hash-mismatch" else "decrypt"
    assert expected.lower() in state.last_verify_error.lower()


def test_restore_verification_cleans_temporary_tree_on_failure(
    tmp_path, monkeypatch, cloud_state_dir
):
    from ormah.cloud import jobs

    settings, store_id = _settings(tmp_path)
    bundle, identity = _verification_bundle(tmp_path, settings, store_id)
    client = FakeCloudClient(bundle=bundle)
    _patch_verification(monkeypatch, client, bundle, identity)
    root = tmp_path / "known-verification-root"
    real_mkdtemp = jobs.tempfile.mkdtemp

    def controlled_mkdtemp(*args, prefix=None, **kwargs):
        if prefix == "ormah-restore-verify-":
            root.mkdir()
            return str(root)
        return real_mkdtemp(*args, prefix=prefix, **kwargs)

    monkeypatch.setattr(jobs.tempfile, "mkdtemp", controlled_mkdtemp)
    monkeypatch.setattr(
        jobs,
        "open_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verification stopped")),
    )

    assert jobs.run_restore_verification(SimpleNamespace(settings=settings)) is False
    assert not root.exists()


def test_scheduler_registers_cloud_backup_and_weekly_verification(tmp_path, monkeypatch):
    from ormah.background import scheduler as scheduler_module

    class FakeScheduler:
        def __init__(self):
            self.jobs = []

        def add_job(self, func, trigger, **kwargs):
            self.jobs.append({"func": func, "trigger": trigger, **kwargs})

        def start(self):
            pass

        def get_jobs(self):
            return self.jobs

    monkeypatch.setattr(scheduler_module, "BackgroundScheduler", FakeScheduler)
    settings = Settings(
        memory_dir=tmp_path / "memory",
        backup_dir=tmp_path / "backups",
        cloud_backup_interval_hours=36,
    )
    engine = SimpleNamespace(
        settings=settings,
        builder=SimpleNamespace(incremental_update=lambda: (0, 0)),
    )

    scheduler, _tracker = scheduler_module.start_scheduler(engine)

    cloud = next(job for job in scheduler.jobs if job["id"] == "cloud_backup")
    verify = next(job for job in scheduler.jobs if job["id"] == "restore_verification")
    assert cloud["trigger"] == "interval"
    assert cloud["hours"] == 36
    assert cloud["next_run_time"] is not None
    assert verify["trigger"] == "interval"
    assert verify["hours"] == 168
