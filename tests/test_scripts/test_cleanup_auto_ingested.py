"""Cleanup script: dry-run is read-only; apply removes only non-preserved sources."""
from __future__ import annotations

from pathlib import Path

import scripts.cleanup_auto_ingested as cleanup_mod
from scripts.cleanup_auto_ingested import (
    EXIT_QUARANTINE_ORPHAN,
    plan_cleanup,
    run_cleanup,
)


def _rows():
    return [
        ("agent:claude_code", 6547),
        ("agent:ingester", 659),
        ("agent:consolidator", 66),
        ("agent:unknown", 112),
        ("system:self", 1),
    ]


def test_plan_preserves_listed_sources():
    to_delete, kept = plan_cleanup(_rows(), {"agent:unknown", "system:self"})
    assert kept == 113
    assert to_delete == {"agent:claude_code", "agent:ingester", "agent:consolidator"}


def test_plan_can_also_preserve_uploads():
    to_delete, kept = plan_cleanup(_rows(), {"agent:unknown", "system:self", "agent:ingester"})
    assert kept == 113 + 659
    assert "agent:ingester" not in to_delete


class _FakeInfo:
    def __init__(self, path: Path) -> None:
        self.path = path


class _FakeBackupService:
    """Stand-in for BackupService used by run_cleanup's forced-backup step."""

    def __init__(self, result):
        self._result = result

    def create(self, *, reason: str, prune: bool = False):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeFileStore:
    def __init__(self, nodes_dir: Path) -> None:
        self.nodes_dir = nodes_dir


class _FakeBuilder:
    def __init__(self, should_fail: bool) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def full_rebuild(self) -> int:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("rebuild boom")
        return 0


class _FakeEngine:
    def __init__(self, nodes_dir: Path, rebuild_should_fail: bool) -> None:
        self.file_store = _FakeFileStore(nodes_dir)
        self.builder = _FakeBuilder(rebuild_should_fail)


def _make_nodes(tmp_path: Path, rows) -> Path:
    """Create one markdown file per row, tagged with its source in frontmatter."""
    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    i = 0
    for source, count in rows:
        for _ in range(count):
            i += 1
            (nodes_dir / f"node-{i}.md").write_text(
                "---\n"
                f"id: node-{i}\n"
                "type: fact\n"
                f"source: {source}\n"
                "created: 2026-07-05T00:00:00Z\n"
                "updated: 2026-07-05T00:00:00Z\n"
                "---\n"
                "body\n",
                encoding="utf-8",
            )
    return nodes_dir


def _small_rows():
    return [
        ("agent:claude_code", 3),
        ("agent:unknown", 2),
        ("system:self", 1),
    ]


def test_run_cleanup_aborts_with_zero_deletes_when_backup_fails(tmp_path):
    rows = _small_rows()
    nodes_dir = _make_nodes(tmp_path, rows)
    engine = _FakeEngine(nodes_dir, rebuild_should_fail=False)
    backup_service = _FakeBackupService(RuntimeError("backup failed"))

    rc = run_cleanup(
        engine,
        backup_service,
        rows=rows,
        preserve={"agent:unknown", "system:self"},
    )

    assert rc == 3
    assert len(list(nodes_dir.glob("*.md"))) == 6


def test_run_cleanup_aborts_with_zero_deletes_when_backup_returns_none(tmp_path):
    rows = _small_rows()
    nodes_dir = _make_nodes(tmp_path, rows)
    engine = _FakeEngine(nodes_dir, rebuild_should_fail=False)
    backup_service = _FakeBackupService(None)

    rc = run_cleanup(
        engine,
        backup_service,
        rows=rows,
        preserve={"agent:unknown", "system:self"},
    )

    assert rc == 3
    assert len(list(nodes_dir.glob("*.md"))) == 6


def test_run_cleanup_restores_quarantine_when_rebuild_fails(tmp_path):
    rows = _small_rows()
    nodes_dir = _make_nodes(tmp_path, rows)
    engine = _FakeEngine(nodes_dir, rebuild_should_fail=True)
    backup_path = tmp_path / "backups" / "memory_2026-07-05_00-00-00"
    backup_path.mkdir(parents=True)
    backup_service = _FakeBackupService(_FakeInfo(backup_path))

    rc = run_cleanup(
        engine,
        backup_service,
        rows=rows,
        preserve={"agent:unknown", "system:self"},
    )

    assert rc == 4
    # All original markdown restored; nothing left in quarantine.
    assert len(list(nodes_dir.glob("*.md"))) == 6


def test_run_cleanup_deletes_after_verified_backup_and_successful_rebuild(tmp_path):
    rows = _small_rows()
    nodes_dir = _make_nodes(tmp_path, rows)
    engine = _FakeEngine(nodes_dir, rebuild_should_fail=False)
    backup_path = tmp_path / "backups" / "memory_2026-07-05_00-00-00"
    backup_path.mkdir(parents=True)
    backup_service = _FakeBackupService(_FakeInfo(backup_path))

    rc = run_cleanup(
        engine,
        backup_service,
        rows=rows,
        preserve={"agent:unknown", "system:self"},
    )

    assert rc == 0
    # 3 claude_code nodes deleted, 3 preserved (2 unknown + 1 system:self) remain.
    assert len(list(nodes_dir.glob("*.md"))) == 3
    assert engine.builder.calls == 1


def test_run_cleanup_refuses_when_orphan_quarantine_exists(tmp_path):
    """A leftover quarantine from a prior interrupted run must abort a new cleanup — otherwise it
    compounds the markdown-outside-nodes_dir / stale-index inconsistency (council-pr I3)."""
    rows = _small_rows()
    nodes_dir = _make_nodes(tmp_path, rows)
    engine = _FakeEngine(nodes_dir, rebuild_should_fail=False)
    backup_path = tmp_path / "backups" / "b"
    backup_path.mkdir(parents=True)
    backup_service = _FakeBackupService(_FakeInfo(backup_path))

    orphan = nodes_dir.parent / "ormah_cleanup_quarantine_prior"
    orphan.mkdir()
    (orphan / "stranded.md").write_text(
        "---\nid: x\nsource: agent:claude_code\n---\nbody\n", encoding="utf-8"
    )

    rc = run_cleanup(engine, backup_service, rows=rows,
                     preserve={"agent:unknown", "system:self"})

    assert rc == EXIT_QUARANTINE_ORPHAN
    assert len(list(nodes_dir.glob("*.md"))) == 6  # deleted nothing
    assert engine.builder.calls == 0               # no rebuild attempted


def test_run_cleanup_partial_restore_skips_second_rebuild(tmp_path, monkeypatch):
    """When the rebuild fails AND restore is only partial, the script must NOT run a second rebuild
    (it would index an incomplete node set) and must keep the un-restored quarantine (council-pr
    I2)."""
    rows = _small_rows()  # 3 claude_code to delete
    nodes_dir = _make_nodes(tmp_path, rows)
    engine = _FakeEngine(nodes_dir, rebuild_should_fail=True)
    backup_path = tmp_path / "backups" / "b"
    backup_path.mkdir(parents=True)
    backup_service = _FakeBackupService(_FakeInfo(backup_path))

    # 3 quarantine-in moves succeed; the FIRST restore move (call #4) fails -> partial restore.
    real_move = cleanup_mod.shutil.move
    seen = {"n": 0}

    def flaky_move(src, dst):
        seen["n"] += 1
        if seen["n"] == 4:
            raise OSError("cannot restore this one")
        return real_move(src, dst)

    monkeypatch.setattr(cleanup_mod.shutil, "move", flaky_move)

    rc = run_cleanup(engine, backup_service, rows=rows,
                     preserve={"agent:unknown", "system:self"})

    assert rc == 4
    assert engine.builder.calls == 1  # partial restore -> the second rebuild is NOT attempted
    # The un-restored file stays quarantined (not destroyed); the operator recovers from backup.
    quarantines = list(nodes_dir.parent.glob("ormah_cleanup_quarantine_*"))
    assert any(any(q.glob("*.md")) for q in quarantines)
