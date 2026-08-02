#!/usr/bin/env python
"""Delete auto-ingested memory nodes, preserving manual ones. Dry-run by default.

Source of truth is the markdown in nodes/; deletes markdown for target nodes then
rebuilds the (atomic) index. A forced, verified backup is taken first (abort if it
does not yield a real artifact). Doomed markdown is moved to a quarantine directory
first and only permanently removed once ``full_rebuild()`` succeeds; on rebuild
failure the quarantined files are restored and the index is rebuilt again from the
untouched node set. Nothing is deleted without --apply AND --preserve.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from ormah.store.markdown import parse_node

# Exit codes for main()
EXIT_OK = 0
EXIT_EMPTY_PRESERVE = 2
EXIT_BACKUP_FAILED = 3
EXIT_REBUILD_FAILED = 4
EXIT_QUARANTINE_ORPHAN = 5


def plan_cleanup(rows, preserve: set[str]) -> tuple[set[str], int]:
    to_delete = {src for src, _ in rows if src not in preserve}
    kept = sum(count for src, count in rows if src in preserve)
    return to_delete, kept


def _print_table(rows) -> None:
    print(f"{'source':<24} {'count':>8}")
    print("-" * 34)
    for src, count in sorted(rows, key=lambda r: -r[1]):
        print(f"{src:<24} {count:>8}")
    print("-" * 34)
    print(f"{'TOTAL':<24} {sum(c for _, c in rows):>8}")


def _node_source(path: Path) -> str | None:
    try:
        return parse_node(path.read_text(encoding="utf-8")).source
    except Exception:
        return None


def run_cleanup(engine, backup_service, *, rows, preserve: set[str]) -> int:
    """Perform the destructive cleanup. Returns a process exit code.

    Steps: forced+verified backup -> quarantine doomed markdown -> full_rebuild
    -> delete quarantine on success, or restore it (and rebuild again) on failure.
    """
    to_delete, _kept = plan_cleanup(rows, preserve)
    nodes_dir = engine.file_store.nodes_dir

    # A leftover quarantine from a prior interrupted run means markdown is still stranded outside
    # nodes_dir with a stale index. Starting a fresh cleanup on top of that would compound the
    # inconsistency, so refuse until the operator resolves it (council-pr I3).
    orphans = [
        d for d in nodes_dir.parent.glob("ormah_cleanup_quarantine_*")
        if d.is_dir() and any(d.iterdir())
    ]
    if orphans:
        print(f"REFUSING: leftover quarantine from an interrupted cleanup: "
              f"{[str(o) for o in orphans]}. Restore those files into {nodes_dir} (or remove them "
              f"if already handled) and rebuild the index, then rerun.", file=sys.stderr)
        return EXIT_QUARANTINE_ORPHAN

    try:
        backup_info = backup_service.create(reason="pre-cleanup", prune=False)
    except Exception as exc:
        print(f"REFUSING: forced backup failed ({exc}); deleting nothing.", file=sys.stderr)
        return EXIT_BACKUP_FAILED

    if backup_info is None or not Path(backup_info.path).exists():
        print("REFUSING: forced backup produced no verified artifact; deleting nothing.", file=sys.stderr)
        return EXIT_BACKUP_FAILED

    quarantine_dir = Path(tempfile.mkdtemp(prefix="ormah_cleanup_quarantine_", dir=str(nodes_dir.parent)))

    def _restore(moved_pairs: list[tuple[Path, Path]]) -> bool:
        """Best-effort restore; each move is isolated so one failure does not strand the rest.
        Returns True only if every quarantined file made it back to nodes_dir."""
        all_back = True
        for original, destination in moved_pairs:
            if not destination.exists():
                continue
            try:
                shutil.move(str(destination), str(original))
            except Exception as exc:  # noqa: BLE001
                all_back = False
                print(f"WARNING: could not restore {destination} -> {original}: {exc}", file=sys.stderr)
        return all_back

    def _drop_quarantine_only_if_empty() -> None:
        """Never rmtree unconditionally — only remove the quarantine once it holds nothing
        un-restored, so a secondary I/O failure can never destroy files that never made it back."""
        if not any(quarantine_dir.iterdir()):
            shutil.rmtree(quarantine_dir, ignore_errors=True)
        else:
            print(f"Quarantine KEPT at {quarantine_dir} (files not fully restored). "
                  f"Recover from backup: {backup_info.path}", file=sys.stderr)

    moved: list[tuple[Path, Path]] = []
    try:
        for path in sorted(nodes_dir.glob("*.md")):
            source = _node_source(path)
            if source in to_delete:
                destination = quarantine_dir / path.name
                shutil.move(str(path), str(destination))
                moved.append((path, destination))
    except Exception as exc:  # move-in failed partway -> restore what we moved, keep any that fail
        print(f"QUARANTINE MOVE FAILED ({exc}); restoring moved files.", file=sys.stderr)
        _restore(moved)
        _drop_quarantine_only_if_empty()
        return EXIT_REBUILD_FAILED

    try:
        engine.builder.full_rebuild()
    except Exception as exc:
        print(f"REBUILD FAILED ({exc}); restoring quarantined files.", file=sys.stderr)
        if _restore(moved):
            # Every file is back in nodes_dir — rebuild the index to match the restored set.
            try:
                engine.builder.full_rebuild()
            except Exception as exc2:
                print(f"SECONDARY REBUILD FAILED ({exc2}); files restored but index stale — "
                      f"rebuild manually. Backup: {backup_info.path}", file=sys.stderr)
        else:
            # Partial restore: some markdown is still only in quarantine. Do NOT rebuild — indexing
            # an incomplete node set would drop the un-restored nodes from the index too. Leave the
            # quarantine and point at the backup (council-pr I2).
            print(f"PARTIAL RESTORE: some files remain only in quarantine; index NOT rebuilt to "
                  f"avoid indexing an incomplete node set. Recover from backup: {backup_info.path}",
                  file=sys.stderr)
        _drop_quarantine_only_if_empty()
        return EXIT_REBUILD_FAILED

    # Success: the doomed files are gone from the index and their markdown is safely quarantined —
    # now it is safe to delete the quarantine.
    shutil.rmtree(quarantine_dir, ignore_errors=True)
    print(f"Deleted {len(moved)} nodes ({sorted(to_delete)}); rebuilt index.")
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--preserve", action="append", default=[])
    args = ap.parse_args(argv)

    from ormah.config import Settings
    from ormah.engine.memory_engine import MemoryEngine
    from ormah.backup import service_from_settings

    settings = Settings()
    engine = MemoryEngine(settings)
    rows = [
        (r[0], r[1])
        for r in engine.db.conn.execute(
            "SELECT source, COUNT(*) FROM nodes GROUP BY source"
        ).fetchall()
    ]
    _print_table(rows)

    preserve = set(args.preserve)
    to_delete, kept = plan_cleanup(rows, preserve)

    if not args.apply:
        print(f"\nDRY-RUN. Would delete: {sorted(to_delete)}; keep {kept} (preserve {sorted(preserve)}).")
        return EXIT_OK

    if not preserve:
        print("REFUSING: empty --preserve.", file=sys.stderr)
        return EXIT_EMPTY_PRESERVE

    backup_service = service_from_settings(settings)
    return run_cleanup(engine, backup_service, rows=rows, preserve=preserve)


if __name__ == "__main__":
    raise SystemExit(main())
