"""One-shot data migrations for the markdown store.

Run a migration against your configured store:
    python -m ormah.store.migrations null-space
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from ormah.index.builder import IndexBuilder
from ormah.index.db import Database
from ormah.models.node import normalize_space
from ormah.store.file_store import FileStore
from ormah.store.markdown import parse_node


def migrate_null_space(nodes_dir: Path, db_path: Path) -> tuple[int, int]:
    """Coerce literal 'null'/'none'/'' space strings to None (#22).

    A handful of stored nodes carried the placeholder string ``space='null'``
    (not SQL NULL), landing them in a phantom "null" space group. The write-path
    guard (``MemoryNode`` field validator) prevents new ones; this cleans the
    existing files and rebuilds the index. Idempotent.

    Returns ``(files_fixed, nodes_reindexed)``.
    """
    fs = FileStore(nodes_dir)
    fixed = 0
    for path in fs.list_paths():
        raw = frontmatter.loads(path.read_text(encoding="utf-8")).metadata.get("space")
        if isinstance(raw, str) and normalize_space(raw) != raw:
            # parse_node runs the validator (normalizes); save re-serializes clean.
            fs.save(parse_node(path.read_text(encoding="utf-8")))
            fixed += 1
    db = Database(db_path)
    db.init_schema()  # no-op on an existing DB; creates tables on a fresh one
    reindexed = IndexBuilder(db, fs).full_rebuild()
    return fixed, reindexed


def repair_global_identity(nodes_dir: Path, db_path: Path) -> tuple[int, int]:
    """Lock the identity cluster as global so auto_cluster never reassigns it (#22).

    The self node and everything edged to it (about_self memories) are global by
    definition. Earlier auto_cluster runs swept some into a project space once their
    placeholder 'null' was migrated to a real None. Reset those to space=None and pin
    them with space_locked=True. Idempotent.

    Returns ``(files_fixed, nodes_reindexed)``.
    """
    fs = FileStore(nodes_dir)
    db = Database(db_path)
    db.init_schema()
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'user_node_id'").fetchone()
    if row is None:
        return 0, IndexBuilder(db, fs).full_rebuild()
    uid = row[0]
    edged = db.conn.execute(
        "SELECT DISTINCT CASE WHEN source_id = ? THEN target_id ELSE source_id END "
        "FROM edges WHERE source_id = ? OR target_id = ?",
        (uid, uid, uid),
    ).fetchall()
    ids = {uid} | {r[0] for r in edged}

    fixed = 0
    for nid in ids:
        node = fs.load(nid)
        if node and (node.space is not None or not node.space_locked):
            node.space = None
            node.space_locked = True
            fs.save(node)
            fixed += 1
    return fixed, IndexBuilder(db, fs).full_rebuild()


def main(argv: list[str] | None = None) -> None:
    import sys

    from ormah.config import Settings

    args = argv if argv is not None else sys.argv[1:]
    cmd = args[0] if args else None
    if cmd not in ("null-space", "repair-identity"):
        print("usage: python -m ormah.store.migrations {null-space|repair-identity}")
        raise SystemExit(2)

    settings = Settings()
    if cmd == "null-space":
        fixed, reindexed = migrate_null_space(settings.nodes_dir, settings.db_path)
        print(f"migrated {fixed} file(s); reindexed {reindexed} node(s)")
    else:
        fixed, reindexed = repair_global_identity(settings.nodes_dir, settings.db_path)
        print(f"locked {fixed} identity node(s); reindexed {reindexed} node(s)")


if __name__ == "__main__":
    main()
