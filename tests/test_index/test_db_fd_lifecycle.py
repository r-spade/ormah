"""Ephemeral threads must not leak their per-thread SQLite connection (FD-leak regression)."""
from __future__ import annotations

import gc
import threading

from ormah.index.db import Database


def test_ephemeral_thread_connection_is_retired(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_schema()  # opens the main-thread connection
    baseline = len(db._all_conns)

    def worker():
        db.conn.execute("SELECT 1").fetchone()  # forces a new per-thread connection

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # `t` still holds a strong ref to the last thread after the loop above (the loop
    # variable outlives the loop) — drop it explicitly, or that one thread never dies.
    del threads, t
    gc.collect()  # run the weakref finalizers for the now-dead threads

    assert len(db._all_conns) <= baseline
