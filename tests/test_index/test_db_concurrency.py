"""Concurrency tests for the thread-local Database connection model."""

import threading
import time

import numpy as np

from ormah.index.db import Database


def _init_db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.init_schema()
    db.init_vec_table(8)
    return db


def test_each_thread_gets_distinct_connection(tmp_path):
    db = _init_db(tmp_path)
    conns: list[int | None] = [None] * 4
    start = threading.Barrier(4)
    release = threading.Event()

    def grab(index: int):
        start.wait(timeout=5)
        conns[index] = id(db.conn)
        release.wait(timeout=5)

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    release.set()
    for t in threads:
        t.join()

    # main thread connection is distinct from every worker's
    main_id = id(db.conn)
    assert None not in conns
    assert len(set(conns)) == 4
    assert main_id not in conns
    db.close()


def test_vector_search_works_from_worker_thread(tmp_path):
    """Regression: vec0 module is loaded per connection; a fresh thread must
    still be able to query node_vectors."""
    db = _init_db(tmp_path)
    vec = np.ones(8, dtype=np.float32)
    import struct

    blob = struct.pack("8f", *vec)
    with db.transaction() as conn:
        conn.execute("INSERT INTO node_vectors (id, embedding) VALUES (?, ?)", ("n1", blob))

    result: dict = {}

    def search():
        try:
            rows = db.conn.execute(
                "SELECT id FROM node_vectors WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
                (blob,),
            ).fetchall()
            result["ids"] = [r[0] for r in rows]
        except Exception as e:  # noqa: BLE001
            result["error"] = repr(e)

    t = threading.Thread(target=search)
    t.start()
    t.join()

    assert "error" not in result, result.get("error")
    assert result["ids"] == ["n1"]
    db.close()


def test_read_during_write_does_not_block(tmp_path):
    """A read on thread B returns promptly while thread A holds a write tx."""
    db = _init_db(tmp_path)
    with db.transaction() as conn:
        conn.execute("INSERT INTO meta (key, value) VALUES ('k', 'v')")

    write_holding = threading.Event()
    release_write = threading.Event()

    def slow_writer():
        with db.transaction() as conn:
            conn.execute("UPDATE meta SET value = 'v2' WHERE key = 'k'")
            write_holding.set()
            release_write.wait(timeout=5)

    wt = threading.Thread(target=slow_writer)
    wt.start()
    assert write_holding.wait(timeout=5)

    # Reader on the main thread must not wait for the writer to commit.
    start = time.monotonic()
    row = db.conn.execute("SELECT value FROM meta WHERE key = 'k'").fetchone()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"read blocked {elapsed:.2f}s on the writer"
    assert row["value"] == "v"  # WAL: reader sees last committed snapshot

    release_write.set()
    wt.join()
    db.close()
