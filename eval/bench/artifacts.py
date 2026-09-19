"""Durable append-only journals and atomic manifests."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class Journal:
    """Only a torn final line is recoverable; corrupt complete rows fail loudly."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.rows = []
        if path.exists():
            with path.open("rb+") as f:
                while True:
                    start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        f.truncate(start)
                        break
                    self.rows.append(json.loads(line))

    def append(self, row: dict) -> None:
        with self.lock, self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            self.rows.append(row)

    def latest(self) -> dict:
        return {row["question_id"]: row for row in self.rows}
