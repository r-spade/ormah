"""Durable write buffer for remember calls when the Ormah server is unreachable.

Pending writes are stored as JSONL at <memory_dir_parent>/pending_writes.jsonl.
The buffer is drained automatically before the next successful remember call.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_BUFFER_FILENAME = "pending_writes.jsonl"


class WriteBuffer:
    """File-backed queue for buffering remember calls that fail due to server downtime.

    Thread safety: uses atomic rename for writes to avoid partial reads.
    """

    def __init__(self, buffer_path: Path) -> None:
        self.path = buffer_path

    def is_empty(self) -> bool:
        return not self.path.exists() or self.path.stat().st_size == 0

    def append(self, args: dict, params: dict) -> None:
        """Append a pending remember call to the buffer."""
        entry = json.dumps({"args": args, "params": params})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")
        logger.info("Buffered remember call (server unreachable). Buffer: %s", self.path)

    def load(self) -> list[dict]:
        """Load all pending entries without clearing the file."""
        if self.is_empty():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed buffer entry: %r", line)
        return entries

    def clear(self) -> None:
        """Remove the buffer file."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to clear write buffer: %s", e)

    def rewrite(self, entries: list[dict]) -> None:
        """Replace buffer contents with the given entries (used after partial drain)."""
        if not entries:
            self.clear()
            return
        tmp = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        os.replace(tmp, self.path)


def buffer_path_for(memory_dir: Path) -> Path:
    """Derive the buffer file path from the configured memory directory."""
    return memory_dir.parent / _BUFFER_FILENAME
