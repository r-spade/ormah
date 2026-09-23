"""Plan configuration edits before writing; remove only exact owned artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from functools import partial
from types import SimpleNamespace
from pathlib import Path

from . import json_config


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ormah-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        if path.exists():
            os.chmod(name, path.stat().st_mode & 0o777)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Installation:
    """An ownership receipt, with exact values and no credentials.

    Existing unowned entries are never adopted or overwritten. A user-edited
    owned entry is left alone on disconnect and produces a clear setup error.
    The receipt is written before mutations, allowing interrupted setup to be
    safely retried or disconnected. Unrelated changes to shared files survive.
    """

    def __init__(self, receipt: Path, *, json5: bool = False):
        self.editor = SimpleNamespace(**{
            name: partial(getattr(json_config, name), json5=json5)
            for name in ("get", "put", "append", "remove_item")
        })
        self.receipt = receipt
        self.records = json.loads(receipt.read_text()) if receipt.exists() else []
        if not isinstance(self.records, list):
            raise ValueError(f"Invalid Ormah ownership receipt: {receipt}")
        self.original: dict[Path, str | None] = {}
        self.pending: dict[Path, str] = {}

    def _read(self, path: Path, default: str = "{}\n") -> str:
        if path not in self.original:
            self.original[path] = path.read_text() if path.exists() else None
            self.pending[path] = self.original[path] if path.exists() else default
        return self.pending[path]

    def _record(self, record: dict) -> bool:
        if record in self.records:
            return True
        if any(r["path"] == record["path"] and r.get("keys") == record.get("keys")
               and r["kind"] != "item" for r in self.records):
            raise ValueError(f"Ormah installation changed at {record['path']}; disconnect first")
        return False

    def value(self, path: Path, keys: list[str], value: object) -> None:
        record = dict(kind="value", path=str(path), keys=keys, value=value)
        owned = self._record(record)
        text = self._read(path)
        found, current = self.editor.get(text, keys)
        if found:
            if owned and current == value:
                return
            raise ValueError(f"Preserving existing configuration: {path}: {'.'.join(keys)}")
        self.pending[path] = self.editor.put(text, keys, value)
        if not owned:
            self.records.append(record)

    def item(self, path: Path, keys: list[str], value: object) -> None:
        record = dict(kind="item", path=str(path), keys=keys, value=value)
        owned = self._record(record)
        text = self._read(path)
        found, current = self.editor.get(text, keys)
        if found and not isinstance(current, list):
            raise ValueError(f"Expected array in {path}: {'.'.join(keys)}")
        if found and value in current:
            if owned:
                return
            raise ValueError(f"Preserving unowned registration: {path}")
        if owned:
            raise ValueError(f"Ormah registration was edited in {path}; disconnect first")
        self.pending[path] = self.editor.append(text, keys, value)
        self.records.append(record)

    def file(self, path: Path, content: str) -> None:
        record = dict(kind="file", path=str(path), value=content)
        owned = self._record(record)
        current = self._read(path, "")
        if self.original[path] is not None and (not owned or current != content):
            raise ValueError(f"Preserving existing file: {path}")
        self.pending[path] = content
        if not owned:
            self.records.append(record)

    def commit(self) -> None:
        for path, original in self.original.items():
            if (path.read_text() if path.exists() else None) != original:
                raise ValueError(f"Configuration changed during setup: {path}; retry")
        atomic_write(self.receipt, json.dumps(self.records, indent=2) + "\n")
        for path, text in self.pending.items():
            if text != self.original[path]:
                atomic_write(path, text)

    def disconnect(self) -> list[str]:
        preserved = []
        deletes = []
        for record in reversed(self.records):
            path = Path(record["path"])
            if not path.exists():
                continue
            text = self._read(path)
            if record["kind"] == "file":
                if text == record["value"]:
                    deletes.append(path)
                else:
                    preserved.append(str(path))
                continue
            keys, value = record["keys"], record["value"]
            found, current = self.editor.get(text, keys)
            if not found:
                continue
            if record["kind"] == "item":
                if isinstance(current, list) and value in current:
                    self.pending[path] = self.editor.remove_item(text, keys, value)
                else:
                    preserved.append(str(path))
            elif current == value:
                self.pending[path] = self.editor.put(text, keys, None, delete=True)
            else:
                preserved.append(str(path))
        # All config has been parsed before any mutation (including corrupt files).
        for path, text in self.pending.items():
            if text != self.original[path]:
                atomic_write(path, text)
        for path in deletes:
            path.unlink()
        self.receipt.unlink(missing_ok=True)
        return preserved

    def intact(self, kind: str | None = None) -> bool:
        records = [r for r in self.records if kind is None or r["kind"] == kind]
        if not records:
            return False
        try:
            for record in records:
                text = Path(record["path"]).read_text()
                if record["kind"] == "file":
                    if text != record["value"]:
                        return False
                else:
                    found, current = self.editor.get(text, record["keys"])
                    if not found:
                        return False
                    if record["kind"] == "item":
                        if not isinstance(current, list) or record["value"] not in current:
                            return False
                    elif current != record["value"]:
                        return False
            return True
        except (OSError, ValueError):
            return False
