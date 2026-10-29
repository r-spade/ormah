"""Content fingerprints for resumable experiments, including uncommitted code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ormah
from ormah.config import Settings

from eval.bench.datasets import sha256_file


def experiment_fingerprint(settings: dict, runtime: dict) -> dict:
    # Hash the code actually imported, not HEAD alone: a dirty checkout can
    # change between invocations without changing either HEAD or its dirty flag.
    # Content hashes also permit documentation-only commits between phases.
    roots = {
        "ormah": Path(ormah.__file__).parent,
        "eval": Path(__file__).resolve().parents[1],
    }
    files = {
        f"{name}/{path.relative_to(root).as_posix()}": sha256_file(path)
        for name, root in roots.items()
        for path in sorted(root.rglob("*.py"))
    }
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    # Capture unpinned settings inherited from ORMAH_* / .env as well, without
    # writing account/config values into an artifact. The per-run DB path does
    # not affect the experiment, so replace it with a constant for hashing.
    effective = Settings(memory_dir=Path("."), **settings).model_dump(mode="json")
    settings_hash = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    return {
        "version": 1,
        "source_sha256": source_hash,
        "settings": dict(settings),
        "effective_settings_sha256": settings_hash,
        "runtime": runtime,
    }


def validate_resume(manifest: dict, current: dict) -> None:
    saved = manifest.get("experiment_fingerprint")
    if saved is None:
        raise ValueError(
            "Legacy run has no experiment fingerprint; use a new --run-id. "
            "Existing results remain readable with 'ormah eval bench report'."
        )
    changed = sorted(key for key in saved.keys() | current.keys() if saved.get(key) != current.get(key))
    if changed:
        raise ValueError(
            "Resume experiment differs from the saved manifest "
            f"({', '.join(changed)}); use a new --run-id."
        )
