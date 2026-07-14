#!/usr/bin/env python3
"""E02 round-trip spike: snapshot → encrypt → R2 → decrypt → verify → restore.

Dev-only script (boto3 lives in the dev dependency group; it never enters the
shipped package). Requires a dev R2 bucket:

    export ORMAH_DEV_R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
    export ORMAH_DEV_R2_ACCESS_KEY_ID="..."
    export ORMAH_DEV_R2_SECRET="..."
    export ORMAH_DEV_R2_BUCKET="ormah-dev"

    uv run python scripts/cloud_spike.py

The transcript this prints is the founding-member demo: the object in the
bucket is ciphertext, and the restored graph is byte-identical to the source.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import uuid
from pathlib import Path

import boto3
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ormah.backup import BackupService  # noqa: E402
from ormah.cloud.bundle import open_bundle, build_bundle  # noqa: E402
from ormah.cloud.crypto import AGE_HEADER, generate_identity, recipient_for  # noqa: E402
from ormah.index.builder import IndexBuilder  # noqa: E402
from ormah.index.db import Database  # noqa: E402
from ormah.store.file_store import FileStore  # noqa: E402
from ormah.models.node import MemoryNode, NodeType  # noqa: E402

STEP = 0


def step(msg: str) -> None:
    global STEP
    STEP += 1
    print(f"\n[{STEP}] {msg}")


def sha_map(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.md"))
        if p.is_file()
    }


def main() -> int:
    endpoint = os.environ.get("ORMAH_DEV_R2_ENDPOINT")
    key_id = os.environ.get("ORMAH_DEV_R2_ACCESS_KEY_ID")
    secret = os.environ.get("ORMAH_DEV_R2_SECRET")
    bucket = os.environ.get("ORMAH_DEV_R2_BUCKET")
    if not all([endpoint, key_id, secret, bucket]):
        print("Missing ORMAH_DEV_R2_* env (ENDPOINT, ACCESS_KEY_ID, SECRET, BUCKET).")
        return 1

    work = Path(tempfile.mkdtemp(prefix="ormah-spike-"))
    print(f"Ormah E02 encrypted round-trip spike — working dir {work}")

    step("Create a scratch memory store with real nodes")
    memory_dir = work / "memory"
    store = FileStore(memory_dir / "nodes")
    for i in range(5):
        store.save(MemoryNode(
            type=NodeType.fact,
            title=f"Spike fact {i}",
            content=f"Memory number {i} for the encrypted round-trip demo.",
        ))
    source_hashes = sha_map(memory_dir / "nodes")
    print(f"    {len(source_hashes)} nodes written")

    step("Local backup (BackupService.create)")
    service = BackupService(memory_dir=memory_dir, backup_dir=work / "backups")
    backup = service.create(reason="spike")
    print(f"    backup: {backup.name} ({backup.node_count} nodes)")

    step("Generate age identity and build encrypted bundle")
    identity = generate_identity()
    store_id = str(uuid.uuid4())
    bundle_path = build_bundle(
        backup.path, work / "spike.age", [recipient_for(identity)],
        store_id=store_id, reason="spike",
    )
    print(f"    bundle: {bundle_path} ({bundle_path.stat().st_size} bytes, store {store_id[:8]}…)")

    step("Presign PUT and upload with httpx (no boto3 in the client path)")
    s3 = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=key_id, aws_secret_access_key=secret,
        region_name="auto",
    )
    object_key = f"spike/{store_id}/{bundle_path.name}"
    put_url = s3.generate_presigned_url(
        "put_object", Params={"Bucket": bucket, "Key": object_key}, ExpiresIn=900
    )
    response = httpx.put(put_url, content=bundle_path.read_bytes(), timeout=120)
    response.raise_for_status()
    print(f"    uploaded to r2://{bucket}/{object_key} (HTTP {response.status_code})")

    step("Presign GET, download, and verify the bucket object is ciphertext")
    get_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": object_key}, ExpiresIn=900
    )
    downloaded = httpx.get(get_url, timeout=120).raise_for_status().content
    assert downloaded == bundle_path.read_bytes(), "download differs from upload"
    assert downloaded.startswith(AGE_HEADER), "object in bucket is not age ciphertext!"
    assert b"Memory number" not in downloaded, "plaintext leaked into the bucket!"
    print(f"    downloaded {len(downloaded)} bytes; starts with {AGE_HEADER!r}; no plaintext")

    step("Decrypt + hash-verify + extract (open_bundle)")
    bundle_copy = work / "downloaded.age"
    bundle_copy.write_bytes(downloaded)
    restored_snapshot = work / "restored-snapshot"
    info = open_bundle(bundle_copy, restored_snapshot, [identity])
    print(f"    manifest verified: {info.file_count} files, store {info.store_id[:8]}…, "
          f"{info.node_count} nodes / {info.deleted_count} deleted")

    step("Restore into a fresh memory dir and rebuild the index")
    fresh_memory = work / "fresh-memory"
    (fresh_memory / "nodes").mkdir(parents=True)
    for sub in ("nodes", "deleted"):
        src = restored_snapshot / sub
        if src.is_dir():
            dest = fresh_memory / sub
            dest.mkdir(exist_ok=True)
            for f in src.glob("*.md"):
                (dest / f.name).write_bytes(f.read_bytes())
    db = Database(fresh_memory / "index.db")
    db.init_schema()
    try:
        rebuilt = IndexBuilder(db, FileStore(fresh_memory / "nodes")).full_rebuild()
    finally:
        db.close()
    print(f"    index rebuilt: {rebuilt} nodes")

    step("Assert counts and per-file hashes match the source exactly")
    restored_hashes = sha_map(fresh_memory / "nodes")
    assert rebuilt == len(source_hashes), f"node count mismatch: {rebuilt} != {len(source_hashes)}"
    assert restored_hashes == source_hashes, "restored node hashes differ from source!"
    print(f"    {len(restored_hashes)}/{len(source_hashes)} files byte-identical ✓")

    step("Clean up the spike object from the bucket")
    s3.delete_object(Bucket=bucket, Key=object_key)
    print("    deleted")

    print("\nSPIKE OK — snapshot → encrypt → R2 → download → decrypt → verify → restore ✓")
    print("The bucket only ever held ciphertext; the keys never left this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
