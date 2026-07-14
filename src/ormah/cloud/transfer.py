"""Data-plane transfers for service-issued presigned URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx


TRANSFER_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def put_file(url: str, path: Path) -> None:
    """PUT an encrypted bundle to a presigned object URL."""
    response = httpx.put(url, content=path.read_bytes(), timeout=TRANSFER_TIMEOUT)
    response.raise_for_status()


def download_file(url: str, path: Path) -> None:
    """Stream a presigned object download to disk."""
    with httpx.stream("GET", url, timeout=TRANSFER_TIMEOUT) as response:
        response.raise_for_status()
        with path.open("wb") as destination:
            for chunk in response.iter_bytes(1024 * 1024):
                destination.write(chunk)
