"""Data-plane transfers for service-issued presigned URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx


TRANSFER_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
ALLOWED_PUT_HEADERS = {
    "content-length",
    "content-type",
    "x-amz-checksum-sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_put_headers(required_headers: dict[str, str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in (required_headers or {}).items():
        normalized = name.lower()
        if normalized not in ALLOWED_PUT_HEADERS:
            raise ValueError(f"Cloud service requested unsupported PUT header: {name}")
        if not isinstance(value, str) or "\r" in value or "\n" in value:
            raise ValueError(f"Cloud service returned an invalid PUT header: {name}")
        headers[normalized] = value
    return headers


def put_file(url: str, path: Path, required_headers: dict[str, str] | None = None) -> None:
    """Stream an encrypted bundle to a presigned object URL."""
    headers = _validated_put_headers(required_headers)
    with path.open("rb") as source:
        response = httpx.put(url, content=source, headers=headers, timeout=TRANSFER_TIMEOUT)
    response.raise_for_status()


def download_file(url: str, path: Path) -> None:
    """Stream a presigned object download to disk."""
    with httpx.stream("GET", url, timeout=TRANSFER_TIMEOUT) as response:
        response.raise_for_status()
        with path.open("wb") as destination:
            for chunk in response.iter_bytes(1024 * 1024):
                destination.write(chunk)
