from pathlib import Path

import httpx
import pytest
import respx

from ormah.cloud.transfer import put_file


@respx.mock
def test_put_file_streams_and_applies_only_signed_headers(tmp_path, monkeypatch):
    path = tmp_path / "bundle.age"
    path.write_bytes(b"ciphertext" * 1024)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError("PUT must stream")),
    )
    route = respx.put("https://objects.example/pending").mock(
        return_value=httpx.Response(200)
    )

    put_file(
        "https://objects.example/pending",
        path,
        {"x-amz-checksum-sha256": "signed-value"},
    )

    request = route.calls[0].request
    assert request.content == b"ciphertext" * 1024
    assert request.headers["x-amz-checksum-sha256"] == "signed-value"


def test_put_file_rejects_unexpected_or_malformed_headers(tmp_path):
    path = tmp_path / "bundle.age"
    path.write_bytes(b"ciphertext")
    with pytest.raises(ValueError, match="unsupported"):
        put_file("https://objects.example/pending", path, {"authorization": "secret"})
    with pytest.raises(ValueError, match="invalid"):
        put_file(
            "https://objects.example/pending",
            path,
            {"content-type": "application/octet-stream\r\nX-Evil: yes"},
        )
