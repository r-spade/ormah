"""Tests for age encryption wrappers."""

from __future__ import annotations

import pytest

from ormah.cloud.crypto import (
    AGE_HEADER,
    CloudCryptoError,
    decrypt_bytes,
    encrypt_bytes,
    generate_identity,
    identity_from_str,
    identity_to_str,
    recipient_for,
    recipient_from_str,
)


def test_roundtrip():
    identity = generate_identity()
    ciphertext = encrypt_bytes(b"secret memory graph", [recipient_for(identity)])

    assert ciphertext.startswith(AGE_HEADER)
    assert b"secret memory graph" not in ciphertext
    assert decrypt_bytes(ciphertext, [identity]) == b"secret memory graph"


def test_identity_string_roundtrip():
    identity = generate_identity()
    s = identity_to_str(identity)
    assert s.startswith("AGE-SECRET-KEY-")

    restored = identity_from_str(s)
    ciphertext = encrypt_bytes(b"data", [recipient_for(identity)])
    assert decrypt_bytes(ciphertext, [restored]) == b"data"


def test_recipient_string_roundtrip():
    identity = generate_identity()
    recipient = recipient_from_str(str(recipient_for(identity)))
    ciphertext = encrypt_bytes(b"data", [recipient])
    assert decrypt_bytes(ciphertext, [identity]) == b"data"


def test_wrong_identity_clean_error():
    identity = generate_identity()
    stranger = generate_identity()
    ciphertext = encrypt_bytes(b"data", [recipient_for(identity)])

    with pytest.raises(CloudCryptoError, match="no matching key"):
        decrypt_bytes(ciphertext, [stranger])


def test_multi_identity_decrypt_after_rotation():
    """Bundles encrypted pre-rotation must decrypt with the full identity list."""
    old_identity = generate_identity()
    old_bundle = encrypt_bytes(b"pre-rotation data", [recipient_for(old_identity)])

    new_identity = generate_identity()  # rotation: new current, old retained
    identities = [new_identity, old_identity]

    assert decrypt_bytes(old_bundle, identities) == b"pre-rotation data"

    new_bundle = encrypt_bytes(b"post-rotation data", [recipient_for(new_identity)])
    assert decrypt_bytes(new_bundle, identities) == b"post-rotation data"


def test_invalid_strings_raise():
    with pytest.raises(CloudCryptoError, match="identity"):
        identity_from_str("not-a-key")
    with pytest.raises(CloudCryptoError, match="recipient"):
        recipient_from_str("not-a-recipient")


def test_empty_lists_raise():
    with pytest.raises(CloudCryptoError):
        encrypt_bytes(b"data", [])
    with pytest.raises(CloudCryptoError):
        decrypt_bytes(b"data", [])
