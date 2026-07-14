"""Thin wrappers around pyrage (age encryption) for snapshot bundles.

All encryption happens in-process on bytes; never shell out to an `age`
binary. Decryption always takes the full identity list (current + retained
pre-rotation identities) so rotated stores stay decryptable.
"""

from __future__ import annotations

import pyrage
from pyrage import x25519


class CloudCryptoError(RuntimeError):
    """Raised when encryption or decryption fails."""


# age binary-format header; used to verify uploaded objects are ciphertext.
AGE_HEADER = b"age-encryption.org/v1"


def generate_identity() -> x25519.Identity:
    return x25519.Identity.generate()


def identity_to_str(identity: x25519.Identity) -> str:
    return str(identity)


def identity_from_str(value: str) -> x25519.Identity:
    try:
        return x25519.Identity.from_str(value.strip())
    except Exception as e:
        raise CloudCryptoError(f"Invalid age identity: {e}") from e


def recipient_from_str(value: str) -> x25519.Recipient:
    try:
        return x25519.Recipient.from_str(value.strip())
    except Exception as e:
        raise CloudCryptoError(f"Invalid age recipient: {e}") from e


def recipient_for(identity: x25519.Identity) -> x25519.Recipient:
    return identity.to_public()


def encrypt_bytes(data: bytes, recipients: list[x25519.Recipient]) -> bytes:
    if not recipients:
        raise CloudCryptoError("No recipients to encrypt to.")
    try:
        return pyrage.encrypt(data, recipients)
    except Exception as e:
        raise CloudCryptoError(f"Encryption failed: {e}") from e


def decrypt_bytes(data: bytes, identities: list[x25519.Identity]) -> bytes:
    if not identities:
        raise CloudCryptoError("No identities to decrypt with.")
    try:
        return pyrage.decrypt(data, identities)
    except pyrage.DecryptError as e:
        raise CloudCryptoError(
            "Decryption failed: no matching key. If this bundle predates a key "
            "rotation, make sure the full cloud.key (all identities) is present."
        ) from e
    except Exception as e:
        raise CloudCryptoError(f"Decryption failed: {e}") from e
