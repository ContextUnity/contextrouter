"""Encryption backends for real_value storage.

Even if the mapping database is compromised, real values
cannot be read without the encryption key.

Backends:
- EphemeralAES256Backend: production — AES-256-GCM with RAM-only keys
- FernetBackend: production (AES-128-CBC + HMAC-SHA256) — requires `cryptography`
- PlaintextBackend: development/testing ONLY
"""

from __future__ import annotations

import os
import time
import warnings
from typing import ClassVar, Protocol, final

from contextunity.core import get_contextunit_logger

from contextunity.router.core.exceptions import RouterPIIError

logger = get_contextunit_logger(__name__)


class EncryptionBackend(Protocol):
    """Protocol defining the encryption contract for PII value storage.

    Implementations must provide symmetric encrypt/decrypt operations
    suitable for protecting real PII values in ``MappingStore``. The
    wire format (ciphertext layout, key management) is backend-specific.
    """

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt a plaintext PII value into an opaque ciphertext blob.

        Args:
            plaintext: The real PII value to protect (e.g. a patient name).

        Returns:
            Ciphertext bytes in the backend's wire format.
        """
        ...

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt a ciphertext blob back to the original plaintext value.

        Args:
            ciphertext: Opaque bytes produced by a prior ``encrypt`` call.

        Returns:
            The original plaintext PII string.
        """
        ...


class _AESGCMSeal(Protocol):
    """Structural protocol matching ``cryptography.hazmat.primitives.ciphers.aead.AESGCM``.

    Used to type-annotate AES-GCM instances without importing the
    ``cryptography`` package at module level (it's an optional dependency).
    """

    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes:
        """Encrypt data with a unique nonce and optional associated data.

        Args:
            nonce: 12-byte unique nonce (must never be reused with the same key).
            data: Plaintext bytes to encrypt.
            associated_data: Optional authenticated-but-unencrypted context.

        Returns:
            Ciphertext bytes including the 16-byte GCM authentication tag.
        """
        ...

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes:
        """Decrypt ciphertext and verify the GCM authentication tag.

        Args:
            nonce: The same 12-byte nonce used during encryption.
            data: Ciphertext bytes (including the appended 16-byte tag).
            associated_data: The same AAD used during encryption, or ``None``.

        Returns:
            Decrypted plaintext bytes.
        """
        ...


@final
class EphemeralAES256Backend:
    """AES-256-GCM encryption with ephemeral keys stored ONLY in RAM.

    Security properties:
    - Keys are generated via os.urandom (CSPRNG) and never written to disk
    - Each key has a TTL; after expiry a new key is generated
    - Old keys are retained in memory for decryption until garbage collected
    - Keys are 32 bytes (256 bits) for AES-256
    - Each encryption uses a unique 12-byte nonce (prepended to ciphertext)

    Usage:
        backend = EphemeralAES256Backend(key_ttl_seconds=3600)
        encrypted = backend.encrypt("sensitive data")
        decrypted = backend.decrypt(encrypted)  # "sensitive data"
    """

    # Wire format: key_id (4 bytes) + nonce (12 bytes) + ciphertext + tag (16 bytes)
    _KEY_ID_LEN: ClassVar[int] = 4
    _NONCE_LEN: ClassVar[int] = 12

    def __init__(self, key_ttl_seconds: int = 3600) -> None:
        """Create a new ephemeral AES-256-GCM backend.

        Generates an initial 256-bit key via ``os.urandom`` and stores it
        exclusively in RAM. Keys older than ``3 × key_ttl_seconds`` are
        automatically purged on rotation.

        Args:
            key_ttl_seconds: Lifetime of each key in seconds before rotation.
                Defaults to 3600 (1 hour).

        Raises:
            ImportError: If the ``cryptography`` package is not installed.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            _ = AESGCM
        except ImportError as e:
            raise ImportError(
                "EphemeralAES256Backend requires `cryptography`. Install via: pip install cryptography"
            ) from e

        self._key_ttl = key_ttl_seconds
        # key_id (bytes) -> (AESGCM instance, created_at)
        self._keys: dict[bytes, tuple[_AESGCMSeal, float]] = {}
        self._active_key_id: bytes = b""
        self._rotate_key()

    def _rotate_key(self) -> None:
        """Generate a new AES-256 key. Old keys are kept for decryption."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = os.urandom(32)  # 256 bits — RAM only, never persisted
        key_id = os.urandom(self._KEY_ID_LEN)
        aesgcm = AESGCM(key)
        self._keys[key_id] = (aesgcm, time.monotonic())
        self._active_key_id = key_id

        # Purge keys older than 3x TTL (they can't decrypt anything useful)
        # Never purge the active key — even if TTL=0
        min_ttl = max(self._key_ttl, 1)  # floor at 1s to avoid self-purge
        cutoff = time.monotonic() - (min_ttl * 3)
        expired = [
            kid for kid, (_, ts) in self._keys.items() if ts < cutoff and kid != self._active_key_id
        ]
        for kid in expired:
            del self._keys[kid]

        logger.debug("Rotated ephemeral AES-256 key (active keys in RAM: %d)", len(self._keys))

    def _get_active_aesgcm(self) -> tuple[bytes, _AESGCMSeal]:
        """Return the active key, rotating first if the TTL has elapsed.

        Returns:
            A ``(key_id, aesgcm)`` pair for the currently active key.
        """
        _, created_at = self._keys[self._active_key_id]
        if time.monotonic() - created_at > self._key_ttl:
            self._rotate_key()
        return self._active_key_id, self._keys[self._active_key_id][0]

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt plaintext with AES-256-GCM.

        Wire format: ``key_id (4 B) ‖ nonce (12 B) ‖ ciphertext ‖ tag (16 B)``.
        The key_id prefix allows ``decrypt`` to locate the correct key even
        after rotation.

        Args:
            plaintext: The PII value to protect.

        Returns:
            Self-describing ciphertext blob (see wire format above).
        """
        key_id, aesgcm = self._get_active_aesgcm()
        nonce = os.urandom(self._NONCE_LEN)
        ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return key_id + nonce + ct

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt ciphertext by extracting the key_id prefix and looking up the key.

        Args:
            ciphertext: Blob produced by ``encrypt`` (key_id ‖ nonce ‖ ct ‖ tag).

        Returns:
            The original plaintext PII string.

        Raises:
            ValueError: If the blob is too short or the key has been rotated out
                (older than ``3 × key_ttl_seconds``).
        """
        if len(ciphertext) < self._KEY_ID_LEN + self._NONCE_LEN + 1:
            raise RouterPIIError("Ciphertext too short")

        key_id = ciphertext[: self._KEY_ID_LEN]
        nonce = ciphertext[self._KEY_ID_LEN : self._KEY_ID_LEN + self._NONCE_LEN]
        ct = ciphertext[self._KEY_ID_LEN + self._NONCE_LEN :]

        entry = self._keys.get(key_id)
        if entry is None:
            raise RouterPIIError(
                "Unknown key_id — the ephemeral key has been rotated out. "
                + "This happens when encrypted data is older than 3x key_ttl."
            )
        aesgcm = entry[0]
        return aesgcm.decrypt(nonce, ct, None).decode("utf-8")

    def destroy_all_keys(self) -> None:
        """Wipe all keys from RAM. After this, NO decryption is possible."""
        self._keys.clear()
        self._active_key_id = b""
        logger.info("All ephemeral AES-256 keys destroyed")

    @property
    def active_key_count(self) -> int:
        """Number of AES-256 keys currently held in RAM.

        Includes both the active key and any retained keys pending
        garbage collection (up to ``3 × key_ttl_seconds`` old).
        """
        return len(self._keys)


@final
class FernetBackend:
    """Production encryption using cryptography.fernet.

    Key must be a URL-safe base64-encoded 32-byte key.
    Generate with: Fernet.generate_key()
    """

    def __init__(self, key: str | bytes) -> None:
        """Create a Fernet encryption backend.

        Args:
            key: A URL-safe base64-encoded 32-byte key. Generate with
                ``cryptography.fernet.Fernet.generate_key()``. Accepts
                both ``str`` and ``bytes`` representations.
        """
        from cryptography.fernet import Fernet

        if isinstance(key, str):
            key = key.encode("utf-8")
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt plaintext using Fernet (AES-128-CBC + HMAC-SHA256).

        Args:
            plaintext: The PII value to encrypt.

        Returns:
            Fernet token bytes (URL-safe base64, includes timestamp).
        """
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt a Fernet token back to the original plaintext.

        Args:
            ciphertext: Fernet token bytes produced by ``encrypt``.

        Returns:
            The original plaintext PII string.
        """
        return self._fernet.decrypt(ciphertext).decode("utf-8")


@final
class PlaintextBackend:
    """Development/testing only — NO real encryption.

    Raises a warning on instantiation.
    """

    def __init__(self, *, suppress_warning: bool = False) -> None:
        """Create a no-op plaintext backend (development/testing only).

        Emits a ``UserWarning`` on instantiation unless *suppress_warning* is set
        (e.g. when ``allow_plaintext_pii`` is explicitly enabled for local dev).
        """
        if not suppress_warning:
            warnings.warn(
                "PlaintextBackend provides NO encryption. Use EphemeralAES256Backend in production.",
                UserWarning,
                stacklevel=2,
            )

    def encrypt(self, plaintext: str) -> bytes:
        """Return plaintext as UTF-8 bytes (no actual encryption).

        Args:
            plaintext: The value to "encrypt" (stored as-is).

        Returns:
            UTF-8 encoded bytes of the original value.
        """
        return plaintext.encode("utf-8")

    def decrypt(self, ciphertext: bytes) -> str:
        """Decode UTF-8 bytes back to string (no actual decryption).

        Args:
            ciphertext: UTF-8 bytes produced by ``encrypt``.

        Returns:
            The original plaintext string.
        """
        return ciphertext.decode("utf-8")


__all__ = [
    "EncryptionBackend",
    "EphemeralAES256Backend",
    "FernetBackend",
    "PlaintextBackend",
]
