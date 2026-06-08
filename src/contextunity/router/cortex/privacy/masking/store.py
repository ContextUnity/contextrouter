"""MappingStore — encrypted storage for PII ↔ token mappings.
Provides consistent masking within a session: same real value always
maps to the same token. Different sessions are isolated.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Literal, Protocol, TypeGuard, final

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError
from contextunity.core.parsing import json_dumps

from contextunity.router.cortex.privacy.masking.contracts import (
    MaskingAuditMetadata,
    MaskingEntityCounts,
    MaskingOperation,
)
from contextunity.router.cortex.privacy.masking.encryption import (
    EncryptionBackend,
    EphemeralAES256Backend,
    PlaintextBackend,
)
from contextunity.router.cortex.privacy.masking.tokens import TokenGenerator

logger = get_contextunit_logger(__name__)

# Token regex for safe unmask scanning
TOKEN_PATTERN = r"[A-Z]{2,5}_[a-f0-9]{4,12}"  # nosec B105 — regex pattern, not a password


class _SqliteCursorLike(Protocol):
    """Typed fetch boundary for sqlite cursors."""

    def fetchone(self) -> object: ...


class _SqliteConnectionLike(Protocol):
    """Typed sqlite connection boundary used by the masking store."""

    def execute(self, sql: str, parameters: tuple[object, ...] = (), /) -> _SqliteCursorLike: ...

    def executescript(self, sql_script: str, /) -> object: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


def _sqlite_fetchone(cursor: _SqliteCursorLike) -> object:
    """Return one sqlite row as an untrusted object boundary."""
    return cursor.fetchone()


def _is_single_value_row(value: object) -> TypeGuard[tuple[object]]:
    """Validate a single-column sqlite row (token ``str`` or ciphertext ``bytes``)."""
    match value:
        case (str(),) | (bytes(),):
            return True
        case _:
            return False


def _is_entity_count_row(value: object) -> TypeGuard[tuple[str, int]]:
    """Validate ``SELECT entity_type, COUNT(*)`` sqlite row."""
    match value:
        case (str(), int()):
            return True
        case _:
            return False


@final
class MappingStore:
    """Encrypted storage for PII value ↔ token mappings.

    Thread-safe via SQLite's own locking. Session-isolated.

    Args:
        session_id: Isolation boundary (e.g. "tenant-a-session", "chat-abc123").
        db_path: Path to SQLite file. Use ":memory:" for testing.
        encryption: Encryption backend for real values.
        token_style: Token generation strategy.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str | Path = ":memory:",
        encryption: EncryptionBackend | None = None,
        token_style: Literal["random_hex", "uuid", "sequential"] = "random_hex",  # nosec B107
    ) -> None:
        """Create a new mapping store backed by SQLite.

        Initializes the database connection, creates the schema tables
        (``shield_identity_map`` and ``shield_masking_audit``), and selects
        the best available encryption backend.

        Args:
            session_id: Isolation boundary ensuring one session's mappings
                are invisible to another (e.g. ``"tenant-a-session"``).
            db_path: Path to the SQLite file. Use ``":memory:"`` for
                ephemeral testing stores.
            encryption: Encryption backend for real PII values. Defaults
                to ``EphemeralAES256Backend`` if ``cryptography`` is available,
                otherwise falls back to ``PlaintextBackend``.
            token_style: Token generation strategy (``"random_hex"`` for 4-char
                hex suffixes, ``"uuid"`` for 12-char, ``"sequential"`` for
                monotonic counters).
        """
        self.session_id = session_id
        self._db_path = str(db_path)
        self._enc = encryption or self._default_encryption()
        self._token_gen = TokenGenerator(style=token_style)
        self._conn = self._connect()
        self._ensure_schema()

    @staticmethod
    def _default_encryption() -> EncryptionBackend:
        """Select the best available encryption backend.

        Prefers ``EphemeralAES256Backend`` (AES-256-GCM, RAM-only keys).
        Falls back to ``PlaintextBackend`` when the ``cryptography``
        package is not installed.

        Returns:
            A ready-to-use encryption backend instance.
        """
        try:
            return EphemeralAES256Backend()
        except ImportError:
            return PlaintextBackend()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL journaling and foreign keys enabled.

        Returns:
            A configured ``sqlite3.Connection`` instance.
        """
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        _ = conn.execute("PRAGMA journal_mode=WAL")
        _ = conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        _ = self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS shield_identity_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                real_value_hash TEXT NOT NULL,
                real_value_enc BLOB NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, token),
                UNIQUE(session_id, entity_type, real_value_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_shield_map_lookup
                ON shield_identity_map(session_id, entity_type, real_value_hash);

            CREATE INDEX IF NOT EXISTS idx_shield_map_token
                ON shield_identity_map(session_id, token);

            CREATE TABLE IF NOT EXISTS shield_masking_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('mask', 'unmask', 'scan', 'destroy')),
                entity_counts TEXT,
                rows_processed INTEGER,
                leaks_detected INTEGER DEFAULT 0,
                performed_at TEXT DEFAULT (datetime('now')),
                performed_by TEXT,
                metadata TEXT
            );
        """)

    @staticmethod
    def _hash(value: str) -> str:
        """Compute the SHA-256 digest of a value for lookup without storing plaintext.

        Args:
            value: The PII string to hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def get_or_create_token(self, real_value: str, entity_type: str, prefix: str) -> str:
        """Return existing token or create a new one. Consistent within session.

        Args:
            real_value: The actual PII value (e.g. "John Doe").
            entity_type: Semantic type (e.g. "doctor").
            prefix: Token prefix (e.g. "DOC").

        Returns:
            Token string like "DOC_7f3a".
        """
        val_hash = self._hash(real_value)
        conn: _SqliteConnectionLike = self._conn

        # Try existing
        existing_cursor = conn.execute(
            """SELECT token FROM shield_identity_map
               WHERE session_id = ? AND entity_type = ? AND real_value_hash = ?""",
            (self.session_id, entity_type, val_hash),
        )
        row_obj = _sqlite_fetchone(existing_cursor)
        if _is_single_value_row(row_obj):
            value = row_obj[0]
            if isinstance(value, str):
                return value

        # Create new token (retry on collision)
        for _attempt in range(10):
            token = self._token_gen.generate(prefix)
            try:
                _ = conn.execute(
                    """INSERT INTO shield_identity_map
                       (session_id, entity_type, real_value_hash, real_value_enc, token)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        self.session_id,
                        entity_type,
                        val_hash,
                        self._enc.encrypt(real_value),
                        token,
                    ),
                )
                conn.commit()
                return token
            except sqlite3.IntegrityError:
                continue  # token collision, retry

        msg = f"Failed to generate unique token for {entity_type} after 10 attempts"
        raise PlatformServiceError(msg)

    def resolve_token(self, token: str) -> str | None:
        """Decrypt and return the real PII value for a masking token.

        Args:
            token: The masking token to resolve (e.g. ``"DOC_7f3a"``).

        Returns:
            The original plaintext value, or ``None`` if the token is not
            found in this session.
        """
        conn: _SqliteConnectionLike = self._conn
        token_cursor = conn.execute(
            """SELECT real_value_enc FROM shield_identity_map
               WHERE session_id = ? AND token = ?""",
            (self.session_id, token),
        )
        row_obj = _sqlite_fetchone(token_cursor)
        if _is_single_value_row(row_obj):
            value = row_obj[0]
            if isinstance(value, bytes):
                return self._enc.decrypt(value)
        return None

    def resolve_all_tokens(self, text: str) -> str:
        """Find all masking tokens in text and replace them with real values.

        Scans the text for patterns matching ``TOKEN_PATTERN`` and resolves
        each via ``resolve_token``. Unknown tokens are left as-is.

        Args:
            text: Text containing masking tokens to unmask.

        Returns:
            Text with all known tokens replaced by their real PII values.
        """
        import re

        def replacer(match: re.Match[str]) -> str:
            """Substitute a token match with its real value if available.

            Args:
                match: Regex match containing a token string.

            Returns:
                The real PII value, or the original token if not found.
            """
            token = match.group(0)
            real = self.resolve_token(token)
            return real if real is not None else token

        return re.sub(TOKEN_PATTERN, replacer, text)

    def get_session_stats(self) -> dict[str, int]:
        """Return per-entity-type counts of masked values in this session.

        Returns:
            Mapping from entity type (e.g. ``"doctor"``) to the number
            of distinct values masked.
        """
        conn: _SqliteConnectionLike = self._conn
        rows_cursor = conn.execute(
            """SELECT entity_type, COUNT(*) FROM shield_identity_map
               WHERE session_id = ? GROUP BY entity_type""",
            (self.session_id,),
        )
        stats: dict[str, int] = {}
        while True:
            row_obj = _sqlite_fetchone(rows_cursor)
            if row_obj is None:
                break
            if not _is_entity_count_row(row_obj):
                continue
            entity_type, count = row_obj
            stats[entity_type] = count
        return stats

    def get_all_real_values_hashed(self) -> set[str]:
        """Return SHA-256 hashes of all real values in this session.

        Used by ``PostMaskScanner`` for hash-based leak detection without
        exposing actual PII values.

        Returns:
            Set of hex-encoded SHA-256 digests.
        """
        conn: _SqliteConnectionLike = self._conn
        rows_cursor = conn.execute(
            """SELECT real_value_hash FROM shield_identity_map
               WHERE session_id = ?""",
            (self.session_id,),
        )
        hashes: set[str] = set()
        while True:
            row_obj = _sqlite_fetchone(rows_cursor)
            if row_obj is None:
                break
            if not _is_single_value_row(row_obj):
                continue
            value = row_obj[0]
            if isinstance(value, str):
                hashes.add(value)
        return hashes

    def destroy_session(self) -> None:
        """Permanently delete all mappings for this session."""
        _ = self._conn.execute(
            "DELETE FROM shield_identity_map WHERE session_id = ?",
            (self.session_id,),
        )
        self.log_audit("destroy")
        self._conn.commit()
        logger.info("Destroyed session: %s", self.session_id)

    def log_audit(
        self,
        operation: MaskingOperation,
        entity_counts: MaskingEntityCounts | None = None,
        rows_processed: int = 0,
        leaks_detected: int = 0,
        metadata: MaskingAuditMetadata | None = None,
    ) -> None:
        """Write an entry to the ``shield_masking_audit`` table.

        Every mask, unmask, scan, and destroy operation should be audited
        for compliance traceability.

        Args:
            operation: The type of masking operation (``"mask"``,
                ``"unmask"``, ``"scan"``, or ``"destroy"``).
            entity_counts: Per-entity-type counts of values processed.
            rows_processed: Total number of data rows involved.
            leaks_detected: Number of PII leaks found (relevant for scan).
            metadata: Arbitrary JSON-serializable audit context.
        """
        _ = self._conn.execute(
            """INSERT INTO shield_masking_audit
               (session_id, operation, entity_counts, rows_processed,
                leaks_detected, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                self.session_id,
                operation,
                json_dumps(entity_counts) if entity_counts else None,
                rows_processed,
                leaks_detected,
                json_dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        """Ensure the SQLite connection is closed on garbage collection."""
        try:
            self._conn.close()
        except Exception:
            pass
