"""PostMaskScanner — validates no PII leaked through masking.
Use case: a doctor name "John D." might appear in a free-text column
like "Error details" that was NOT classified as a PII column.
The scanner catches these leaks.
Two detection strategies:
1. Hash-based: compare against known real values (always available)
2. Presidio-based: ML detection of PII patterns (optional, requires presidio)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import final

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import SecurityError

from contextunity.router.cortex.privacy.masking.contracts import (
    PresidioAnalyzerLike,
    as_pandas_frame,
    load_presidio_analyzer,
)

logger = get_contextunit_logger(__name__)


def _row_label(value: object) -> int | str:
    """Normalize dataframe row identifiers for leak reports."""
    if isinstance(value, (int, str)):
        return value
    return str(value)


@dataclass
class PIILeak:
    """A detected PII leak in masked data."""

    column: str
    row: int | str
    leak_type: str  # "known_value" | "presidio"
    value_preview: str = ""  # First 20 chars for debugging (NOT the full PII!)
    entities: list[str] = field(default_factory=list)  # Presidio entity types


@final
class PostMaskScanner:
    """Scan masked data for residual PII.

    Args:
        known_values_hashes: Set of SHA-256 hashes of known PII values
                             (from MappingStore.get_all_real_values_hashed()).
        use_presidio: Whether to use Presidio for ML-based detection.
        presidio_language: Language for Presidio analysis.
    """

    def __init__(
        self,
        known_values_hashes: set[str] | None = None,
        use_presidio: bool = False,
        presidio_language: str = "uk",
    ) -> None:
        """Create a post-mask scanner.

        Initializes the hash-based detection layer and optionally loads
        the Presidio NLP analyzer for ML-based entity recognition.

        Args:
            known_values_hashes: SHA-256 digests of real PII values to
                detect via exact hash matching. Typically obtained from
                ``MappingStore.get_all_real_values_hashed()``.
            use_presidio: Whether to attempt loading the Presidio analyzer
                for ML-based PII detection as a secondary strategy.
            presidio_language: ISO language code for Presidio analysis
                (e.g. ``"uk"`` for Ukrainian, ``"en"`` for English).
        """
        self._known_hashes = known_values_hashes or set()
        self._language = presidio_language
        self._presidio: PresidioAnalyzerLike | None = None

        if use_presidio:
            try:
                self._presidio = load_presidio_analyzer()
                logger.info("Presidio analyzer initialized for language: %s", presidio_language)
            except ImportError:
                logger.warning(
                    "presidio-analyzer not installed — ML-based PII detection disabled. Install with: pip install presidio-analyzer"
                )

    @staticmethod
    def _hash(value: str) -> str:
        """Compute the SHA-256 digest of a value for hash-based detection.

        Args:
            value: The string to hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def scan_text(self, text: str, source: str = "text") -> list[PIILeak]:
        """Scan free text for residual PII using both detection strategies.

        Checks individual words and bigrams against the known-values hash
        set, then optionally runs Presidio NLP analysis.

        Args:
            text: The masked text to verify.
            source: Label identifying the text origin (used in ``PIILeak.column``).

        Returns:
            List of detected leaks (empty if the text is clean).
        """
        leaks: list[PIILeak] = []

        if not text:
            return leaks

        # Strategy 1: hash-based (known values)
        if self._known_hashes:
            # Check words and multi-word phrases
            words = text.split()
            for i, word in enumerate(words):
                if self._hash(word) in self._known_hashes:
                    leaks.append(
                        PIILeak(
                            column=source,
                            row=i,
                            leak_type="known_value",
                            value_preview=word[:20],
                        )
                    )
                # Also check bigrams (for names like "John D.")
                if i + 1 < len(words):
                    bigram = f"{word} {words[i + 1]}"
                    if self._hash(bigram) in self._known_hashes:
                        leaks.append(
                            PIILeak(
                                column=source,
                                row=i,
                                leak_type="known_value",
                                value_preview=bigram[:20],
                            )
                        )

        # Strategy 2: Presidio (if available)
        if self._presidio:
            try:
                results = self._presidio.analyze(text, language=self._language)
                for result in results:
                    leaks.append(
                        PIILeak(
                            column=source,
                            row=0,
                            leak_type="presidio",
                            entities=[result.entity_type],
                        )
                    )
            except Exception as e:  # graceful-degrade: Presidio optional; scan continues
                logger.warning("Presidio analysis failed: %s", e)

        return leaks

    def scan_dataframe(self, df: object) -> list[PIILeak]:
        """Scan all string columns of a masked DataFrame for residual PII.

        Iterates over every cell in ``object``-typed columns and applies
        both hash-based and Presidio detection strategies.

        Args:
            df: A pandas-like DataFrame (must satisfy ``PandasFrameLike``).

        Returns:
            List of detected leaks across all columns and rows.
        """
        frame = as_pandas_frame(df)

        leaks: list[PIILeak] = []

        for col in frame.select_dtypes(include=["object"]).columns:
            for idx, value in frame[col].items():
                if not isinstance(value, str) or not value.strip():
                    continue

                # Strategy 1: known value hash match
                if self._known_hashes and self._hash(value) in self._known_hashes:
                    leaks.append(
                        PIILeak(
                            column=col,
                            row=_row_label(idx),
                            leak_type="known_value",
                            value_preview=value[:20],
                        )
                    )

                # Strategy 2: Presidio
                if self._presidio:
                    try:
                        results = self._presidio.analyze(value, language=self._language)
                        for result in results:
                            leaks.append(
                                PIILeak(
                                    column=col,
                                    row=_row_label(idx),
                                    leak_type="presidio",
                                    entities=[result.entity_type],
                                )
                            )
                    except Exception as e:  # graceful-degrade: one bad row must not abort scan
                        # Log the failure so silent Per-row drops are
                        # diagnosable in production. A single bad row
                        # must not abort the whole DataFrame scan.
                        logger.warning(
                            "PostMaskScanner: presidio analyze failed for %s[%s]: %s",
                            col,
                            _row_label(idx),
                            e,
                        )

        return leaks


@final
class PIILeakageError(SecurityError):
    """Raised when PII is detected in masked data.

    Inherits from contextunity.core.exceptions.SecurityError so that gRPC error
    handlers map this to PERMISSION_DENIED automatically.
    """

    code: str = "PII_LEAKAGE"

    def __init__(self, leaks: list[PIILeak]) -> None:
        """Construct a PII leakage error from detected leaks.

        The error message includes a summary of up to 5 leak locations
        in ``column[row]:type`` format for triage without exposing the
        actual PII values.

        Args:
            leaks: List of ``PIILeak`` instances that triggered the error.
        """
        self.leaks = leaks
        details = ", ".join(f"{leak.column}[{leak.row}]:{leak.leak_type}" for leak in leaks[:5])
        super().__init__(message=f"PII detected in {len(leaks)} location(s): {details}")
