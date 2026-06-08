"""contextunity.router.cortex.privacy Anonymizer — the core privacy proxy pipeline.
Pipeline: request → mask PII → [LLM] → unmask PII → response
This module orchestrates PIIMasker and PIIUnmasker to provide a
transparent anonymization layer for outbound LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import final

from contextunity.core import get_contextunit_logger

from contextunity.router.cortex.privacy.masking import (
    MappingStore,
    MaskingConfig,
    PIIMasker,
    PIIUnmasker,
)

logger = get_contextunit_logger(__name__)

__all__ = ["Anonymizer", "AnonymizationResult"]


@dataclass
class AnonymizationResult:
    """Result of an anonymization or deanonymization operation.

    Attributes:
        text: The processed text (masked or unmasked).
        entities_masked: Count of PII entities found and masked.
        entity_types: Set of entity types detected.
        original_length: Length of original text.
        masked_length: Length of masked text.
    """

    text: str
    entities_masked: int = 0
    entity_types: set[str] = field(default_factory=set)
    original_length: int = 0
    masked_length: int = 0

    @property
    def was_modified(self) -> bool:
        """Whether any PII entities were detected and masked."""
        return self.entities_masked > 0


@final
class Anonymizer:
    """Privacy proxy that masks PII before LLM calls and restores it after.

    Usage:
        store = MappingStore()
        config = MaskingConfig(text_entity_rules=[...])
        anon = Anonymizer(store=store, config=config)

        # Before LLM call
        result = anon.anonymize("Доктор Іваненко призначив лікування")
        send_to_llm(result.text)  # "DOC_7f3a призначив лікування"

        # After LLM response
        response = anon.deanonymize(llm_response_text)
    """

    def __init__(
        self,
        store: MappingStore,
        config: MaskingConfig,
    ) -> None:
        """Create a new privacy proxy.

        Initializes internal ``PIIMasker`` and ``PIIUnmasker`` instances
        and resets per-session anonymization counters.

        Args:
            store: Encrypted mapping store for consistent token generation
                and resolution within the session.
            config: Masking configuration defining PII entity patterns,
                column rules, and transform specifications.
        """
        self._store = store
        self._config = config
        self._masker = PIIMasker(store=store, config=config)
        self._unmasker = PIIUnmasker(store=store)
        self._stats: dict[str, int] = {
            "anonymize_calls": 0,
            "deanonymize_calls": 0,
            "entities_total": 0,
        }

    def anonymize(self, text: str) -> AnonymizationResult:
        """Mask PII in text before sending to an external LLM.

        Args:
            text: Raw text potentially containing PII.

        Returns:
            AnonymizationResult with masked text and metadata.
        """
        if not text:
            return AnonymizationResult(text="", original_length=0, masked_length=0)

        original_length = len(text)
        masked_text = self._masker.mask_text(text)
        masked_length = len(masked_text)

        # Count entities by examining store growth
        entity_types: set[str] = set()
        entities_masked = 0

        # Check what changed
        if masked_text != text:
            # Count token-like patterns in output that weren't in input
            import re

            token_pattern = re.compile(r"[A-Z]{2,5}_[0-9a-f]{4,}")
            tokens_in_output: set[str] = set(token_pattern.findall(masked_text))
            tokens_in_input: set[str] = set(token_pattern.findall(text))
            new_tokens = tokens_in_output - tokens_in_input
            entities_masked = len(new_tokens)

            for token in new_tokens:
                prefix = token.split("_")[0]
                entity_types.add(prefix)

        self._stats["anonymize_calls"] += 1
        self._stats["entities_total"] += entities_masked

        logger.debug(
            "Anonymized: %d entities masked (%s), %d→%d chars",
            entities_masked,
            ", ".join(sorted(entity_types)) or "none",
            original_length,
            masked_length,
        )

        return AnonymizationResult(
            text=masked_text,
            entities_masked=entities_masked,
            entity_types=entity_types,
            original_length=original_length,
            masked_length=masked_length,
        )

    def deanonymize(self, text: str) -> str:
        """Restore PII in LLM response text.

        Args:
            text: LLM response containing tokens (e.g. "DOC_7f3a").

        Returns:
            Text with tokens replaced by original PII values.
        """
        if not text:
            return ""

        restored = self._unmasker.unmask_text(text)
        self._stats["deanonymize_calls"] += 1

        logger.debug("Deanonymized: %d chars", len(restored))
        return restored

    def anonymize_dict(self, data: dict[str, object]) -> dict[str, object]:
        """Mask PII in a dictionary (e.g. ContextUnit payload).

        Args:
            data: Dict with values that may contain PII.

        Returns:
            New dict with PII values replaced by tokens.
        """
        return self._masker.mask_dict(data)

    def get_stats(self) -> dict[str, int]:
        """Return cumulative anonymization statistics for this session.

        Returns:
            Mapping with keys ``"anonymize_calls"``, ``"deanonymize_calls"``,
            and ``"entities_total"``.
        """
        return dict(self._stats)

    def reset(self) -> None:
        """Clear all stored mappings and stats. Useful for testing."""
        self._store.destroy_session()
        self._stats = {"anonymize_calls": 0, "deanonymize_calls": 0, "entities_total": 0}
