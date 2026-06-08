"""PIIMasker — mask PII in text and DataFrames.
Business-agnostic: the consumer provides column_rules / text_entity_rules
that define WHAT to mask. PIIMasker handles HOW to mask.
Sandwich Architecture:
    Consumer (knows semantics) → PIIMasker (agnostic) → LLM (sees tokens only)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import final

from contextunity.core import get_contextunit_logger

from contextunity.router.cortex.privacy.masking.config import EntityRule, MaskingConfig
from contextunity.router.cortex.privacy.masking.contracts import PandasFrameLike, as_pandas_frame
from contextunity.router.cortex.privacy.masking.store import MappingStore

logger = get_contextunit_logger(__name__)


# Transforms for special columns
def _age_bucket(value: object) -> str:
    raw = str(value)
    if raw.isdigit():
        decade = (int(raw) // 10) * 10
        return f"{decade}-{decade + 9}"
    return raw


TRANSFORMS: dict[str, Callable[[object], str]] = {"age_bucket": _age_bucket}


@final
class PIIMasker:
    """Mask PII values in text and DataFrames.

    Args:
        store: MappingStore for consistent token↔value mapping.
        config: MaskingConfig defining what to mask and how.
    """

    def __init__(self, store: MappingStore, config: MaskingConfig) -> None:
        """Create a new PII masker.

        Pre-compiles regex patterns from ``config.text_entity_rules`` for
        efficient repeated text scanning.

        Args:
            store: Encrypted mapping store for consistent token ↔ value
                lookups within a session.
            config: Masking configuration defining column rules,
                text entity patterns, and transform specifications.
        """
        self._store = store
        self._config = config
        self._compiled_patterns: list[tuple[EntityRule, re.Pattern[str]]] = []

        # Pre-compile text patterns
        for rule in config.text_entity_rules:
            compiled = rule.compiled_pattern()
            if compiled:
                self._compiled_patterns.append((rule, compiled))

    def mask_text(self, text: str) -> str:
        """Mask PII patterns in free text.

        Scans for patterns defined in text_entity_rules and replaces matches
        with consistent tokens.

        Args:
            text: Input text potentially containing PII.

        Returns:
            Text with PII replaced by tokens (e.g. "DOC_7f3a").
        """
        if not text or not self._compiled_patterns:
            return text

        masked = text
        for rule, pattern in self._compiled_patterns:

            def _replace(match: re.Match[str], r: EntityRule = rule) -> str:
                """Replace a regex match with a consistent masking token.

                Args:
                    match: The regex match containing the PII value.
                    r: The entity rule bound via default arg to avoid
                        late-binding closure issues.

                Returns:
                    Token string (e.g. ``"DOC_7f3a"``).
                """
                real_value = match.group(0)
                return self._store.get_or_create_token(real_value, r.entity_type, r.prefix)

            masked = pattern.sub(_replace, masked)

        return masked

    def mask_value(self, value: str, entity_type: str, prefix: str) -> str:
        """Mask a single known PII value.

        Args:
            value: The PII value to mask.
            entity_type: Entity type (e.g. "doctor").
            prefix: Token prefix (e.g. "DOC").

        Returns:
            Token string.
        """
        return self._store.get_or_create_token(value, entity_type, prefix)

    def mask_dataframe(self, df: object) -> PandasFrameLike:
        """Mask PII columns in a pandas DataFrame.

        Applies:
        1. column_rules: replace values with tokens
        2. drop_columns: remove columns entirely
        3. transform_columns: apply value transforms (e.g. age bucketing)

        Args:
            df: pandas DataFrame.

        Returns:
            New DataFrame with PII masked/dropped/transformed.
        """
        frame = as_pandas_frame(df)

        result = frame.copy()
        entity_counts: dict[str, int] = {}

        # 1. Drop columns
        for col in self._config.drop_columns:
            if col in result.columns:
                result = result.drop(columns=[col])
                logger.debug("Dropped column: %s", col)

        # 2. Transform columns
        for col, transform_name in self._config.transform_columns.items():
            if col in result.columns and transform_name in TRANSFORMS:
                result[col] = result[col].apply(TRANSFORMS[transform_name])
                logger.debug("Transformed column: %s → %s", col, transform_name)

        # 3. Mask columns by rules
        for col, rule in self._config.column_rules.items():
            if col not in result.columns:
                continue
            count = 0
            for idx, val in result[col].items():
                if val is not None and str(val).strip():
                    token = self._store.get_or_create_token(str(val), rule.entity_type, rule.prefix)
                    result.at[idx, col] = token
                    count += 1
            entity_counts[rule.entity_type] = count

        # Audit
        self._store.log_audit(
            "mask",
            entity_counts=entity_counts,
            rows_processed=len(result),
        )

        return result

    def mask_dict(
        self, data: dict[str, object], column_rules: dict[str, EntityRule] | None = None
    ) -> dict[str, object]:
        """Mask PII values in a dict using column_rules.

        Args:
            data: Dict with keys matching column names.
            column_rules: Override rules (default: use config.column_rules).

        Returns:
            New dict with PII values replaced by tokens.
        """
        rules = column_rules or self._config.column_rules
        result: dict[str, object] = {}
        for key, value in data.items():
            if key in rules and value is not None and str(value).strip():
                rule = rules[key]
                result[key] = self._store.get_or_create_token(
                    str(value), rule.entity_type, rule.prefix
                )
            else:
                result[key] = value
        return result
