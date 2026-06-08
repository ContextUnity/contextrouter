"""Default PII entity rules for contextunity.router.cortex.privacy.

Business-agnostic patterns that cover common PII types.
Projects can override these by passing a custom MaskingConfig
to PiiSession or Anonymizer.

Usage:
    from contextunity.router.cortex.privacy.masking.defaults import DEFAULT_MASKING_CONFIG
"""

from __future__ import annotations

from pathlib import Path

from contextunity.core import get_contextunit_logger

from contextunity.router.cortex.privacy.masking.config import EntityRule, MaskingConfig

logger = get_contextunit_logger(__name__)

# ── Load Default Config from YAML ─────────────────────────────────

_DEFAULTS_YAML = Path(__file__).parent / "rules" / "defaults.yaml"


def _load_default_masking_config() -> MaskingConfig:
    try:
        return MaskingConfig.from_yaml(_DEFAULTS_YAML)
    except Exception as e:
        logger.warning("Failed to load default MaskingConfig from YAML: %s", e)
        return MaskingConfig()


DEFAULT_MASKING_CONFIG = _load_default_masking_config()

ALL_TEXT_RULES: list[EntityRule] = DEFAULT_MASKING_CONFIG.text_entity_rules

# Export specific rules for backward compatibility if needed
# Note: they are not directly exposed as global variables anymore,
# relying on ALL_TEXT_RULES instead.

__all__ = [
    "ALL_TEXT_RULES",
    "DEFAULT_MASKING_CONFIG",
]
