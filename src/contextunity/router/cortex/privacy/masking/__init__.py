"""Router Privacy Masking — PII masking gateway (business-agnostic).

Core classes:
    PIIMasker      — mask PII in text and DataFrames
    PIIUnmasker    — unmask PII tokens back to real values
    MappingStore   — encrypted storage for PII↔token mappings
    MaskingConfig  — configuration for masking rules
    EntityRule     — per-entity-type masking rule
    PostMaskScanner — validates no PII leaked through (optional)
    DEFAULT_MASKING_CONFIG — sensible defaults for Ukrainian + international PII
"""

from contextunity.router.cortex.privacy.masking.config import EntityRule, MaskingConfig
from contextunity.router.cortex.privacy.masking.defaults import DEFAULT_MASKING_CONFIG
from contextunity.router.cortex.privacy.masking.masker import PIIMasker
from contextunity.router.cortex.privacy.masking.store import MappingStore
from contextunity.router.cortex.privacy.masking.unmasker import PIIUnmasker

__all__ = [
    "PIIMasker",
    "PIIUnmasker",
    "MappingStore",
    "MaskingConfig",
    "EntityRule",
    "DEFAULT_MASKING_CONFIG",
]
