"""contextunity.router Privacy — PII anonymization and persona engine.

Core API:
    from contextunity.router.cortex.privacy import Anonymizer, AnonymizationResult
    from contextunity.router.cortex.privacy import PersonaEngine, PersonaTemplate
    from contextunity.router.cortex.privacy.masking import PIIMasker, PIIUnmasker, MappingStore
"""

from contextunity.router.cortex.privacy.anonymizer import AnonymizationResult, Anonymizer
from contextunity.router.cortex.privacy.masking import (
    DEFAULT_MASKING_CONFIG,
    EntityRule,
    MappingStore,
    MaskingConfig,
    PIIMasker,
    PIIUnmasker,
)
from contextunity.router.cortex.privacy.persona import Persona, PersonaEngine, PersonaTemplate

__all__ = [
    "Anonymizer",
    "AnonymizationResult",
    "DEFAULT_MASKING_CONFIG",
    "EntityRule",
    "MappingStore",
    "MaskingConfig",
    "PIIMasker",
    "PIIUnmasker",
    "Persona",
    "PersonaEngine",
    "PersonaTemplate",
]
