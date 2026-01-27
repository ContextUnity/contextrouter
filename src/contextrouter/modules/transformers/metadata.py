"""Metadata mapping transformer (pure-ish).

Example usage:
- normalize keys/casing
- enrich with additional metadata
"""

from __future__ import annotations

from contextcore import ContextUnit

from contextrouter.core.registry import register_transformer

from .base import Transformer


@register_transformer("metadata_mapper")
class MetadataTransformer(Transformer):
    name = "metadata_mapper"

    async def transform(self, unit: ContextUnit) -> ContextUnit:
        # Deterministic normalization hook (no domain-specific logic here).
        # Metadata is stored in payload, not as a separate attribute
        payload = unit.payload or {}
        if not isinstance(payload, dict):
            payload = {}
        unit.payload = payload
        return self._with_provenance(unit, self.name)


__all__ = ["MetadataTransformer"]
