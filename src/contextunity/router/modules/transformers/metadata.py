"""Metadata mapping transformer (pure-ish).
Example usage:
- normalize keys/casing
- enrich with additional metadata
"""

from __future__ import annotations

from typing import override

from contextunity.core import ContextUnit

from contextunity.router.core.registry import register_transformer

from .base import Transformer


@register_transformer("metadata_mapper")
class MetadataTransformer(Transformer):
    """Deterministic metadata normalization hook — ensures ``payload`` is a dict."""

    name: str = "metadata_mapper"

    @override
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """Normalize payload to a shallow dict copy and mark provenance."""
        # Deterministic normalization hook (no domain-specific logic here).
        # Metadata is stored in payload, not as a separate attribute
        unit.payload = dict(unit.payload)
        return self.with_provenance(unit, self.name)


__all__ = ["MetadataTransformer"]
