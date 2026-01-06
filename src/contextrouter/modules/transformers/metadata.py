"""Metadata mapping transformer (pure-ish).

Example usage:
- normalize keys/casing
- enrich with additional metadata
"""

from __future__ import annotations

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.registry import register_transformer

from .base import Transformer


@register_transformer("metadata_mapper")
class MetadataTransformer(Transformer):
    name = "metadata_mapper"

    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        # Deterministic normalization hook (no domain-specific logic here).
        envelope.metadata = dict(envelope.metadata or {})
        return self._with_provenance(envelope, self.name)


__all__ = ["MetadataTransformer"]
