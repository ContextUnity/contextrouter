"""Transformer base utilities -- shared helpers and abstract contracts for data transformation steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from contextunity.core import ContextUnit
from contextunity.core.sdk.interfaces import JsonConfigurableTransformer


class Transformer(JsonConfigurableTransformer, ABC):
    """Convenience base for JSON-configured Router transformers."""

    name: str = "transformer"

    def with_provenance(self, unit: ContextUnit, step: str) -> ContextUnit:
        """Append *step* to *unit.provenance* and return the unit for chaining."""
        if step.strip():
            unit.provenance.append(step.strip())
        return unit

    def _with_provenance(self, unit: ContextUnit, step: str) -> ContextUnit:
        """Backward-compatible wrapper for legacy callers."""
        return self.with_provenance(unit, step)

    @override
    @abstractmethod
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """Apply this transformer’s logic to *unit* and return the mutated result."""


__all__ = ["Transformer"]
