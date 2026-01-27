"""Transformer base utilities."""

from __future__ import annotations

from abc import abstractmethod

from contextcore import ContextUnit

from contextrouter.core.interfaces import BaseTransformer


class Transformer(BaseTransformer):
    """Convenience base class for transformers."""

    name: str = "transformer"

    def _with_provenance(self, unit: ContextUnit, step: str) -> ContextUnit:
        # Single source-of-truth: append to provenance list.
        if step.strip():
            unit.provenance.append(step.strip())
        return unit

    @abstractmethod
    async def transform(self, envelope: ContextUnit) -> ContextUnit: ...


__all__ = ["Transformer"]
