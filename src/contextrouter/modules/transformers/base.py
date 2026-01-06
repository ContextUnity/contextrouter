"""Transformer base utilities."""

from __future__ import annotations

from abc import abstractmethod

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseTransformer


class Transformer(BaseTransformer):
    """Convenience base class for transformers."""

    name: str = "transformer"

    def _with_provenance(self, envelope: BisquitEnvelope, step: str) -> BisquitEnvelope:
        # Single source-of-truth: BisquitEnvelope.add_trace appends to provenance.
        envelope.add_trace(step)
        return envelope

    @abstractmethod
    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope: ...


__all__ = ["Transformer"]
