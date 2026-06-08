"""Summarization transformer (stub).
This will be implemented by wiring `model_registry.get_llm()` and producing a
summary field in `envelope.metadata`.
"""

from __future__ import annotations

from typing import override

from contextunity.core import ContextUnit

from contextunity.router.core.registry import register_transformer

from .base import Transformer


@register_transformer("summarizer")
class SummarizationTransformer(Transformer):
    """Stub — registered as ``summarizer`` but not yet wired to an LLM; only marks provenance."""

    name: str = "summarizer"

    @override
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """No-op: append provenance tag and return the unit unchanged."""
        # Placeholder: do not invent summarization logic here.
        return self.with_provenance(unit, self.name)


__all__ = ["SummarizationTransformer"]
