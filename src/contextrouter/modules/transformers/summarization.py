"""Summarization transformer (stub).

This will be implemented by wiring `model_registry.get_llm()` and producing a
summary field in `envelope.metadata`.
"""

from __future__ import annotations

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.registry import register_transformer

from .base import Transformer


@register_transformer("summarizer")
class SummarizationTransformer(Transformer):
    name = "summarizer"

    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        # Placeholder: do not invent summarization logic here.
        return self._with_provenance(envelope, self.name)


__all__ = ["SummarizationTransformer"]
