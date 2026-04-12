"""RAG Evaluation Framework (LLM-as-a-judge).

This module provides base classes for automated evaluation of RAG responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..models import RetrievedDoc


@dataclass(frozen=True)
class EvalResult:
    score: float  # 0.0 to 1.0
    reasoning: str
    metadata: dict[str, Any]


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    @abstractmethod
    async def evaluate(
        self, query: str, response: str, context: list[RetrievedDoc] | None = None
    ) -> EvalResult:
        """Evaluate a RAG response."""
        raise NotImplementedError


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates if the response is faithful to the retrieved context."""

    async def evaluate(
        self, query: str, response: str, context: list[RetrievedDoc] | None = None
    ) -> EvalResult:
        # Placeholder for LLM-as-a-judge logic
        raise NotImplementedError("FaithfulnessEvaluator requires an LLM judge implementation.")


class RelevancyEvaluator(BaseEvaluator):
    """Evaluates if the response is relevant to the user query."""

    async def evaluate(
        self, query: str, response: str, context: list[RetrievedDoc] | None = None
    ) -> EvalResult:
        # Placeholder for LLM-as-a-judge logic
        raise NotImplementedError("RelevancyEvaluator requires an LLM judge implementation.")
