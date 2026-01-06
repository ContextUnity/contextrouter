"""Generic retrieval pipeline (non-RAG).

This is a capability-agnostic pipeline that returns BisquitEnvelopes from providers.
It does NOT know about citations, reranking, or RAG-type limits.

Specialized pipelines (e.g. RAG) should compose this and add their own steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextrouter.core import BiscuitToken, BisquitEnvelope
from contextrouter.core.types import QueryLike

from .orchestrator import RetrievalOrchestrator


@dataclass(frozen=True)
class PipelineResult:
    envelopes: list[BisquitEnvelope]


class BaseRetrievalPipeline:
    """A generic provider-based retrieval pipeline."""

    def __init__(self, *, orchestrator: RetrievalOrchestrator | None = None) -> None:
        self._orch = orchestrator or RetrievalOrchestrator()

    async def execute(
        self,
        query: QueryLike,
        *,
        token: BiscuitToken,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        providers: list[str] | None = None,
    ) -> PipelineResult:
        res = await self._orch.search(
            query,
            limit=limit,
            filters=filters,
            token=token,
            providers=providers,
        )
        return PipelineResult(envelopes=res.envelopes)


__all__ = ["BaseRetrievalPipeline", "PipelineResult"]
