"""Generic retrieval pipeline (non-RAG).
This is a capability-agnostic pipeline that returns ContextUnit from providers.
It does NOT know about citations, reranking, or RAG-type limits.
Specialized pipelines (e.g. RAG) should compose this and add their own steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextunity.core import ContextUnit
from contextunity.core.types import JsonDict

from contextunity.router.core.types import QueryLike

from .orchestrator import RetrievalOrchestrator


@dataclass(frozen=True)
class PipelineResult:
    """Immutable result of a provider-based retrieval pipeline execution."""

    units: list[ContextUnit]


class BaseRetrievalPipeline:
    """A generic provider-based retrieval pipeline."""

    def __init__(self, *, orchestrator: RetrievalOrchestrator | None = None) -> None:
        """Accept or create the default ``RetrievalOrchestrator``."""
        self._orch: RetrievalOrchestrator = orchestrator or RetrievalOrchestrator()

    async def execute(
        self,
        query: QueryLike,
        *,
        limit: int = 5,
        filters: JsonDict | None = None,
        providers: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the operation task."""
        res = await self._orch.search(
            query,
            limit=limit,
            filters=filters,
            providers=providers,
        )
        return PipelineResult(units=res.units)


__all__ = ["BaseRetrievalPipeline", "PipelineResult"]
