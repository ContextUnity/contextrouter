"""Reranker abstraction for RAG pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RetrievedDoc


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        documents: list[RetrievedDoc],
        top_n: int | None = None,
        source_type: str | None = None,
    ) -> list[RetrievedDoc]:
        """Return documents sorted by relevance."""


__all__ = ["BaseReranker"]
