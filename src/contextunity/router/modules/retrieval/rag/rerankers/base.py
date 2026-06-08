"""Reranker abstraction -- base class defining the reranking contract for RAG result re-scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RetrievedDoc


class BaseReranker(ABC):
    """Abstract contract for reranking a list of ``RetrievedDoc`` by relevance."""

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
