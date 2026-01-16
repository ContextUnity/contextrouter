"""Vertex AI Ranking API reranker."""

from __future__ import annotations

from ..models import RetrievedDoc
from ..ranking import rerank_documents
from .base import BaseReranker


class VertexReranker(BaseReranker):
    async def rerank(
        self,
        *,
        query: str,
        documents: list[RetrievedDoc],
        top_n: int | None = None,
        source_type: str | None = None,
    ) -> list[RetrievedDoc]:
        return await rerank_documents(
            query=query,
            documents=documents,
            top_n=top_n,
            source_type=source_type,
        )


__all__ = ["VertexReranker"]
