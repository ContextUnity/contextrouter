"""Vertex AI Ranking API reranker -- re-scores RAG results using Google Cloud semantic ranking."""

from __future__ import annotations

from typing import override

from ..models import RetrievedDoc
from ..ranking import rerank_documents
from .base import BaseReranker


class VertexReranker(BaseReranker):
    """Reranker backed by Vertex AI Discovery Engine ranking API."""

    @override
    async def rerank(
        self,
        *,
        query: str,
        documents: list[RetrievedDoc],
        top_n: int | None = None,
        source_type: str | None = None,
    ) -> list[RetrievedDoc]:
        """Send *documents* to the Vertex Ranking API and return them re-scored by semantic relevance to *query*."""
        return await rerank_documents(
            query=query,
            documents=documents,
            top_n=top_n,
            source_type=source_type,
        )


__all__ = ["VertexReranker"]
