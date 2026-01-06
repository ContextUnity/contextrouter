"""Vertex embeddings provider (placeholder).

Current contextrouter runtime uses Vertex AI Search for retrieval and does not
depend on explicit embedding generation. This module exists for future providers
and flow pipelines that require embeddings.
"""

from __future__ import annotations

from contextrouter.core.config import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseEmbeddings
from ..registry import model_registry


@model_registry.register_embeddings("vertex", "text-embedding")
class VertexEmbeddings(BaseEmbeddings):
    def __init__(self, config: Config) -> None:
        self._cfg = config

    async def embed_query(self, text: str, *, token: BiscuitToken | None = None) -> list[float]:
        _ = text, token
        raise NotImplementedError(
            "VertexEmbeddings is not implemented in this codebase yet. "
            "Current retrieval uses Vertex AI Search; add a Vertex embeddings backend when needed."
        )

    async def embed_documents(
        self, texts: list[str], *, token: BiscuitToken | None = None
    ) -> list[list[float]]:
        _ = texts, token
        raise NotImplementedError(
            "VertexEmbeddings is not implemented in this codebase yet. "
            "Current retrieval uses Vertex AI Search; add a Vertex embeddings backend when needed."
        )


__all__ = ["VertexEmbeddings"]
