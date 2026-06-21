"""Vertex AI embedding provider -- Google Cloud text-embedding model adapter."""

from __future__ import annotations

import asyncio
from typing import override

from contextunity.core.exceptions import ConfigurationError

from contextunity.router.core import RouterConfig

from ..base import BaseEmbeddings
from ..registry import model_registry
from .boundary import (
    VertexTextEmbeddingModel,
    init_vertexai,
    load_vertex_text_embedding_model,
    vertex_embedding_values,
)


@model_registry.register_embeddings("vertex", "text-embedding")
class VertexEmbeddings(BaseEmbeddings):
    """Vertex AI ``TextEmbeddingModel`` adapter — delegates to the Google Cloud SDK via a thread executor."""

    _cfg: RouterConfig
    _model: VertexTextEmbeddingModel | None

    def __init__(self, config: RouterConfig, *, model_name: str | None = None, **_: object) -> None:
        """Store *config* and resolve *model_name* (default: ``textembedding-gecko@003``)."""
        resolved_name = (model_name or "").strip() or "textembedding-gecko@003"
        super().__init__(provider="vertex", model_name=resolved_name)
        self._cfg = config
        self._model = None

    @override
    async def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single query string."""
        if not text:
            return []
        model = self._ensure_model()
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(None, lambda: vertex_embedding_values(model, [text]))
        return rows[0] if rows else []

    @override
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of documents."""
        if not texts:
            return []
        model = self._ensure_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: vertex_embedding_values(model, texts))

    def _ensure_model(self) -> VertexTextEmbeddingModel:
        """Lazy-init the Vertex AI SDK and ``TextEmbeddingModel``; raises ``RuntimeError`` if project/location is missing."""
        if self._model is not None:
            return self._model
        if not self._cfg.vertex.project_id:
            raise ConfigurationError("vertex.project_id must be set for Vertex embeddings")
        if not self._cfg.vertex.location:
            raise ConfigurationError("vertex.location must be set for Vertex embeddings")
        init_vertexai(self._cfg.vertex.project_id, self._cfg.vertex.location)
        self._model = load_vertex_text_embedding_model(self._model_name)
        return self._model


__all__ = ["VertexEmbeddings"]
