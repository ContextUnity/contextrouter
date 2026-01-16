"""Vertex embeddings provider."""

from __future__ import annotations

import asyncio

from contextrouter.core import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseEmbeddings
from ..registry import model_registry


@model_registry.register_embeddings("vertex", "text-embedding")
class VertexEmbeddings(BaseEmbeddings):
    def __init__(self, config: Config, *, model_name: str | None = None, **_: object) -> None:
        self._cfg = config
        self._model_name = (model_name or "").strip() or "textembedding-gecko@003"
        self._model = None

    async def embed_query(self, text: str, *, token: BiscuitToken | None = None) -> list[float]:
        _ = token
        if not text:
            return []
        model = self._ensure_model()
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, lambda: model.get_embeddings([text])[0].values)
        return [float(x) for x in vec]

    async def embed_documents(
        self, texts: list[str], *, token: BiscuitToken | None = None
    ) -> list[list[float]]:
        _ = token
        if not texts:
            return []
        model = self._ensure_model()
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, lambda: model.get_embeddings(texts))
        return [[float(x) for x in v.values] for v in vecs]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self._cfg.vertex.project_id:
            raise RuntimeError("vertex.project_id must be set for Vertex embeddings")
        if not self._cfg.vertex.location:
            raise RuntimeError("vertex.location must be set for Vertex embeddings")
        try:
            import vertexai  # type: ignore
            from vertexai.preview.language_models import TextEmbeddingModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "vertexai SDK is required for Vertex embeddings. "
                "Install extras: contextrouter[vertex]"
            ) from exc
        vertexai.init(project=self._cfg.vertex.project_id, location=self._cfg.vertex.location)
        self._model = TextEmbeddingModel.from_pretrained(self._model_name)
        return self._model


__all__ = ["VertexEmbeddings"]
