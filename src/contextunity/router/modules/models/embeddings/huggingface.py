"""HuggingFace embeddings provider.
This implementation is intentionally minimal:
- `hf/sentence-transformers` is registered as the key
- the optional dependency is not installed by default
"""

from __future__ import annotations

import asyncio
from typing import override

from contextunity.router.core import RouterConfig

from ..base import BaseEmbeddings
from ..registry import model_registry
from .boundary import (
    SentenceTransformerModel,
    first_encode_row,
    float_vector_from_encode,
    float_vectors_from_encode_batch,
    load_sentence_transformer,
)


@model_registry.register_embeddings("hf", "sentence-transformers")
class HuggingFaceEmbeddings(BaseEmbeddings):
    """HuggingFace Sentence Transformers embeddings provider."""

    _model: SentenceTransformerModel | None
    _cfg: RouterConfig

    def __init__(self, config: RouterConfig, *, model_name: str | None = None, **_: object) -> None:
        """Store *config* and resolve *model_name* (default: ``all-mpnet-base-v2``)."""
        resolved_name = (model_name or "").strip() or "all-mpnet-base-v2"
        super().__init__(provider="hf", model_name=resolved_name)
        self._cfg = config
        self._model = None

    @override
    async def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single query string."""
        if not text:
            return []
        model = self._ensure_model()
        loop = asyncio.get_running_loop()

        def _encode_query() -> list[float]:
            batch = model.encode([text])
            return float_vector_from_encode(first_encode_row(batch))

        return await loop.run_in_executor(None, _encode_query)

    @override
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of documents."""
        if not texts:
            return []
        model = self._ensure_model()
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(None, lambda: model.encode(texts))
        return float_vectors_from_encode_batch(encoded)

    def _ensure_model(self) -> SentenceTransformerModel:
        """Lazy-load the ``SentenceTransformer`` model; raises ``ImportError`` if the optional dependency is missing."""
        if self._model is not None:
            return self._model
        self._model = load_sentence_transformer(self._model_name)
        return self._model


__all__ = ["HuggingFaceEmbeddings"]
