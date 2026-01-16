"""HuggingFace embeddings provider.

This implementation is intentionally minimal:
- `hf/sentence-transformers` is registered as the key
- the optional dependency is not installed by default
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from contextrouter.core import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseEmbeddings
from ..registry import model_registry

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@model_registry.register_embeddings("hf", "sentence-transformers")
class HuggingFaceEmbeddings(BaseEmbeddings):
    """HuggingFace Sentence Transformers embeddings provider."""

    _model: "SentenceTransformer | None"
    _model_name: str

    def __init__(self, config: Config, *, model_name: str | None = None, **_: object) -> None:
        self._cfg = config
        self._model = None
        self._model_name = (model_name or "").strip() or "all-mpnet-base-v2"

    async def embed_query(self, text: str, *, token: BiscuitToken | None = None) -> list[float]:
        _ = token
        if not text:
            return []
        model = self._ensure_model()
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, lambda: model.encode([text])[0])
        return [float(x) for x in vec.tolist()]

    async def embed_documents(
        self, texts: list[str], *, token: BiscuitToken | None = None
    ) -> list[list[float]]:
        _ = token
        if not texts:
            return []
        model = self._ensure_model()
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, lambda: model.encode(texts))
        return [[float(x) for x in v.tolist()] for v in vecs]

    def _ensure_model(self) -> "SentenceTransformer":
        """Lazy-load the SentenceTransformer model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformers requires `contextrouter[hf-embeddings]`."
            ) from e
        self._model = SentenceTransformer(self._model_name)
        return self._model


__all__ = ["HuggingFaceEmbeddings"]
