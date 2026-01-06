"""HuggingFace embeddings stub.

Functionality targets:
- sentence-transformers (local embeddings - priority)
- HuggingFace Inference API (remote)
"""

from __future__ import annotations

from contextrouter.core.config import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseEmbeddings
from ..registry import model_registry


@model_registry.register_embeddings("hf", "sentence-transformers")
class HuggingFaceEmbeddings(BaseEmbeddings):
    def __init__(self, config: Config) -> None:
        self._cfg = config

    async def embed_query(self, text: str, *, token: BiscuitToken | None = None) -> list[float]:
        _ = text, token
        raise NotImplementedError("Install 'sentence-transformers' to use local HF embeddings.")

    async def embed_documents(
        self, texts: list[str], *, token: BiscuitToken | None = None
    ) -> list[list[float]]:
        _ = texts, token
        raise NotImplementedError("Install 'sentence-transformers' to use local HF embeddings.")


__all__ = ["HuggingFaceEmbeddings"]
