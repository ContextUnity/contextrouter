"""Base interfaces for model providers (multimodal LLM + embeddings)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from contextrouter.core.tokens import BiscuitToken

from .types import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
)


class BaseModel(ABC):
    """Multimodal model interface.

    Providers must implement:
    - `capabilities` property
    - `generate()` async method
    - `stream()` async generator

    Optionally override:
    - `get_token_count()` if a real tokenizer is available
    """

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Declare what modalities this model supports."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> ModelResponse:
        """Generate a response from the model."""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream tokens from the model."""
        raise NotImplementedError

    def get_token_count(self, text: str) -> int:
        """Count tokens in text.

        Default implementation uses word splitting as a rough approximation.
        Override this method if the provider has access to a real tokenizer.
        """
        if not text:
            return 0
        return max(1, len(text.split()))


class BaseEmbeddings(ABC):
    """Vectorization model interface."""

    @abstractmethod
    async def embed_query(self, text: str, *, token: BiscuitToken | None = None) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(
        self, texts: list[str], *, token: BiscuitToken | None = None
    ) -> list[list[float]]:
        raise NotImplementedError


__all__ = ["BaseModel", "BaseEmbeddings"]
