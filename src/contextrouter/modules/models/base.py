"""Base interfaces for model providers (multimodal LLM + embeddings)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from contextrouter.core.tokens import ContextToken

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
    - `generate_batch()` for optimized batch processing
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
        token: ContextToken | None = None,
    ) -> ModelResponse:
        """Generate a response from the model."""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        request: ModelRequest,
        *,
        token: ContextToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream tokens from the model."""
        raise NotImplementedError

    async def generate_batch(
        self,
        requests: list[ModelRequest],
        *,
        token: ContextToken | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for multiple requests.

        Default implementation uses asyncio.gather for parallel execution.
        Providers with native batch APIs (e.g., Gemini) can override for efficiency.

        Note: This is still real-time generation, not async batch jobs.
        For async batch processing (e.g., OpenAI Batch API), use separate methods.
        """
        import asyncio

        results = await asyncio.gather(
            *[self.generate(req, token=token) for req in requests],
            return_exceptions=True,
        )
        # Convert exceptions to proper errors
        responses = []
        for result in results:
            if isinstance(result, Exception):
                from .types import ModelError

                raise ModelError(f"Batch generation failed: {result}")
            responses.append(result)
        return responses

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
    async def embed_query(self, text: str, *, token: ContextToken | None = None) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(
        self, texts: list[str], *, token: ContextToken | None = None
    ) -> list[list[float]]:
        raise NotImplementedError


__all__ = ["BaseModel", "BaseEmbeddings"]
