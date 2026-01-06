"""Base interfaces for model providers (LLM + embeddings)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Sequence

from contextrouter.core.tokens import BiscuitToken


class BaseLLM(ABC):
    """Text generation model interface."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        tools: Sequence[Any] | None = None,
        *,
        token: BiscuitToken | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, prompt: str, *, token: BiscuitToken | None = None) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def as_chat_model(self) -> Any:
        """Return underlying chat model object (LangChain), when available."""
        raise NotImplementedError


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


__all__ = ["BaseLLM", "BaseEmbeddings"]
