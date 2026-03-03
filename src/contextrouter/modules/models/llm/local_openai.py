"""Local OpenAI-compatible LLM provider (vLLM/Ollama/etc).

This uses `openai` AsyncOpenAI with a custom base_url to connect
to locally-running OpenAI-compatible servers.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import ContextToken

from ..base import BaseModel
from ..registry import model_registry
from ..types import (
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
)
from ._openai_compat import build_native_openai_messages

logger = logging.getLogger(__name__)


class _BaseLocalOpenAI(BaseModel):
    """Base class for local OpenAI-compatible providers."""

    def __init__(
        self,
        config: Config,
        *,
        provider: str,
        base_url: str,
        model_name: str | None = None,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        try:
            from langfuse.openai import AsyncOpenAI
        except ImportError:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "Local OpenAI-compatible providers require `openai` package."
                ) from e

        self._provider = provider
        self._model_name = (model_name or "").strip() or "llama3.1"
        self._base_url = base_url.strip()

        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=(api_key or "local-key"),
            max_retries=config.llm.max_retries,
            **kwargs,
        )

        # Local servers support images for vision models; audio requires separate endpoint.
        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=False
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        _ = token
        messages = build_native_openai_messages(request)
        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "timeout": request.timeout_sec,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = str(choice.message.content or "")

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider=self._provider,
                model_name=self._model_name,
                model_key=f"{self._provider}/{self._model_name}",
            ),
        )

    async def stream(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = token
        messages = build_native_openai_messages(request)
        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "timeout": request.timeout_sec,
            "stream": True,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens

        full = ""
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content
                if delta:
                    full += delta
                    yield TextDeltaEvent(delta=delta)
        yield FinalTextEvent(text=full)


@model_registry.register_llm("local", "*")
class LocalOllamaLLM(_BaseLocalOpenAI):
    """Local OpenAI-compatible provider (defaulted to Ollama base URL)."""

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        super().__init__(
            config,
            provider="local",
            base_url=(config.local.ollama_base_url or "http://localhost:11434/v1"),
            model_name=model_name,
            **kwargs,
        )


@model_registry.register_llm("local-vllm", "*")
class LocalVllmLLM(_BaseLocalOpenAI):
    """Local OpenAI-compatible provider (defaulted to vLLM base URL)."""

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        super().__init__(
            config,
            provider="local-vllm",
            base_url=(config.local.vllm_base_url or "http://localhost:8000/v1"),
            model_name=model_name,
            **kwargs,
        )


__all__ = ["LocalOllamaLLM", "LocalVllmLLM"]
