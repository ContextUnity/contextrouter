"""Groq LLM provider (OpenAI-compatible API).

Groq is known for its ultra-fast inference using custom LPU chips.
It is OpenAI-compatible at the HTTP level.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import ContextToken

from ..base import BaseModel
from ..registry import model_registry
from ..types import (
    AudioPart,
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
)
from ._openai_compat import build_native_openai_messages, generate_asr_openai_compat

logger = logging.getLogger(__name__)


@model_registry.register_llm("groq", "*")
class GroqLLM(BaseModel):
    """Groq provider (OpenAI-compatible).

    Features ultra-fast Whisper ASR and vision support for compatible models.
    """

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        try:
            from langfuse.openai import AsyncOpenAI
        except ImportError:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "GroqLLM requires `openai` package. Install with `pip install openai`."
                ) from e

        self._cfg = config
        self._model_name = (model_name or "").strip() or "llama-3.3-70b-versatile"
        self._base_url = (config.groq.base_url or "").strip() or "https://api.groq.com/openai/v1"

        self._client = AsyncOpenAI(
            api_key=(config.groq.api_key or "skip"),
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
            **kwargs,
        )

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=True
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        _ = token
        if any(isinstance(p, AudioPart) for p in request.parts):
            return await generate_asr_openai_compat(
                request,
                base_url=self._base_url,
                api_key=self._cfg.groq.api_key,
                provider="groq",
                whisper_model="whisper-large-v3",
            )

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
                provider="groq",
                model_name=self._model_name,
                model_key=f"groq/{self._model_name}",
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


__all__ = ["GroqLLM"]
