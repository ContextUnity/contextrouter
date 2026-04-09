"""RunPod LLM provider (OpenAI-compatible API).

RunPod Serverless provides OpenAI-compatible endpoints for vLLM and TGI.
Custom workers can support additional modalities.
"""

from __future__ import annotations

from typing import AsyncIterator

from contextcore import get_context_unit_logger
from contextcore.tokens import ContextToken

from contextrouter.core import Config

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

logger = get_context_unit_logger(__name__)


@model_registry.register_llm("runpod", "*")
class RunPodLLM(BaseModel):
    """RunPod provider (OpenAI-compatible).

    Supports text and images for vision models deployed on RunPod.
    """

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        try:
            from langfuse.openai import AsyncOpenAI
        except ImportError:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "RunPodLLM requires `openai` package. Install with `pip install openai`."
                ) from e

        self._cfg = config
        self._model_name = (model_name or "").strip() or "runpod-model"
        self._base_url = (config.runpod.base_url or "").strip()

        self._client = AsyncOpenAI(
            api_key=(config.runpod.api_key or "skip"),
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
            **kwargs,
        )

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
                provider="runpod",
                model_name=self._model_name,
                model_key=f"runpod/{self._model_name}",
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


__all__ = ["RunPodLLM"]
