"""Inception Labs LLM provider (OpenAI-compatible API).

Mercury-2 is a diffusion-based language model from Inception Labs.
It is fully OpenAI-compatible at the HTTP level.

Docs: https://docs.inceptionlabs.ai/get-started/models
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
    UsageStats,
)
from ._openai_compat import build_native_openai_messages

logger = logging.getLogger(__name__)


@model_registry.register_llm("inception", "*")
class InceptionLLM(BaseModel):
    """Inception Labs provider (OpenAI-compatible).

    Mercury-2 features:
    - 128k token context window (max 50k output tokens)
    - Diffusion-based generation (not autoregressive)
    - Tool use / function calling support
    - Structured outputs via response_format
    - Reasoning effort control: instant, low, medium, high
    - Temperature range: 0.5 – 1.0 (default 0.75)
    """

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        try:
            from langfuse.openai import AsyncOpenAI
        except ImportError:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "InceptionLLM requires `openai` package. Install with `pip install openai`."
                ) from e

        self._cfg = config
        self._model_name = (model_name or "").strip() or "mercury-2"

        # Support kwargs overrides (e.g. from ContextUnit metadata/payload)
        kw_api_key = kwargs.pop("api_key", None)
        kw_base_url = kwargs.pop("base_url", None)

        self._base_url = (
            kw_base_url or config.inception.base_url or "https://api.inceptionlabs.ai/v1"
        ).strip()

        # If project passed an API key via ContextUnit, use it instead of router env
        final_api_key = kw_api_key or config.inception.api_key or None

        self._client = AsyncOpenAI(
            api_key=final_api_key,
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
            **kwargs,
        )

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=False, supports_audio=False
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        from ..types import ModelQuotaExhaustedError, ModelRateLimitError

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

        # Pass reasoning_effort from config if set
        reasoning_effort = self._cfg.inception.reasoning_effort
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if request.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        provider_info = ProviderInfo(
            provider="inception",
            model_name=self._model_name,
            model_key=f"inception/{self._model_name}",
        )

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "insufficient_quota" in error_str or "billing" in error_str:
                raise ModelQuotaExhaustedError(
                    f"Inception API quota exhausted: {e}", provider_info=provider_info
                ) from e
            elif "rate_limit" in error_str or "too many requests" in error_str:
                raise ModelRateLimitError(
                    f"Inception API rate limited: {e}", provider_info=provider_info
                ) from e
            raise

        choice = response.choices[0]
        text = str(choice.message.content or "")

        usage_val = None
        if hasattr(response, "usage") and response.usage:
            usage_val = UsageStats(
                input_tokens=getattr(response.usage, "prompt_tokens", 0),
                output_tokens=getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            ).estimate_cost(self._model_name)

        return ModelResponse(
            text=text,
            usage=usage_val,
            raw_provider=provider_info,
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

        reasoning_effort = self._cfg.inception.reasoning_effort
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        full = ""
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full += delta
                        yield TextDeltaEvent(delta=delta)
        except Exception:
            raise

        yield FinalTextEvent(text=full)


__all__ = ["InceptionLLM"]
