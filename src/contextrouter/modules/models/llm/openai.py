"""OpenAI LLM provider (OpenAI API).

This implementation is a thin adapter from the multimodal `BaseModel` contract to
`langchain-openai`'s `ChatOpenAI` so LangGraph can still stream tokens when used
inside a LangChain/LangGraph runnable pipeline.
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
    UsageStats,
)
from ._openai_compat import (
    build_openai_messages,
    generate_asr_openai_compat,
)

logger = logging.getLogger(__name__)


@model_registry.register_llm("openai", "*")
class OpenAILLM(BaseModel):
    """OpenAI LLM provider.

    Supports multimodal inputs (text + images) and audio (ASR via Whisper).
    """

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        **kwargs: object,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("OpenAILLM requires `contextrouter[models-openai]`.") from e

        self._cfg = config
        self._model_name = (model_name or "gpt-5.1").strip() or "gpt-5.1"
        # Disable SDK retries - we handle fallback ourselves via FallbackModel
        # This prevents wasting time on quota errors that will never succeed
        self._model = ChatOpenAI(
            model=self._model_name,
            api_key=(config.openai.api_key or None),
            organization=config.openai.organization,
            max_retries=0,  # No SDK retries - fallback handles it
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
        from ..types import ModelQuotaExhaustedError, ModelRateLimitError, ProviderInfo
        
        _ = token
        if any(isinstance(p, AudioPart) for p in request.parts):
            return await generate_asr_openai_compat(
                request,
                base_url="https://api.openai.com/v1",
                api_key=self._cfg.openai.api_key,
                provider="openai",
                whisper_model="whisper-1",
            )

        messages = build_openai_messages(request)
        
        # Reasoning models (gpt-5*, o1*) require max_completion_tokens, not max_tokens
        # They also don't support custom temperature (must be 1)
        is_reasoning_model = any(
            x in self._model_name.lower() 
            for x in ["gpt-5", "o1-", "o1_", "o3-", "o3_"]
        )
        
        bind_kwargs: dict = {
            "timeout": request.timeout_sec,
        }
        
        if is_reasoning_model:
            # Reasoning models need more tokens for CoT + response
            # Use max_completion_tokens (not max_tokens!)
            bind_kwargs["max_completion_tokens"] = request.max_output_tokens
            # Temperature must be 1 for reasoning models
        else:
            bind_kwargs["max_tokens"] = request.max_output_tokens
            bind_kwargs["temperature"] = request.temperature
            
        model = self._model.bind(**bind_kwargs)
        
        provider_info = ProviderInfo(
            provider="openai",
            model_name=self._model_name,
            model_key=f"openai/{self._model_name}",
        )
        
        try:
            msg = await model.ainvoke(messages)
        except Exception as e:
            # Convert OpenAI-specific errors to our error types for proper fallback
            error_str = str(e).lower()
            if "insufficient_quota" in error_str or "billing" in error_str:
                raise ModelQuotaExhaustedError(
                    f"OpenAI quota exhausted: {e}",
                    provider_info=provider_info,
                ) from e
            elif "rate_limit" in error_str or "too many requests" in error_str:
                raise ModelRateLimitError(
                    f"OpenAI rate limited: {e}",
                    provider_info=provider_info,
                ) from e
            raise  # Re-raise other errors as-is
            
        text = getattr(msg, "content", "")
        
        # Debug: log when content is empty
        if not text:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"OpenAI returned empty content. Full response: {msg}")
            # Check for refusal or other issues
            if hasattr(msg, "refusal") and msg.refusal:
                logger.warning(f"OpenAI REFUSED to respond: {msg.refusal}")

        usage = self._extract_usage(msg)

        return ModelResponse(
            text=str(text or ""),
            usage=usage,
            raw_provider=provider_info,
        )

    async def stream(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = token
        messages = build_openai_messages(request)
        # gpt-5-mini only supports temperature=1, skip if model doesn't support it
        bind_kwargs: dict = {
            "max_tokens": request.max_output_tokens,
            "timeout": request.timeout_sec,
        }
        if "gpt-5-mini" not in self._model_name:
            bind_kwargs["temperature"] = request.temperature
        model = self._model.bind(**bind_kwargs)

        full = ""
        async for chunk in model.astream(messages):
            delta = getattr(chunk, "content", None)
            if isinstance(delta, str) and delta:
                full += delta
                yield TextDeltaEvent(delta=delta)
        yield FinalTextEvent(text=full)

    def _extract_usage(self, msg: object) -> UsageStats | None:
        """Extract usage stats from LangChain message metadata."""
        try:
            meta = getattr(msg, "response_metadata", None) or {}
            u = meta.get("token_usage") if isinstance(meta, dict) else None
            if isinstance(u, dict):
                return UsageStats(
                    input_tokens=int(u.get("prompt_tokens") or 0),
                    output_tokens=int(u.get("completion_tokens") or 0),
                    total_tokens=int(u.get("total_tokens") or 0),
                )
        except Exception:
            pass
        return None


__all__ = ["OpenAILLM"]
