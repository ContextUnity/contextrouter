"""OpenAI LLM provider (OpenAI API).

This implementation is a thin adapter from the multimodal `BaseModel` contract to
`langchain-openai`'s `ChatOpenAI` so LangGraph can still stream tokens when used
inside a LangChain/LangGraph runnable pipeline.

When `enable_web_search=True`, uses OpenAI's Responses API with web_search_preview
tool for real-time web search capabilities.
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

    Args:
        config: Router configuration
        model_name: OpenAI model name (e.g., "gpt-4o", "gpt-5-mini")
        enable_web_search: If True, use Responses API with web_search_preview tool
        **kwargs: Additional LangChain ChatOpenAI kwargs
    """

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        enable_web_search: bool = False,
        **kwargs: object,
    ) -> None:
        self._cfg = config
        self._model_name = (model_name or "gpt-5.1").strip() or "gpt-5.1"
        self._enable_web_search = enable_web_search
        self._langchain_model = None
        self._openai_client = None

        # Initialize appropriate client based on mode
        if enable_web_search:
            # Use native OpenAI client for Responses API
            self._init_openai_client()
        else:
            # Use LangChain for Chat Completions API
            self._init_langchain_model(**kwargs)

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=True
        )

    def _init_langchain_model(self, **kwargs: object) -> None:
        """Initialize LangChain ChatOpenAI for standard chat completions."""
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("OpenAILLM requires `contextrouter[models-openai]`.") from e

        # Disable SDK retries - we handle fallback ourselves via FallbackModel
        self._langchain_model = ChatOpenAI(
            model=self._model_name,
            api_key=(self._cfg.openai.api_key or None),
            organization=self._cfg.openai.organization,
            max_retries=0,  # No SDK retries - fallback handles it
            **kwargs,
        )

    def _init_openai_client(self) -> None:
        """Initialize native OpenAI client for Responses API with web search."""
        try:
            from openai import AsyncOpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OpenAI web search requires `openai>=1.0.0`. Install with: pip install openai"
            ) from e

        if not self._cfg.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required for web search")

        self._openai_client = AsyncOpenAI(api_key=self._cfg.openai.api_key)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        """Generate response using Chat Completions or Responses API."""
        if self._enable_web_search:
            return await self._generate_with_web_search(request, token=token)
        else:
            return await self._generate_chat_completions(request, token=token)

    async def _generate_with_web_search(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        """Generate using OpenAI Responses API with web_search_preview tool."""
        from ..types import ModelQuotaExhaustedError, ModelRateLimitError

        _ = token

        provider_info = ProviderInfo(
            provider="openai",
            model_name=self._model_name,
            model_key=f"openai/{self._model_name}",
        )

        # Build input from request
        input_text = request.system or ""
        for part in request.parts:
            if hasattr(part, "text"):
                input_text += "\n" + part.text

        logger.info("OpenAI Responses API request with web_search (model=%s)", self._model_name)

        try:
            response = await self._openai_client.responses.create(
                model=self._model_name,
                tools=[{"type": "web_search_preview"}],
                input=input_text,
            )
        except Exception as e:
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
            raise

        # Extract text from response
        text = ""
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text = content.text
                        break

        logger.info("OpenAI Responses API returned %s chars", len(text))

        # Extract usage if available
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = UsageStats(
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )

        return ModelResponse(
            text=text,
            usage=usage,
            raw_provider=provider_info,
        )

    async def _generate_chat_completions(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
        """Generate using standard Chat Completions API via LangChain."""
        from ..types import ModelQuotaExhaustedError, ModelRateLimitError

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
        is_reasoning_model = any(
            x in self._model_name.lower() for x in ["gpt-5", "o1-", "o1_", "o3-", "o3_"]
        )

        bind_kwargs: dict = {
            "timeout": request.timeout_sec,
        }

        if is_reasoning_model:
            bind_kwargs["max_completion_tokens"] = request.max_output_tokens
            reasoning_effort = self._cfg.openai.reasoning_effort or "minimal"
            bind_kwargs["reasoning"] = {"effort": reasoning_effort}
        else:
            bind_kwargs["max_tokens"] = request.max_output_tokens
            bind_kwargs["temperature"] = request.temperature

        if request.response_format == "json_object" and not is_reasoning_model:
            bind_kwargs["response_format"] = {"type": "json_object"}

        model = self._langchain_model.bind(**bind_kwargs)

        provider_info = ProviderInfo(
            provider="openai",
            model_name=self._model_name,
            model_key=f"openai/{self._model_name}",
        )

        try:
            msg = await model.ainvoke(messages)
        except Exception as e:
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
            raise

        text = getattr(msg, "content", "")

        # Handle reasoning model responses
        if isinstance(text, list):
            text_parts = []
            for block in text:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            text = "".join(text_parts)

        if not text:
            logger.warning("OpenAI returned empty content. Full response: %s", msg)
            if hasattr(msg, "refusal") and msg.refusal:
                logger.warning("OpenAI REFUSED to respond: %s", msg.refusal)

        usage = self._extract_usage(msg)

        return ModelResponse(
            text=str(text or ""),
            usage=usage,
            raw_provider=provider_info,
        )

    async def stream(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream tokens (only supported for Chat Completions API)."""
        if self._enable_web_search:
            # Web search doesn't support streaming - fall back to generate
            response = await self.generate(request, token=token)
            yield FinalTextEvent(text=response.text)
            return

        _ = token
        messages = build_openai_messages(request)
        bind_kwargs: dict = {
            "max_tokens": request.max_output_tokens,
            "timeout": request.timeout_sec,
        }
        if "gpt-5-mini" not in self._model_name:
            bind_kwargs["temperature"] = request.temperature
        model = self._langchain_model.bind(**bind_kwargs)

        full = ""
        async for chunk in model.astream(messages):
            delta = getattr(chunk, "content", None)
            if isinstance(delta, str) and delta:
                full += delta
                yield TextDeltaEvent(delta=delta)
        yield FinalTextEvent(text=full)

    def _extract_usage(self, msg: object) -> UsageStats | None:
        """Extract usage stats from LangChain message metadata.

        Handles two formats:
        - Chat Completions API: response_metadata.token_usage.{prompt,completion}_tokens
        - Responses API (gpt-5/o-series with reasoning): usage_metadata.{input,output}_tokens
        """
        try:
            # 1. Chat Completions format (response_metadata.token_usage)
            meta = getattr(msg, "response_metadata", None) or {}
            if isinstance(meta, dict):
                u = meta.get("token_usage")
                if isinstance(u, dict) and u.get("prompt_tokens") is not None:
                    return UsageStats(
                        input_tokens=int(u.get("prompt_tokens") or 0),
                        output_tokens=int(u.get("completion_tokens") or 0),
                        total_tokens=int(u.get("total_tokens") or 0),
                    )

            # 2. LangChain usage_metadata (works for both APIs)
            um = getattr(msg, "usage_metadata", None)
            if isinstance(um, dict) and um.get("input_tokens") is not None:
                return UsageStats(
                    input_tokens=int(um.get("input_tokens") or 0),
                    output_tokens=int(um.get("output_tokens") or 0),
                    total_tokens=int(um.get("total_tokens") or 0),
                )
        except Exception:
            pass
        return None


__all__ = ["OpenAILLM"]
