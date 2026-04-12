"""OpenAI LLM provider (OpenAI API).

This implementation is a thin adapter from the multimodal `BaseModel` contract to
`langchain-openai`'s `ChatOpenAI` so LangGraph can still stream tokens when used
inside a LangChain/LangGraph runnable pipeline.

When `enable_web_search=True`, uses OpenAI's Responses API with web_search_preview
tool for real-time web search capabilities.
"""

from __future__ import annotations

from typing import AsyncIterator

from contextunity.core import get_contextunit_logger
from contextunity.core.tokens import ContextToken

from contextunity.router.core import Config

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
from ._openai_compat import build_native_openai_messages, generate_asr_openai_compat

logger = get_contextunit_logger(__name__)


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
        model_name: str | None = None,
        enable_web_search: bool = False,
        **kwargs: object,
    ) -> None:
        self._cfg = config
        self._model_name = (model_name or "gpt-5.1").strip() or "gpt-5.1"
        self._enable_web_search = enable_web_search
        self._openai_client = None

        self._init_openai_client(**kwargs)

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=True
        )

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

    def _init_openai_client(self, **kwargs: object) -> None:
        """Initialize native OpenAI client."""
        try:
            from langfuse.openai import AsyncOpenAI
        except ImportError:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAILLM requires `openai` package. Install with `pip install openai`."
                ) from e

        api_key = kwargs.pop("api_key", self._cfg.openai.api_key or None)
        base_url = kwargs.pop("base_url", None)

        self._openai_client = AsyncOpenAI(
            api_key=api_key,
            organization=self._cfg.openai.organization,
            base_url=base_url,
            max_retries=0,
            **kwargs,
        )

    async def _generate_chat_completions(
        self, request: ModelRequest, *, token: ContextToken | None = None
    ) -> ModelResponse:
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

        # o1/o3: dedicated reasoning models — need developer role + max_completion_tokens + reasoning_effort
        # gpt-5: reasoning model — uses system role + max_completion_tokens + reasoning_effort
        # others: system role + max_tokens + temperature
        name_lower = self._model_name.lower()
        uses_developer_role = any(x in name_lower for x in ["o1-", "o1_", "o3-", "o3_"])
        uses_completion_tokens = any(x in name_lower for x in ["gpt-5", "o1-", "o1_", "o3-", "o3_"])

        messages = build_native_openai_messages(request, is_reasoning_model=uses_developer_role)

        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "timeout": request.timeout_sec,
        }

        if uses_completion_tokens:
            if request.max_output_tokens:
                kwargs["max_completion_tokens"] = request.max_output_tokens
            # Use per-request reasoning_effort if provided, else config, else omit (API defaults to medium)
            re_value = (
                getattr(request, "reasoning_effort", None) or self._cfg.openai.reasoning_effort
            )
            if re_value:
                kwargs["reasoning_effort"] = re_value
        else:
            if request.max_output_tokens:
                kwargs["max_tokens"] = request.max_output_tokens
            kwargs["temperature"] = request.temperature

        if request.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
            # OpenAI requires the word "json" in messages when using json_object format
            all_text = " ".join(
                m.get("content", "") for m in messages if isinstance(m.get("content"), str)
            )
            if "json" not in all_text.lower():
                for m in reversed(messages):
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        m["content"] += "\nRespond in JSON."
                        break

        provider_info = ProviderInfo(
            provider="openai",
            model_name=self._model_name,
            model_key=f"openai/{self._model_name}",
        )

        logger.info(
            "OpenAI Chat API call: model=%s, msgs=[%s], reasoning_effort=%s, response_format=%s",
            self._model_name,
            ", ".join(f"{m.get('role')}({len(str(m.get('content', '')))!s}ch)" for m in messages),
            kwargs.get("reasoning_effort", "N/A"),
            kwargs.get("response_format", "N/A"),
        )

        try:
            response = await self._openai_client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "insufficient_quota" in error_str or "billing" in error_str:
                raise ModelQuotaExhaustedError(
                    f"OpenAI quota exhausted: {e}", provider_info=provider_info
                ) from e
            elif "rate_limit" in error_str or "too many requests" in error_str:
                raise ModelRateLimitError(
                    f"OpenAI rate limited: {e}", provider_info=provider_info
                ) from e
            raise

        choice = response.choices[0]
        text = str(choice.message.content or "")

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason and finish_reason != "stop":
            logger.warning(
                "OpenAI response finish_reason=%s (model=%s, text_len=%d). "
                "If 'length', response was truncated — increase max_completion_tokens.",
                finish_reason,
                self._model_name,
                len(text),
            )

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
        if self._enable_web_search:
            response = await self.generate(request, token=token)
            yield FinalTextEvent(text=response.text)
            return

        name_lower = self._model_name.lower()
        uses_developer_role = any(x in name_lower for x in ["o1-", "o1_", "o3-", "o3_"])
        uses_completion_tokens = any(x in name_lower for x in ["gpt-5", "o1-", "o1_", "o3-", "o3_"])
        messages = build_native_openai_messages(request, is_reasoning_model=uses_developer_role)

        kwargs = {
            "model": self._model_name,
            "messages": messages,
            "timeout": request.timeout_sec,
            "stream": True,
        }

        if uses_completion_tokens:
            if request.max_output_tokens:
                kwargs["max_completion_tokens"] = request.max_output_tokens
            kwargs["reasoning_effort"] = self._cfg.openai.reasoning_effort or "minimal"
        else:
            if request.max_output_tokens:
                kwargs["max_tokens"] = request.max_output_tokens
            kwargs["temperature"] = request.temperature

        if request.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        full = ""
        try:
            stream = await self._openai_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full += delta
                        yield TextDeltaEvent(delta=delta)
        except Exception:
            # handle gracefully or re-raise based on logic
            raise

        yield FinalTextEvent(text=full)

    def _extract_usage(self, msg: object):
        pass


__all__ = ["OpenAILLM"]
