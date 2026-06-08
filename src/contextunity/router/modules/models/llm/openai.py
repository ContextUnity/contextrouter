"""OpenAI LLM provider (OpenAI API).
This implementation is a thin adapter from the multimodal `BaseModel` contract to
`langchain-openai`'s `ChatOpenAI` so LangGraph can still stream tokens when used
inside a LangChain/LangGraph runnable pipeline.
When `enable_web_search=True`, uses OpenAI's Responses API with web_search_preview
tool for real-time web search capabilities.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.core import RouterConfig

from ..base import BaseLLM as BaseModel
from ..boundary_common import (
    ensure_json_hint_in_openai_messages,
    iter_openai_stream_text,
    openai_first_choice_text,
    resolve_json_object_mode,
    resolve_max_output_tokens,
    resolve_temperature,
)
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
    TextPart,
    UsageStats,
)
from .openai_boundary import (
    await_call,
    choice_finish_reason,
    invoke_openai_chat_create,
    load_async_openai_client,
    responses_output_text,
)
from .openai_compat import build_native_openai_messages, generate_asr_openai_compat
from .types import OpenAIProviderConfig, ReasoningEffort, parse_reasoning_effort

logger = get_contextunit_logger(__name__)


def _raise_typed_openai_error(exc: Exception, provider_info: ProviderInfo) -> None:
    """Re-raise OpenAI SDK exceptions as typed model errors.

    Raises:
        ModelQuotaExhaustedError: On authentication/billing failures.
        ModelRateLimitError: On rate-limit responses.
    """
    from ..types import ModelQuotaExhaustedError, ModelRateLimitError

    try:
        from openai import AuthenticationError, RateLimitError
    except ImportError:
        return  # SDK not available — let caller re-raise original

    if isinstance(exc, RateLimitError):
        raise ModelRateLimitError(
            "OpenAI rate limit exceeded", provider_info=provider_info
        ) from exc
    if isinstance(exc, AuthenticationError):
        raise ModelQuotaExhaustedError(
            "OpenAI authentication failed (quota or billing)", provider_info=provider_info
        ) from exc


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
        config: RouterConfig,
        model_name: str | None = None,
        enable_web_search: bool = False,
        **kwargs: object,
    ) -> None:
        """Create an ``AsyncOpenAI`` client using the direct OpenAI API key and org settings."""
        resolved_name = (model_name or "gpt-5.1").strip() or "gpt-5.1"
        super().__init__(provider="openai", model_name=resolved_name)
        self._cfg: RouterConfig = config
        self._enable_web_search: bool = enable_web_search
        self._openai_client: object | None = None

        self._init_openai_client(**kwargs)

        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=True
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the OpenAI backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the openai SDK and return a complete response."""
        if self._enable_web_search:
            return await self._generate_with_web_search(request)
        else:
            return await self._generate_chat_completions(request)

    async def _generate_with_web_search(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Use the OpenAI Responses API with the ``web_search_preview`` tool."""

        provider_info = ProviderInfo(
            provider="openai",
            model_name=self._model_name,
            model_key=f"openai/{self._model_name}",
        )

        # Build input from request
        input_text = request.system or ""
        for part in request.parts:
            if isinstance(part, TextPart):
                input_text += "\n" + part.text

        logger.debug("OpenAI Responses API request with web_search (model=%s)", self._model_name)

        try:
            if self._openai_client is None:
                raise ConfigurationError("OpenAI client is not initialized")
            responses_api: object = getattr(self._openai_client, "responses", None)
            if responses_api is None:
                raise ConfigurationError(
                    "OpenAI client does not expose responses API",
                    provider="openai",
                )
            create_responses = getattr(responses_api, "create", None)
            if not callable(create_responses):
                raise ConfigurationError(
                    "OpenAI responses API is unavailable",
                    provider="openai",
                )
            response = await await_call(
                create_responses,
                model=self._model_name,
                tools=[{"type": "web_search_preview"}],
                input=input_text,
            )
        except Exception as e:
            _raise_typed_openai_error(e, provider_info)
            raise

        # Extract text from response
        text = responses_output_text(response)

        logger.debug("OpenAI Responses API returned %s chars", len(text))

        # Extract usage if available
        usage = None
        usage_obj: object = getattr(response, "usage", None)
        if usage_obj is not None:
            usage = UsageStats(
                input_tokens=getattr(usage_obj, "input_tokens", 0),
                output_tokens=getattr(usage_obj, "output_tokens", 0),
                total_tokens=getattr(usage_obj, "total_tokens", 0),
            )

        return ModelResponse(
            text=text,
            usage=usage,
            raw_provider=provider_info,
        )

    def _init_openai_client(self, **kwargs: object) -> None:
        """Initialize native OpenAI client."""
        api_key = kwargs.pop("api_key", self._cfg.openai.api_key or None)
        base_url = kwargs.pop("base_url", None)
        api_key_str = api_key if isinstance(api_key, str) else self._cfg.openai.api_key
        base_url_str = base_url if isinstance(base_url, str) else None

        if kwargs:
            raise ConfigurationError(
                "Unsupported OpenAI model init kwargs",
                provider="openai",
                invalid_kwargs=sorted(kwargs.keys()),
            )

        self._openai_client = load_async_openai_client(
            api_key=api_key_str,
            organization=self._cfg.openai.organization,
            base_url=base_url_str,
            max_retries=0,
        )

    def _get_client(self) -> object:
        """Return the async OpenAI SDK client."""
        if self._openai_client is None:
            raise ConfigurationError("OpenAI client is not initialized", provider="openai")
        return self._openai_client

    async def _generate_chat_completions(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Dispatch to ASR (for audio parts) or standard chat completions."""
        if any(isinstance(p, AudioPart) for p in request.parts):
            return await generate_asr_openai_compat(
                request,
                base_url="https://api.openai.com/v1",
                api_key=self._cfg.openai.api_key,
                provider="openai",
                whisper_model="whisper-1",
            )

        # Validate provider config from manifest
        pc = OpenAIProviderConfig.model_validate(request.provider_config)

        # o1/o3: dedicated reasoning models — need developer role + max_completion_tokens + reasoning_effort
        # gpt-5: reasoning model — uses system role + max_completion_tokens + reasoning_effort
        # others: system role + max_tokens + temperature
        name_lower = self._model_name.lower()
        uses_developer_role = any(x in name_lower for x in ["o1-", "o1_", "o3-", "o3_"])
        uses_completion_tokens = any(x in name_lower for x in ["gpt-5", "o1-", "o1_", "o3-", "o3_"])

        messages = build_native_openai_messages(request, is_reasoning_model=uses_developer_role)

        _max_tok = resolve_max_output_tokens(
            request_max_output_tokens=request.max_output_tokens,
            provider_max_tokens=pc.get_max_tokens(),
        )
        max_tokens = _max_tok if not uses_completion_tokens else None
        max_completion_tokens = _max_tok if uses_completion_tokens else None
        temperature = resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        if uses_completion_tokens:
            temperature = None
        reasoning_effort: ReasoningEffort | None = None
        if uses_completion_tokens:
            # Use per-request reasoning_effort if provided, else config, else omit (API defaults to medium)
            raw_re_value = pc.reasoning_effort or self._cfg.openai.reasoning_effort
            reasoning_effort = parse_reasoning_effort(raw_re_value)

        wants_json_object = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )
        if wants_json_object:
            ensure_json_hint_in_openai_messages(messages)

        provider_info = ProviderInfo(
            provider="openai",
            model_name=self._model_name,
            model_key=f"openai/{self._model_name}",
        )

        logger.info(
            "OpenAI Chat API call: model=%s, msgs=[%s], reasoning_effort=%s, response_format=%s",
            self._model_name,
            ", ".join(f"{m.get('role')}({len(str(m.get('content', '')))!s}ch)" for m in messages),
            reasoning_effort if reasoning_effort is not None else "N/A",
            "json_object" if wants_json_object else "N/A",
        )

        try:
            response_obj = await invoke_openai_chat_create(
                self._get_client(),
                model=self._model_name,
                messages=messages,
                timeout=request.timeout_sec,
                temperature=temperature,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"} if wants_json_object else None,
            )
        except Exception as e:
            _raise_typed_openai_error(e, provider_info)
            raise

        choices_obj: object = getattr(response_obj, "choices", None)
        from contextunity.core.types import is_object_list

        from contextunity.router.modules.models.boundary_common import first_list_item

        if not is_object_list(choices_obj) or not choices_obj:
            raise ConfigurationError("OpenAI response missing choices", provider="openai")
        choice = first_list_item(choices_obj)
        if choice is None:
            raise ConfigurationError("OpenAI response missing choices", provider="openai")
        text = openai_first_choice_text(response_obj)

        finish_reason = choice_finish_reason(choice)
        if finish_reason and finish_reason != "stop":
            logger.warning(
                (
                    "OpenAI response finish_reason=%s (model=%s, text_len=%d). "
                    "If 'length', response was truncated — increase max_completion_tokens."
                ),
                finish_reason,
                self._model_name,
                len(text),
            )

        usage_val = None
        usage_obj: object = getattr(response_obj, "usage", None)
        if usage_obj is not None:
            usage_val = UsageStats(
                input_tokens=getattr(usage_obj, "prompt_tokens", 0),
                output_tokens=getattr(usage_obj, "completion_tokens", 0),
                total_tokens=getattr(usage_obj, "total_tokens", 0),
            ).estimate_cost(self._model_name)

        return ModelResponse(
            text=text,
            usage=usage_val,
            raw_provider=provider_info,
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the openai SDK."""
        if self._enable_web_search:

            async def _web_search_stream() -> AsyncIterator[ModelStreamEvent]:
                """Fallback stream via full Responses API call."""
                response = await self._generate(request)
                yield FinalTextEvent(text=response.text)

            return _web_search_stream()

        pc = OpenAIProviderConfig.model_validate(request.provider_config)
        name_lower = self._model_name.lower()
        uses_developer_role = any(x in name_lower for x in ["o1-", "o1_", "o3-", "o3_"])
        uses_completion_tokens = any(x in name_lower for x in ["gpt-5", "o1-", "o1_", "o3-", "o3_"])
        messages = build_native_openai_messages(request, is_reasoning_model=uses_developer_role)

        _max_tok = resolve_max_output_tokens(
            request_max_output_tokens=request.max_output_tokens,
            provider_max_tokens=pc.get_max_tokens(),
        )
        max_tokens = _max_tok if not uses_completion_tokens else None
        max_completion_tokens = _max_tok if uses_completion_tokens else None
        temperature = resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        if uses_completion_tokens:
            temperature = None
        reasoning_effort: ReasoningEffort | None = None
        if uses_completion_tokens:
            raw_re_value = pc.reasoning_effort or self._cfg.openai.reasoning_effort
            reasoning_effort = parse_reasoning_effort(raw_re_value) or "minimal"
        wants_json_object = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )
        if wants_json_object:
            ensure_json_hint_in_openai_messages(messages)

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the openai SDK stream."""
            full = ""
            stream_obj = await invoke_openai_chat_create(
                self._get_client(),
                model=self._model_name,
                messages=messages,
                timeout=request.timeout_sec,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"} if wants_json_object else None,
            )
            async for delta in iter_openai_stream_text(stream_obj):
                full += delta
                yield TextDeltaEvent(delta=delta)
            yield FinalTextEvent(text=full)

        return _event_stream()

    def _extract_usage(self, _msg: object) -> None:
        """Extract token usage from the raw provider response (no-op)."""
        return None


__all__ = ["OpenAILLM"]
