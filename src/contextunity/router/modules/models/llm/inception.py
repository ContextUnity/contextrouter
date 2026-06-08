"""Inception Labs LLM provider (OpenAI-compatible API).
Mercury-2 is a diffusion-based language model from Inception Labs.
It is fully OpenAI-compatible at the HTTP level.
Docs: https://docs.inceptionlabs.ai/get-started/models
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.core import RouterConfig

from ..base import BaseLLM as BaseModel
from ..boundary_common import (
    ensure_json_hint_in_openai_messages,
    invoke_openai_chat_create,
    iter_openai_stream_text,
    openai_first_choice_text,
    resolve_json_object_mode,
    resolve_max_output_tokens,
    resolve_temperature,
)
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
from .openai_compat import build_native_openai_messages
from .types import InceptionProviderConfig, parse_reasoning_effort

logger = get_contextunit_logger(__name__)


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

    def __init__(
        self, config: RouterConfig, *, model_name: str | None = None, **kwargs: object
    ) -> None:
        """Create an ``AsyncOpenAI`` client targeting the Inception Labs diffusion-model endpoint."""
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ConfigurationError(
                "InceptionLLM requires `openai` package. Install with `pip install openai`."
            ) from e

        resolved_name = (model_name or "").strip() or "mercury-2"
        super().__init__(provider="inception", model_name=resolved_name)
        self._cfg: RouterConfig = config

        # Support kwargs overrides (e.g. from ContextUnit metadata/payload)
        kw_api_key = kwargs.pop("api_key", None)
        kw_base_url = kwargs.pop("base_url", None)
        kw_base_url_str = kw_base_url if isinstance(kw_base_url, str) else None

        self._base_url: str = (
            kw_base_url_str or config.inception.base_url or "https://api.inceptionlabs.ai/v1"
        ).strip()

        # If project passed an API key via ContextUnit, use it instead of router env
        final_api_key = (
            kw_api_key if isinstance(kw_api_key, str) else (config.inception.api_key or None)
        )

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=final_api_key,
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
        )
        if kwargs:
            logger.debug("Ignoring unsupported InceptionLLM kwargs: %s", sorted(kwargs.keys()))

        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True, supports_image=False, supports_audio=False
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the Inception backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the OpenAI-compatible Inception endpoint and return a complete response."""
        from ..types import ModelQuotaExhaustedError, ModelRateLimitError

        messages = build_native_openai_messages(request)

        pc = InceptionProviderConfig.model_validate(request.provider_config)
        reasoning_effort = parse_reasoning_effort(self._cfg.inception.reasoning_effort)
        json_mode = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )
        if json_mode:
            ensure_json_hint_in_openai_messages(messages)

        provider_info = ProviderInfo(
            provider="inception",
            model_name=self._model_name,
            model_key=f"inception/{self._model_name}",
        )

        try:
            response_obj = await invoke_openai_chat_create(
                self._client,
                model=self._model_name,
                messages=messages,
                timeout=request.timeout_sec,
                temperature=resolve_temperature(
                    request_temperature=request.temperature,
                    provider_temperature=pc.temperature,
                ),
                max_tokens=resolve_max_output_tokens(
                    request_max_output_tokens=request.max_output_tokens,
                    provider_max_tokens=pc.get_max_tokens(),
                ),
                response_format={"type": "json_object"} if json_mode else None,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            try:
                from openai import AuthenticationError, RateLimitError
            except ImportError:
                raise e

            if isinstance(e, RateLimitError):
                raise ModelRateLimitError(
                    "Inception API rate limit exceeded", provider_info=provider_info
                ) from e
            if isinstance(e, AuthenticationError):
                raise ModelQuotaExhaustedError(
                    "Inception API authentication failed", provider_info=provider_info
                ) from e
            raise

        text = openai_first_choice_text(response_obj)

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
        """Stream token deltas from the OpenAI-compatible Inception endpoint."""
        pc = InceptionProviderConfig.model_validate(request.provider_config)
        messages = build_native_openai_messages(request)
        reasoning_effort = parse_reasoning_effort(self._cfg.inception.reasoning_effort)
        json_mode = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )
        if json_mode:
            ensure_json_hint_in_openai_messages(messages)

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the OpenAI-compatible Inception endpoint stream."""
            full = ""
            stream_obj = await invoke_openai_chat_create(
                self._client,
                model=self._model_name,
                messages=messages,
                timeout=request.timeout_sec,
                stream=True,
                temperature=resolve_temperature(
                    request_temperature=request.temperature,
                    provider_temperature=pc.temperature,
                ),
                max_tokens=resolve_max_output_tokens(
                    request_max_output_tokens=request.max_output_tokens,
                    provider_max_tokens=pc.get_max_tokens(),
                ),
                response_format={"type": "json_object"} if json_mode else None,
                reasoning_effort=reasoning_effort,
            )
            async for delta in iter_openai_stream_text(stream_obj):
                full += delta
                yield TextDeltaEvent(delta=delta)

            yield FinalTextEvent(text=full)

        return _event_stream()


__all__ = ["InceptionLLM"]
