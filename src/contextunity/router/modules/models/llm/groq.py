"""Groq LLM provider (OpenAI-compatible API).
Groq is known for its ultra-fast inference using custom LPU chips.
It is OpenAI-compatible at the HTTP level.
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
    invoke_openai_chat_create,
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
)
from .openai_compat import build_native_openai_messages, generate_asr_openai_compat
from .types import GroqProviderConfig

logger = get_contextunit_logger(__name__)


@model_registry.register_llm("groq", "*")
class GroqLLM(BaseModel):
    """Groq provider (OpenAI-compatible).

    Features ultra-fast Whisper ASR and vision support for compatible models.
    """

    def __init__(
        self, config: RouterConfig, *, model_name: str | None = None, **kwargs: object
    ) -> None:
        """Create an ``AsyncOpenAI`` client pointing at the Groq LPU endpoint."""
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ConfigurationError(
                "GroqLLM requires `openai` package. Install with `pip install openai`."
            ) from e

        resolved_name = (model_name or "").strip() or "llama-3.3-70b-versatile"
        super().__init__(provider="groq", model_name=resolved_name)
        self._cfg: RouterConfig = config
        self._base_url: str = (
            config.groq.base_url or ""
        ).strip() or "https://api.groq.com/openai/v1"

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=(config.groq.api_key or "skip"),
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
        )
        if kwargs:
            logger.debug("Ignoring unsupported GroqLLM kwargs: %s", sorted(kwargs.keys()))

        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=True
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the Groq backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the OpenAI-compatible Groq endpoint and return a complete response."""
        if any(isinstance(p, AudioPart) for p in request.parts):
            return await generate_asr_openai_compat(
                request,
                base_url=self._base_url,
                api_key=self._cfg.groq.api_key,
                provider="groq",
                whisper_model="whisper-large-v3",
            )

        messages = build_native_openai_messages(request)
        pc = GroqProviderConfig.model_validate(request.provider_config)
        json_mode = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )
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
        )
        text = openai_first_choice_text(response_obj)

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="groq",
                model_name=self._model_name,
                model_key=f"groq/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the OpenAI-compatible Groq endpoint."""
        pc = GroqProviderConfig.model_validate(request.provider_config)
        messages = build_native_openai_messages(request)
        json_mode = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the OpenAI-compatible Groq endpoint stream."""
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
            )
            async for delta in iter_openai_stream_text(stream_obj):
                full += delta
                yield TextDeltaEvent(delta=delta)
            yield FinalTextEvent(text=full)

        return _event_stream()


__all__ = ["GroqLLM"]
