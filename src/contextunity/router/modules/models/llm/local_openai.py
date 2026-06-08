"""Local OpenAI-compatible LLM provider (vLLM/Ollama/etc).
This uses `openai` AsyncOpenAI with a custom base_url to connect
to locally-running OpenAI-compatible servers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, override

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
    FinalTextEvent,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
)
from .openai_compat import build_native_openai_messages
from .types import LocalProviderConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = get_contextunit_logger(__name__)


class _BaseLocalOpenAI(BaseModel):
    """Base class for local OpenAI-compatible providers."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        provider: str,
        base_url: str,
        model_name: str | None = None,
        api_key: object | None = None,
        **kwargs: object,
    ) -> None:
        """Create an ``AsyncOpenAI`` client pointing at the given local *base_url*."""
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ConfigurationError(
                "Local OpenAI-compatible providers require `openai` package."
            ) from e

        self._provider: str = provider
        resolved_name = (model_name or "").strip() or "llama3.1"
        super().__init__(provider=provider, model_name=resolved_name)
        self._base_url: str = base_url.strip()

        api_key_value = api_key if isinstance(api_key, str) and api_key else "local-key"
        self._client: AsyncOpenAI = AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key_value,
            max_retries=config.llm.max_retries,
        )
        if kwargs:
            logger.debug("Ignoring unsupported local provider kwargs: %s", sorted(kwargs.keys()))

        # Local servers support images for vision models; audio requires separate endpoint.
        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=False
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the local OpenAI-compatible backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the local vLLM/Ollama endpoint and return a complete response."""
        messages = build_native_openai_messages(request)
        pc = LocalProviderConfig.model_validate(request.provider_config)
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
                provider=self._provider,
                model_name=self._model_name,
                model_key=f"{self._provider}/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the local vLLM/Ollama endpoint."""
        messages = build_native_openai_messages(request)

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the local vLLM/Ollama endpoint stream."""
            full = ""
            _pc = LocalProviderConfig.model_validate(request.provider_config)
            json_mode = resolve_json_object_mode(
                request_response_format=request.response_format,
                provider_response_format=_pc.response_format,
            )
            stream_obj = await invoke_openai_chat_create(
                self._client,
                model=self._model_name,
                messages=messages,
                timeout=request.timeout_sec,
                stream=True,
                temperature=resolve_temperature(
                    request_temperature=request.temperature,
                    provider_temperature=_pc.temperature,
                ),
                max_tokens=resolve_max_output_tokens(
                    request_max_output_tokens=request.max_output_tokens,
                    provider_max_tokens=_pc.get_max_tokens(),
                ),
                response_format={"type": "json_object"} if json_mode else None,
            )
            async for delta in iter_openai_stream_text(stream_obj):
                full += delta
                yield TextDeltaEvent(delta=delta)
            yield FinalTextEvent(text=full)

        return _event_stream()


@model_registry.register_llm("local", "*")
class LocalOllamaLLM(_BaseLocalOpenAI):
    """Local OpenAI-compatible provider (defaulted to Ollama base URL)."""

    def __init__(
        self, config: RouterConfig, *, model_name: str | None = None, **kwargs: object
    ) -> None:
        """Delegate to base with the Ollama default URL (``localhost:11434``)."""
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

    def __init__(
        self, config: RouterConfig, *, model_name: str | None = None, **kwargs: object
    ) -> None:
        """Delegate to base with the vLLM default URL (``localhost:8000``)."""
        super().__init__(
            config,
            provider="local-vllm",
            base_url=(config.local.vllm_base_url or "http://localhost:8000/v1"),
            model_name=model_name,
            **kwargs,
        )


__all__ = ["LocalOllamaLLM", "LocalVllmLLM"]
