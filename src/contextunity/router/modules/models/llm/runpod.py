"""RunPod LLM provider (OpenAI-compatible API).
RunPod Serverless provides OpenAI-compatible endpoints for vLLM and TGI.
Custom workers can support additional modalities.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.core import RouterConfig

from ..base import BaseLLM as BaseModel
from ..boundary_common import (
    invoke_openai_chat_create,
    iter_openai_stream_text,
    openai_choice_text,
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
from .openai_boundary import load_async_openai_client
from .openai_compat import build_native_openai_messages
from .types import RunPodProviderConfig

logger = get_contextunit_logger(__name__)


@model_registry.register_llm("runpod", "*")
class RunPodLLM(BaseModel):
    """RunPod provider (OpenAI-compatible).

    Supports text and images for vision models deployed on RunPod.
    """

    _client: object
    _cfg: RouterConfig
    _base_url: str
    _capabilities: ModelCapabilities

    def __init__(
        self, config: RouterConfig, *, model_name: str | None = None, **kwargs: object
    ) -> None:
        """Create an ``AsyncOpenAI`` client targeting the RunPod serverless endpoint."""
        resolved_name = (model_name or "").strip() or "runpod-model"
        super().__init__(provider="runpod", model_name=resolved_name)
        self._cfg = config
        self._base_url = (config.runpod.base_url or "").strip()

        self._client = load_async_openai_client(
            api_key=(config.runpod.api_key or "skip"),
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
            **kwargs,
        )

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=False
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the RunPod backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the OpenAI-compatible RunPod serverless endpoint and return a complete response."""
        messages = build_native_openai_messages(request)
        pc = RunPodProviderConfig.model_validate(request.provider_config)
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
        choices_obj: object = getattr(response_obj, "choices", None)
        from contextunity.core.types import is_object_list

        from contextunity.router.modules.models.boundary_common import first_list_item

        if not is_object_list(choices_obj) or not choices_obj:
            raise ConfigurationError("RunPod returned no completion choices")
        first_choice = first_list_item(choices_obj)
        if first_choice is None:
            raise ConfigurationError("RunPod response missing choices", provider="runpod")
        text = openai_choice_text(first_choice)

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="runpod",
                model_name=self._model_name,
                model_key=f"runpod/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the OpenAI-compatible RunPod serverless endpoint."""
        messages = build_native_openai_messages(request)
        pc = RunPodProviderConfig.model_validate(request.provider_config)
        json_mode = resolve_json_object_mode(
            request_response_format=request.response_format,
            provider_response_format=pc.response_format,
        )

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the OpenAI-compatible RunPod serverless endpoint stream."""
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


__all__ = ["RunPodLLM"]
