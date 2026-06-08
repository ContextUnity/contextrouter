"""Perplexity LLM provider (Perplexity Sonar API).
Perplexity provides LLM with built-in search capabilities.
Uses Llama models with real-time web search.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import override

import httpx
from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.sdk.payload import get_int, get_json_dict, get_str
from contextunity.core.types import JsonDict, is_json_dict, is_object_list

from contextunity.router.core import RouterConfig

from ..base import BaseLLM as BaseModel
from ..registry import model_registry
from ..types import (
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
from .types import PerplexityProviderConfig

logger = get_contextunit_logger(__name__)

# Default model
DEFAULT_MODEL = "sonar"


def _first_choice_content(data: Mapping[str, object]) -> str:
    """Extract assistant text from an OpenAI-style chat completion payload."""
    choices_raw = data.get("choices")
    if not is_object_list(choices_raw) or not choices_raw:
        return ""
    first_choice_obj: object = choices_raw[0]
    if not is_json_dict(first_choice_obj):
        return ""
    message = get_json_dict(first_choice_obj, "message")
    return get_str(message, "content")


def _extract_usage(data: Mapping[str, object]) -> UsageStats | None:
    """Extract token usage statistics from the provider response."""
    usage_raw = data.get("usage")
    if not is_json_dict(usage_raw):
        return None
    return UsageStats(
        input_tokens=get_int(usage_raw, "prompt_tokens"),
        output_tokens=get_int(usage_raw, "completion_tokens"),
        total_tokens=get_int(usage_raw, "total_tokens"),
    )


def _build_request_payload(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    pc: PerplexityProviderConfig,
    request: ModelRequest,
    search_recency_filter: str | None,
    return_citations: bool,
    stream: bool = False,
) -> dict[str, object]:
    """Build the Perplexity chat-completions JSON body."""
    from ..boundary_common import (
        resolve_json_object_mode,
        resolve_max_output_tokens,
        resolve_temperature,
    )

    payload: dict[str, object] = {
        "model": model_name,
        "messages": messages,
        "temperature": resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        or 0.7,
    }
    _max = resolve_max_output_tokens(
        request_max_output_tokens=request.max_output_tokens,
        provider_max_tokens=pc.get_max_tokens(),
    )
    if _max:
        payload["max_tokens"] = _max
    if resolve_json_object_mode(
        request_response_format=request.response_format,
        provider_response_format=pc.response_format,
    ):
        payload["response_format"] = {"type": "json_object"}
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    if return_citations:
        payload["return_citations"] = return_citations
    if stream:
        payload["stream"] = True
    return payload


@model_registry.register_llm("perplexity", "*")
class PerplexityLLM(BaseModel):
    """Perplexity LLM provider with built-in search.

    Supports:
    - Real-time web search (sonar-*-online models)
    - Citations and sources
    - Search recency filtering
    """

    def __init__(
        self,
        config: RouterConfig,
        *,
        model_name: str | None = None,
        search_recency_filter: str | None = "day",
        return_citations: bool = True,
        **kwargs: object,
    ) -> None:
        """Create an ``AsyncOpenAI`` client targeting the Perplexity search-augmented inference API."""
        resolved_name = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        super().__init__(provider="perplexity", model_name=resolved_name)
        self._cfg: RouterConfig = config
        self._search_recency_filter: str | None = search_recency_filter
        self._return_citations: bool = return_citations
        self._base_url: str = "https://api.perplexity.ai"

        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True,
            supports_image=False,
            supports_audio=False,
        )
        if kwargs:
            logger.debug(
                "Ignoring unsupported PerplexityLLM kwargs: %s",
                sorted(kwargs.keys()),
            )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the Perplexity backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the Perplexity chat completions API and return a complete response."""
        pc = PerplexityProviderConfig.model_validate(request.provider_config)
        messages = self._build_messages(request)
        payload = _build_request_payload(
            model_name=self._model_name,
            messages=messages,
            pc=pc,
            request=request,
            search_recency_filter=self._search_recency_filter,
            return_citations=self._return_citations,
        )

        logger.debug(
            "Perplexity request: model=%s, recency=%s",
            self._model_name,
            self._search_recency_filter,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._cfg.perplexity.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=request.timeout_sec or 60.0,
            )
            _ = response.raise_for_status()
            data_raw: object = json_loads(response.text)
            if not is_json_dict(data_raw):
                data: JsonDict = {}
            else:
                data = data_raw

        content = _first_choice_content(data)
        usage = _extract_usage(data)

        return ModelResponse(
            text=content,
            usage=usage,
            raw_provider=ProviderInfo(
                provider="perplexity",
                model_name=self._model_name,
                model_key=f"perplexity/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the Perplexity chat completions API."""
        pc = PerplexityProviderConfig.model_validate(request.provider_config)
        messages = self._build_messages(request)
        payload = _build_request_payload(
            model_name=self._model_name,
            messages=messages,
            pc=pc,
            request=request,
            search_recency_filter=self._search_recency_filter,
            return_citations=self._return_citations,
            stream=True,
        )

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the Perplexity chat completions API stream."""
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._cfg.perplexity.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=request.timeout_sec or 60.0,
                ) as response:
                    _ = response.raise_for_status()
                    full = ""

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_raw: object = json_loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if not is_json_dict(chunk_raw):
                            continue
                        choices_raw = chunk_raw.get("choices")
                        if not isinstance(choices_raw, list) or not choices_raw:
                            continue
                        first_choice = choices_raw[0]
                        if not is_json_dict(first_choice):
                            continue
                        delta = get_json_dict(first_choice, "delta")
                        delta_text = get_str(delta, "content")
                        if delta_text:
                            full += delta_text
                            yield TextDeltaEvent(delta=delta_text)

                    yield FinalTextEvent(text=full)

        return _event_stream()

    def _build_messages(self, request: ModelRequest) -> list[dict[str, str]]:
        """Convert ``ModelRequest`` to Perplexity messages format."""
        messages: list[dict[str, str]] = []

        if request.system:
            messages.append(
                {
                    "role": "system",
                    "content": request.system,
                }
            )

        user_content = ""
        for part in request.parts:
            if isinstance(part, TextPart):
                user_content += part.text

        if user_content:
            messages.append(
                {
                    "role": "user",
                    "content": user_content,
                }
            )

        return messages


__all__ = ["PerplexityLLM"]
