"""Anthropic LLM provider (Claude) via `langchain-anthropic`.
This provider is the same class of integration as OpenAI/Vertex: remote HTTP API.
Requires: `uv add contextunity.router[models-anthropic]`
Config: `ANTHROPIC_API_KEY` env var or `config.anthropic.api_key`
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from contextunity.core import get_contextunit_logger
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from contextunity.router.core import RouterConfig, set_env_default
from contextunity.router.modules.models.types import ModelError

from ..base import BaseLLM as BaseModel
from ..boundary_common import resolve_max_output_tokens, resolve_temperature
from ..registry import model_registry
from ..types import (
    FinalTextEvent,
    ImagePart,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ProviderInfo,
    TextDeltaEvent,
    TextPart,
)
from .types import AnthropicProviderConfig

logger = get_contextunit_logger(__name__)


def _build_anthropic_messages(
    request: ModelRequest,
) -> tuple[str | None, list[BaseMessage]]:
    """Convert ``ModelRequest`` parts to Anthropic messages (system + multimodal).

    Returns:
        Tuple of optional system prompt and list of ``BaseMessage``.

    Raises:
        ModelError: If no parts are supplied.
    """
    if not request.parts:
        raise ModelError("Request must contain at least one part")

    system = request.system.strip() if request.system else None
    has_images = any(isinstance(p, ImagePart) for p in request.parts)

    if has_images:
        # Build multimodal content for Claude
        content: list[str | dict[str, object]] = []
        for part in request.parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if part.data_b64:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.mime,
                                "data": part.data_b64,
                            },
                        }
                    )
                elif part.uri:
                    # Claude supports URL images via the "url" source type
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part.uri,
                            },
                        }
                    )
        return system, [HumanMessage(content=content)]
    else:
        # Text-only
        text_parts = [p.text for p in request.parts if isinstance(p, TextPart)]
        if not text_parts:
            raise ModelError("AnthropicLLM requires at least one TextPart")
        user = "".join(text_parts)
        return system, [HumanMessage(content=user)]


@model_registry.register_llm("anthropic", "*")
class AnthropicLLM(BaseModel):
    """Claude via langchain-anthropic (text-only in this abstraction)."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        **kwargs: object,
    ) -> None:
        """Configure the Claude client via ``langchain-anthropic``.

        Raises:
            ModuleNotFoundError: If the ``langchain-anthropic`` extra is missing.
        """
        resolved_name = (model_name or "").strip() or "claude-sonnet-4.5"
        super().__init__(provider="anthropic", model_name=resolved_name)
        self._cfg: RouterConfig = config
        api_key_raw = kwargs.pop("api_key", config.anthropic.api_key or None)
        api_key = api_key_raw if isinstance(api_key_raw, str) else None
        # Claude supports text and images natively
        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=False
        )

        # Prefer passing key explicitly; also set env default for libraries that rely on it.
        if isinstance(api_key, str) and api_key:
            set_env_default("ANTHROPIC_API_KEY", api_key)

        from .langchain_anthropic_boundary import load_chat_anthropic_factory

        chat_anthropic_cls = load_chat_anthropic_factory()

        resolved_temperature = self._cfg.llm.temperature if temperature is None else temperature
        resolved_max_tokens = (
            self._cfg.llm.max_output_tokens if max_output_tokens is None else max_output_tokens
        )

        if kwargs:
            logger.debug("Ignoring unsupported AnthropicLLM kwargs: %s", sorted(kwargs.keys()))

        client_kwargs: dict[str, object] = {
            "model": self._model_name,
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
            "timeout": self._cfg.llm.timeout_sec,
        }
        if isinstance(api_key, str) and api_key:
            client_kwargs["api_key"] = api_key

        try:
            self._client: BaseChatModel = chat_anthropic_cls(**client_kwargs)
        except TypeError:
            _ = client_kwargs.pop("api_key", None)
            self._client = chat_anthropic_cls(**client_kwargs)

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Return text + image capability flags."""
        return self._capabilities

    @override
    async def _generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Invoke Claude and return the full response."""
        system, user_messages = _build_anthropic_messages(request)

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.extend(user_messages)

        pc = AnthropicProviderConfig.model_validate(request.provider_config)
        bind_kwargs: dict[str, object] = {}
        temperature = resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        _max = resolve_max_output_tokens(
            request_max_output_tokens=request.max_output_tokens,
            provider_max_tokens=pc.get_max_tokens(),
        )
        if _max is not None:
            bind_kwargs["max_tokens"] = _max
        if request.timeout_sec is not None:
            bind_kwargs["timeout"] = request.timeout_sec
        client = self._client.bind(**bind_kwargs) if bind_kwargs else self._client
        msg = await client.ainvoke(messages)
        content_obj: object = getattr(msg, "content", "")
        text = content_obj if isinstance(content_obj, str) else str(content_obj)

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="anthropic",
                model_name=self._model_name,
                model_key=f"anthropic/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Return a streaming iterator of Claude completion deltas."""
        system, user_messages = _build_anthropic_messages(request)
        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.extend(user_messages)

        pc = AnthropicProviderConfig.model_validate(request.provider_config)
        bind_kwargs: dict[str, object] = {}
        temperature = resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        _max = resolve_max_output_tokens(
            request_max_output_tokens=request.max_output_tokens,
            provider_max_tokens=pc.get_max_tokens(),
        )
        if _max is not None:
            bind_kwargs["max_tokens"] = _max
        if request.timeout_sec is not None:
            bind_kwargs["timeout"] = request.timeout_sec
        client = self._client.bind(**bind_kwargs) if bind_kwargs else self._client

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the langchain-anthropic stream."""
            full = ""
            async for chunk in client.astream(messages):
                c = getattr(chunk, "content", "")
                if isinstance(c, str) and c:
                    full += c
                    yield TextDeltaEvent(delta=c)
            if full:
                yield FinalTextEvent(text=full)

        return _event_stream()

    @override
    def get_token_count(self, text: str) -> int:
        """Estimate token count via a conservative heuristic."""
        if not text:
            return 0
        # Claude tokenization varies; keep a conservative heuristic for now.
        return max(1, len(text) // 4)


__all__ = ["AnthropicLLM"]
