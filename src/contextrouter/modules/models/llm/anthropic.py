"""Anthropic LLM provider (Claude) via `langchain-anthropic`.

This provider is the same class of integration as OpenAI/Vertex: remote HTTP API.

Requires: `uv add contextrouter[models-anthropic]`
Config: `ANTHROPIC_API_KEY` env var or `config.anthropic.api_key`
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from contextrouter.core import Config, set_env_default
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseModel
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

logger = logging.getLogger(__name__)


def _build_anthropic_messages(
    request: ModelRequest,
) -> tuple[str | None, list[object]]:
    """Convert ModelRequest parts to Anthropic messages (system + user + multimodal)."""
    if not request.parts:
        raise ValueError("Request must contain at least one part")

    system = request.system.strip() if request.system else None
    has_images = any(isinstance(p, ImagePart) for p in request.parts)

    from langchain_core.messages import HumanMessage

    if has_images:
        # Build multimodal content for Claude
        content: list[dict[str, object]] = []
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
            raise ValueError("AnthropicLLM requires at least one TextPart")
        user = "".join(text_parts)
        return system, [HumanMessage(content=user)]


@model_registry.register_llm("anthropic", "*")
class AnthropicLLM(BaseModel):
    """Claude via langchain-anthropic (text-only in this abstraction)."""

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        **_: object,
    ) -> None:
        self._cfg = config
        self._model_name = (model_name or "").strip() or "claude-sonnet-4.5"
        # Claude supports text and images natively
        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=False
        )

        # Prefer passing key explicitly; also set env default for libraries that rely on it.
        if config.anthropic.api_key:
            set_env_default("ANTHROPIC_API_KEY", config.anthropic.api_key)

        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "Anthropic provider requires `contextrouter[models-anthropic]`."
            ) from e

        kwargs: dict[str, object] = {
            "model": self._model_name,
            "temperature": self._cfg.llm.temperature if temperature is None else temperature,
            "max_tokens": self._cfg.llm.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens,
            "timeout": self._cfg.llm.timeout_sec,
        }

        # Some versions accept `api_key`, some rely on env; try both.
        if self._cfg.anthropic.api_key:
            kwargs["api_key"] = self._cfg.anthropic.api_key

        try:
            self._client = ChatAnthropic(**kwargs)
        except TypeError:
            kwargs.pop("api_key", None)
            self._client = ChatAnthropic(**kwargs)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> ModelResponse:
        _ = token
        system, user_messages = _build_anthropic_messages(request)

        from langchain_core.messages import SystemMessage

        messages: list[object] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.extend(user_messages)

        client = self._client.bind(
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            timeout=request.timeout_sec,
        )
        msg = await client.ainvoke(messages)
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="anthropic",
                model_name=self._model_name,
                model_key=f"anthropic/{self._model_name}",
            ),
        )

    async def stream(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = token
        system, user_messages = _build_anthropic_messages(request)

        from langchain_core.messages import SystemMessage

        messages: list[object] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.extend(user_messages)

        client = self._client.bind(
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            timeout=request.timeout_sec,
        )

        full = ""
        async for chunk in client.astream(messages):
            c = getattr(chunk, "content", "")
            if isinstance(c, str) and c:
                full += c
                yield TextDeltaEvent(delta=c)
        if full:
            yield FinalTextEvent(text=full)

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0
        # Claude tokenization varies; keep a conservative heuristic for now.
        return max(1, len(text) // 4)


__all__ = ["AnthropicLLM"]
