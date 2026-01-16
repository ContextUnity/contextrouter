"""OpenAI LLM provider (OpenAI API).

This implementation is a thin adapter from the multimodal `BaseModel` contract to
`langchain-openai`'s `ChatOpenAI` so LangGraph can still stream tokens when used
inside a LangChain/LangGraph runnable pipeline.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import BiscuitToken

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
    """

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        **kwargs: object,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("OpenAILLM requires `contextrouter[models-openai]`.") from e

        self._cfg = config
        self._model_name = (model_name or "gpt-5.1").strip() or "gpt-5.1"
        self._model = ChatOpenAI(
            model=self._model_name,
            api_key=(config.openai.api_key or None),
            organization=config.openai.organization,
            max_retries=config.llm.max_retries,
            **kwargs,
        )

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=True
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> ModelResponse:
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
        model = self._model.bind(
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            timeout=request.timeout_sec,
        )
        msg = await model.ainvoke(messages)
        text = getattr(msg, "content", "")

        usage = self._extract_usage(msg)

        return ModelResponse(
            text=str(text or ""),
            usage=usage,
            raw_provider=ProviderInfo(
                provider="openai",
                model_name=self._model_name,
                model_key=f"openai/{self._model_name}",
            ),
        )

    async def stream(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = token
        messages = build_openai_messages(request)
        model = self._model.bind(
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            timeout=request.timeout_sec,
        )

        full = ""
        async for chunk in model.astream(messages):
            delta = getattr(chunk, "content", None)
            if isinstance(delta, str) and delta:
                full += delta
                yield TextDeltaEvent(delta=delta)
        yield FinalTextEvent(text=full)

    def _extract_usage(self, msg: object) -> UsageStats | None:
        """Extract usage stats from LangChain message metadata."""
        try:
            meta = getattr(msg, "response_metadata", None) or {}
            u = meta.get("token_usage") if isinstance(meta, dict) else None
            if isinstance(u, dict):
                return UsageStats(
                    input_tokens=int(u.get("prompt_tokens") or 0),
                    output_tokens=int(u.get("completion_tokens") or 0),
                    total_tokens=int(u.get("total_tokens") or 0),
                )
        except Exception:
            pass
        return None


__all__ = ["OpenAILLM"]
