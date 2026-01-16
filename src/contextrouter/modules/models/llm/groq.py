"""Groq LLM provider (OpenAI-compatible API).

Groq is known for its ultra-fast inference using custom LPU chips.
It is OpenAI-compatible at the HTTP level.
"""

from __future__ import annotations

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
)
from ._openai_compat import (
    build_openai_messages,
    generate_asr_openai_compat,
)


@model_registry.register_llm("groq", "*")
class GroqLLM(BaseModel):
    """Groq provider (OpenAI-compatible).

    Features ultra-fast Whisper ASR and vision support for compatible models.
    """

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("GroqLLM requires `contextrouter[models-openai]`.") from e

        self._cfg = config
        self._model_name = (model_name or "").strip() or "llama-3.3-70b-versatile"
        self._base_url = (config.groq.base_url or "").strip() or "https://api.groq.com/openai/v1"

        self._model = ChatOpenAI(
            model=self._model_name,
            api_key=(config.groq.api_key or None),
            base_url=self._base_url,
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
                base_url=self._base_url,
                api_key=self._cfg.groq.api_key,
                provider="groq",
                whisper_model="whisper-large-v3",
            )

        messages = build_openai_messages(request)
        model = self._model.bind(
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            timeout=request.timeout_sec,
        )
        msg = await model.ainvoke(messages)
        text = getattr(msg, "content", "")

        return ModelResponse(
            text=str(text or ""),
            raw_provider=ProviderInfo(
                provider="groq",
                model_name=self._model_name,
                model_key=f"groq/{self._model_name}",
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


__all__ = ["GroqLLM"]
