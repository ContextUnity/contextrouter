"""RunPod LLM provider (OpenAI-compatible API).

RunPod Serverless provides OpenAI-compatible endpoints for vLLM and TGI.
Custom workers can support additional modalities.
"""

from __future__ import annotations

from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseModel
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
from ._openai_compat import build_openai_messages


@model_registry.register_llm("runpod", "*")
class RunPodLLM(BaseModel):
    """RunPod provider (OpenAI-compatible).

    Supports text and images for vision models deployed on RunPod.
    """

    def __init__(self, config: Config, *, model_name: str | None = None, **kwargs: object) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("RunPodLLM requires `contextrouter[models-openai]`.") from e

        self._cfg = config
        self._model_name = (model_name or "").strip() or "runpod-model"
        self._base_url = (config.runpod.base_url or "").strip()

        self._model = ChatOpenAI(
            model=self._model_name,
            api_key=(config.runpod.api_key or None),
            base_url=self._base_url,
            max_retries=config.llm.max_retries,
            **kwargs,
        )

        self._capabilities = ModelCapabilities(
            supports_text=True, supports_image=True, supports_audio=False
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> ModelResponse:
        _ = token
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
                provider="runpod",
                model_name=self._model_name,
                model_key=f"runpod/{self._model_name}",
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


__all__ = ["RunPodLLM"]
