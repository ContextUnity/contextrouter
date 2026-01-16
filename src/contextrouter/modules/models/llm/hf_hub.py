"""HuggingFace Hub LLM provider (Remote Inference API).

This uses the `huggingface_hub.InferenceClient` to call models hosted on Hugging Face Hub
or Inference Endpoints.
"""

from __future__ import annotations

import base64
import logging
from typing import AsyncIterator

from contextrouter.core import Config
from contextrouter.core.tokens import BiscuitToken

from ..base import BaseModel
from ..registry import model_registry
from ..types import (
    AudioPart,
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


@model_registry.register_llm("hf-hub", "*")
class HuggingFaceHubLLM(BaseModel):
    """HuggingFace Hub provider (Remote Inference API)."""

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        task: str | None = None,
        **kwargs: object,
    ) -> None:
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("HuggingFaceHubLLM requires `contextrouter[models-hf-hub]`.") from e

        self._cfg = config
        self._model_name = (model_name or "").strip() or "mistralai/Mistral-7B-Instruct-v0.2"
        self._task = (task or "text-generation").strip() or "text-generation"

        api_key = config.hf_hub.api_key or ""
        base_url = config.hf_hub.base_url or "https://api-inference.huggingface.co/v1"

        self._client = AsyncInferenceClient(
            model=self._model_name,
            token=(api_key or None),
            base_url=(base_url if base_url else None),
            **kwargs,
        )

        # Capabilities depend on task - we only claim what we actually implement
        image_tasks = {"image-to-text", "visual-question-answering", "image-classification"}
        self._capabilities = ModelCapabilities(
            supports_text=True,
            supports_image=(self._task in image_tasks),
            supports_audio=(self._task == "automatic-speech-recognition"),
            supports_video=False,  # Video tasks not yet implemented
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> ModelResponse:
        _ = token
        if self._task == "automatic-speech-recognition":
            return await self._generate_asr(request)
        if self._task == "text-classification":
            return await self._generate_classification(request)
        if self._task == "image-to-text" or any(isinstance(p, ImagePart) for p in request.parts):
            return await self._generate_image_to_text(request)

        # For text generation, we prefer chat_completion if available
        try:
            prompt = request.to_text_prompt()
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": prompt})

            resp = await self._client.chat_completion(
                messages=messages,
                max_tokens=request.max_output_tokens or 512,
                temperature=request.temperature or 0.7,
            )
            text = resp.choices[0].message.content
            return ModelResponse(
                text=str(text or ""),
                raw_provider=ProviderInfo(
                    provider="hf-hub",
                    model_name=self._model_name,
                    model_key=f"hf-hub/{self._model_name}",
                ),
            )
        except Exception as e:
            logger.debug("chat_completion failed, falling back to text_generation: %s", e)
            # Fallback to plain text generation if chat template is not supported
            prompt = request.to_text_prompt()
            if request.system:
                prompt = f"{request.system}\n\n{prompt}"

            resp_text = await self._client.text_generation(
                prompt,
                max_new_tokens=request.max_output_tokens or 512,
                temperature=request.temperature or 0.7,
            )
            return ModelResponse(
                text=str(resp_text or ""),
                raw_provider=ProviderInfo(
                    provider="hf-hub",
                    model_name=self._model_name,
                    model_key=f"hf-hub/{self._model_name}",
                ),
            )

    async def _generate_asr(self, request: ModelRequest) -> ModelResponse:
        audio_parts = [p for p in request.parts if isinstance(p, AudioPart)]
        if not audio_parts:
            raise ValueError("ASR requires at least one AudioPart")

        part = audio_parts[0]
        data: bytes
        if part.uri:
            # InferenceClient can take a URL or a file path
            # But let's be safe and pass bytes if it's base64 or read it if it's local
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(part.uri)
                data = resp.content
        elif part.data_b64:
            data = base64.b64decode(part.data_b64)
        else:
            raise ValueError("AudioPart must have uri or data_b64")

        # InferenceClient.automatic_speech_recognition is synchronous in some versions,
        # but AsyncInferenceClient should have it async.
        # Check current implementation of AsyncInferenceClient
        out = await self._client.automatic_speech_recognition(data)
        return ModelResponse(
            text=str(getattr(out, "text", out)),
            raw_provider=ProviderInfo(
                provider="hf-hub",
                model_name=self._model_name,
                model_key=f"hf-hub/{self._model_name}",
            ),
        )

    async def _generate_classification(self, request: ModelRequest) -> ModelResponse:
        prompt = request.to_text_prompt(include_system=True)
        out = await self._client.text_classification(prompt)
        # out is usually list of dicts with label/score
        return ModelResponse(
            text=str(out),
            raw_provider=ProviderInfo(
                provider="hf-hub",
                model_name=self._model_name,
                model_key=f"hf-hub/{self._model_name}",
            ),
        )

    async def _generate_image_to_text(self, request: ModelRequest) -> ModelResponse:
        image_parts = [p for p in request.parts if isinstance(p, ImagePart)]
        if not image_parts:
            raise ValueError("Image-to-text requires at least one ImagePart")

        part = image_parts[0]
        data: bytes
        if part.uri:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(part.uri)
                data = resp.content
        elif part.data_b64:
            data = base64.b64decode(part.data_b64)
        else:
            raise ValueError("ImagePart must have uri or data_b64")

        # Use image_to_text or visual_question_answering depending on parts
        text_parts = [p for p in request.parts if isinstance(p, TextPart)]
        if text_parts:
            # VQA path
            prompt = text_parts[0].text
            out = await self._client.visual_question_answering(data, prompt)
            text = str(getattr(out, "answer", out))
        else:
            # Captioning path
            out = await self._client.image_to_text(data)
            text = str(getattr(out, "generated_text", out))

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf-hub",
                model_name=self._model_name,
                model_key=f"hf-hub/{self._model_name}",
            ),
        )

    async def stream(
        self, request: ModelRequest, *, token: BiscuitToken | None = None
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = token
        # Try chat_completion streaming
        try:
            prompt = request.to_text_prompt()
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": prompt})

            full = ""
            async for chunk in await self._client.chat_completion(
                messages=messages,
                max_tokens=request.max_output_tokens or 512,
                temperature=request.temperature or 0.7,
                stream=True,
            ):
                delta = chunk.choices[0].delta.content
                if delta:
                    full += delta
                    yield TextDeltaEvent(delta=delta)
            yield FinalTextEvent(text=full)
        except Exception as e:
            logger.debug("chat_completion streaming failed for HF Hub: %s", e)
            # Fallback to non-streaming generate if everything fails
            # or implement text_generation(stream=True)
            res = await self.generate(request)
            yield TextDeltaEvent(delta=res.text)
            yield FinalTextEvent(text=res.text)


__all__ = ["HuggingFaceHubLLM"]
