"""HuggingFace Hub LLM provider (Remote Inference API).
This uses the `huggingface_hub.InferenceClient` to call models hosted on Hugging Face Hub
or Inference Endpoints.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import override

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError

from contextunity.router.core import RouterConfig
from contextunity.router.modules.models.types import ModelError

from ..base import BaseLLM as BaseModel
from ..boundary_common import first_list_item, resolve_max_output_tokens, resolve_temperature
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
from .hf_hub_boundary import HFHubAsyncClient, load_async_inference_client
from .types import HFHubProviderConfig

logger = get_contextunit_logger(__name__)


def _optional_attr(obj: object, name: str) -> object | None:
    value: object = getattr(obj, name, None)
    return value


def _object_text(obj: object, attr: str, default: str = "") -> str:
    val = _optional_attr(obj, attr)
    if val is None:
        return default
    return str(val)


def _first_choice(obj: object) -> object | None:
    choices_obj: object = _optional_attr(obj, "choices")
    return first_list_item(choices_obj)


def _chat_completion_text(resp: object) -> str:
    first = _first_choice(resp)
    if first is None:
        return ""
    message = _optional_attr(first, "message")
    if message is None:
        return ""
    content = _optional_attr(message, "content")
    return str(content) if content is not None else ""


def _chat_delta_content(chunk: object) -> str:
    first = _first_choice(chunk)
    if first is None:
        return ""
    delta = _optional_attr(first, "delta")
    if delta is None:
        return ""
    content = _optional_attr(delta, "content")
    return str(content) if content is not None else ""


@model_registry.register_llm("hf-hub", "*")
class HuggingFaceHubLLM(BaseModel):
    """HuggingFace Hub provider (Remote Inference API)."""

    _client: HFHubAsyncClient

    def __init__(
        self,
        config: RouterConfig,
        *,
        model_name: str | None = None,
        task: str | None = None,
        **kwargs: object,
    ) -> None:
        """Create an ``AsyncInferenceClient`` and resolve the HF API key and optional endpoint URL."""
        resolved_name = (model_name or "").strip() or "mistralai/Mistral-7B-Instruct-v0.2"
        super().__init__(provider="hf-hub", model_name=resolved_name)
        self._cfg: RouterConfig = config
        self._task: str = (task or "text-generation").strip() or "text-generation"

        api_key = config.hf_hub.api_key or ""
        base_url = config.hf_hub.base_url

        hf_kwargs: dict[str, object] = {}
        if base_url:
            hf_kwargs["base_url"] = base_url
        else:
            hf_kwargs["model"] = self._model_name

        try:
            self._client = load_async_inference_client(
                token=(api_key or None),
                **hf_kwargs,
                **kwargs,
            )
        except ImportError as e:  # pragma: no cover
            raise ConfigurationError(
                "HuggingFaceHubLLM requires `contextunity.router[models-hf-hub]`."
            ) from e

        image_tasks = {"image-to-text", "visual-question-answering", "image-classification"}
        self._capabilities: ModelCapabilities = ModelCapabilities(
            supports_text=True,
            supports_image=(self._task in image_tasks),
            supports_audio=(self._task == "automatic-speech-recognition"),
            supports_video=False,
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the HuggingFace Hub backend."""
        return self._capabilities

    @override
    async def _generate(self, request: ModelRequest) -> ModelResponse:
        """Call the InferenceClient HTTP API and return a complete response."""
        if self._task == "automatic-speech-recognition":
            return await self._generate_asr(request)
        if self._task == "text-classification":
            return await self._generate_classification(request)
        if self._task == "image-to-text" or any(isinstance(p, ImagePart) for p in request.parts):
            return await self._generate_image_to_text(request)

        pc = HFHubProviderConfig.model_validate(request.provider_config)
        temperature = resolve_temperature(
            request_temperature=request.temperature,
            provider_temperature=pc.temperature,
        )
        max_tokens = (
            resolve_max_output_tokens(
                request_max_output_tokens=request.max_output_tokens,
                provider_max_tokens=pc.get_max_tokens(),
            )
            or 512
        )
        try:
            prompt = request.to_text_prompt()
            messages: list[dict[str, str]] = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": prompt})

            resp = await self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else 0.7,
            )
            text = _chat_completion_text(resp)
            return ModelResponse(
                text=text,
                raw_provider=ProviderInfo(
                    provider="hf-hub",
                    model_name=self._model_name,
                    model_key=f"hf-hub/{self._model_name}",
                ),
            )
        except Exception as e:
            logger.debug("chat_completion failed, falling back to text_generation: %s", e)
            prompt = request.to_text_prompt()
            if request.system:
                prompt = f"{request.system}\n\n{prompt}"

            resp_text = await self._client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature is not None else 0.7,
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
        """Transcribe the first ``AudioPart`` via the HF Hub speech-recognition endpoint."""
        audio_parts = [p for p in request.parts if isinstance(p, AudioPart)]
        if not audio_parts:
            raise ModelError("ASR requires at least one AudioPart")

        part = audio_parts[0]
        data: bytes
        if part.uri:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(part.uri)
                data = resp.content
        elif part.data_b64:
            data = base64.b64decode(part.data_b64)
        else:
            raise ModelError("AudioPart must have uri or data_b64")

        out = await self._client.automatic_speech_recognition(data)
        return ModelResponse(
            text=_object_text(out, "text", str(out)),
            raw_provider=ProviderInfo(
                provider="hf-hub",
                model_name=self._model_name,
                model_key=f"hf-hub/{self._model_name}",
            ),
        )

    async def _generate_classification(self, request: ModelRequest) -> ModelResponse:
        """Run the text-classification endpoint and return raw label/score output."""
        prompt = request.to_text_prompt(include_system=True)
        out = await self._client.text_classification(prompt)
        return ModelResponse(
            text=str(out),
            raw_provider=ProviderInfo(
                provider="hf-hub",
                model_name=self._model_name,
                model_key=f"hf-hub/{self._model_name}",
            ),
        )

    async def _generate_image_to_text(self, request: ModelRequest) -> ModelResponse:
        """Caption or answer a VQA question on the first ``ImagePart`` via the HF Hub endpoint."""
        image_parts = [p for p in request.parts if isinstance(p, ImagePart)]
        if not image_parts:
            raise ModelError("Image-to-text requires at least one ImagePart")

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
            raise ModelError("ImagePart must have uri or data_b64")

        text_parts = [p for p in request.parts if isinstance(p, TextPart)]
        if text_parts:
            prompt = text_parts[0].text
            out = await self._client.visual_question_answering(data, prompt)
            text = _object_text(out, "answer", str(out))
        else:
            out = await self._client.image_to_text(data)
            text = _object_text(out, "generated_text", str(out))

        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf-hub",
                model_name=self._model_name,
                model_key=f"hf-hub/{self._model_name}",
            ),
        )

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the InferenceClient HTTP API."""

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the InferenceClient HTTP API stream."""
            try:
                pc = HFHubProviderConfig.model_validate(request.provider_config)
                temperature = resolve_temperature(
                    request_temperature=request.temperature,
                    provider_temperature=pc.temperature,
                )
                max_tokens = (
                    resolve_max_output_tokens(
                        request_max_output_tokens=request.max_output_tokens,
                        provider_max_tokens=pc.get_max_tokens(),
                    )
                    or 512
                )
                prompt = request.to_text_prompt()
                messages: list[dict[str, str]] = []
                if request.system:
                    messages.append({"role": "system", "content": request.system})
                messages.append({"role": "user", "content": prompt})

                full = ""
                stream = await self._client.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature if temperature is not None else 0.7,
                    stream=True,
                )
                async for chunk in stream:
                    delta = _chat_delta_content(chunk)
                    if delta:
                        full += delta
                        yield TextDeltaEvent(delta=delta)
                yield FinalTextEvent(text=full)
            except Exception as e:
                logger.debug("chat_completion streaming failed for HF Hub: %s", e)
                res = await self._generate(request)
                yield TextDeltaEvent(delta=res.text)
                yield FinalTextEvent(text=res.text)

        return _event_stream()


__all__ = ["HuggingFaceHubLLM"]
