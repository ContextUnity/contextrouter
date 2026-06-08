"""HuggingFace Transformers LLM provider.
⚠️  WARNING: This provider requires heavy dependencies (`torch`, `transformers`)
and is designed for local inference. It is NOT suitable for:
- High-throughput scenarios
- Very large models on limited hardware
Use cases:
- CPU-based development/testing
- Small specialized models
- Offline environments
Requires: `uv add contextunity.router[hf-transformers]`
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from collections.abc import AsyncIterator
from importlib import import_module
from typing import Protocol, override

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.types import is_object_list

from contextunity.router.core import RouterConfig
from contextunity.router.modules.models.types import ModelError

from ..base import BaseLLM as BaseModel
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
from .transformers_boundary import (
    classification_label_score,
    load_transformers_pipeline,
    object_dict_get_str,
    tokenizer_encode_length,
    torch_cuda_device_index,
)

logger = get_contextunit_logger(__name__)


class _CallablePipeline(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...

    @property
    def tokenizer(self) -> object | None: ...


def _wrap_pipeline(inner: object) -> _CallablePipeline:
    """Adapt a transformers pipeline instance to a typed callable surface."""

    class _PipelineWrapper:
        def __call__(self, *args: object, **kwargs: object) -> object:
            if callable(inner):
                return inner(*args, **kwargs)
            caller_obj: object = getattr(inner, "__call__", None)
            if callable(caller_obj):
                return caller_obj(*args, **kwargs)
            raise ConfigurationError("HuggingFace pipeline is not callable")

        @property
        def tokenizer(self) -> object | None:
            return getattr(inner, "tokenizer", None)

    return _PipelineWrapper()


@model_registry.register_llm("hf", "*")
class HuggingFaceLLM(BaseModel):
    """HuggingFace Transformers provider for local inference.

    ⚠️  WARNING: Requires 'transformers' and 'torch' packages.
    Not recommended for heavy models; start with small models and scale up carefully.
    """

    def __init__(
        self,
        config: RouterConfig,
        *,
        model_name: str | None = None,
        task: str | None = None,
        **_kwargs: object,
    ) -> None:
        """Resolve the local base URL and HF-tokenizer model name from config."""
        resolved_name = model_name or "distilgpt2"
        super().__init__(provider="hf", model_name=resolved_name)
        self._cfg: RouterConfig = config
        self._task: str = (task or "text-generation").strip() or "text-generation"

        # Lazy initialization - load model only when needed
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._pipeline: _CallablePipeline | None = None

        self._capabilities: ModelCapabilities = self._capabilities_for_task(self._task)

        logger.warning(
            (
                "HuggingFaceLLM initialized with model '%s'. "
                "This provider is for local transformers inference and may be slow for large models. "
                "For heavy models / high throughput prefer vLLM."
            ),
            self._model_name,
        )

    @property
    @override
    def capabilities(self) -> ModelCapabilities:
        """Declare modality support for the HuggingFace backend."""
        return self._capabilities

    def _capabilities_for_task(self, task: str) -> ModelCapabilities:
        """Derive capabilities from hf pipeline task."""
        t = (task or "").strip().lower()
        supports_audio = t in {"automatic-speech-recognition", "audio-classification"}
        supports_image = t in {"image-classification", "object-detection", "image-to-text"}
        # Video tasks exist in transformers but we haven't implemented them yet
        supports_video = False
        return ModelCapabilities(
            supports_text=True,
            supports_image=supports_image,
            supports_audio=supports_audio,
            supports_video=supports_video,
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy load the transformers model and tokenizer."""
        if self._pipeline is not None:
            return

        pipeline_fn = load_transformers_pipeline()

        try:
            logger.info("Loading HuggingFace model: %s", self._model_name)
            # Use a transformers pipeline for simple inference.
            # If CUDA is available, allow GPU usage; otherwise fallback to CPU.
            device = torch_cuda_device_index()

            self._pipeline = _wrap_pipeline(
                pipeline_fn(
                    self._task,
                    model=self._model_name,
                    device=device,
                    torch_dtype="auto",
                    trust_remote_code=False,  # Security: don't run arbitrary code
                )
            )
            logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            logger.error("Failed to load HuggingFace model '%s': %s", self._model_name, e)
            raise ConfigurationError(f"Failed to load model {self._model_name}") from e

    @override
    async def _generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Call the langchain HuggingFacePipeline and return a complete response."""
        if self._task == "automatic-speech-recognition":
            return await self._generate_asr(request)
        if self._task == "text-classification":
            return await self._generate_text_classification(request)
        if self._task == "image-classification" or self._task == "object-detection":
            return await self._generate_vision_task(request)

        # Extract text from request
        if not request.parts:
            raise ModelError("Request must contain at least one part")

        text_parts = [part.text for part in request.parts if isinstance(part, TextPart)]
        if not text_parts:
            raise ModelError("HuggingFaceLLM requires at least one text part")

        prompt = "".join(text_parts)
        if request.system:
            prompt = f"{request.system}\n\n{prompt}"

        # Run in thread pool since transformers is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._generate_sync, prompt)

        # Clean up common artifacts from text generation
        result = self._clean_generated_text(prompt, result)

        return ModelResponse(
            text=result,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    async def _generate_text_classification(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Run the text-classification pipeline and return the top label/score as JSON text."""
        prompt = request.to_text_prompt(include_system=True)
        if not prompt:
            raise ModelError("text-classification requires at least one TextPart")

        self._ensure_model_loaded()
        loop = asyncio.get_event_loop()

        def _run() -> str:
            """Run the classification pipeline and format the top label/score pair."""
            pipeline = self._pipeline
            if pipeline is None:
                raise ConfigurationError("HuggingFace pipeline is not initialized")
            out = pipeline(prompt)
            if is_object_list(out) and out:
                label, score = classification_label_score(out[0])
            else:
                label, score = classification_label_score(out)
            if label is not None:
                return f'{{"label": "{label}", "score": {score}}}'
            return str(out)

        text = await loop.run_in_executor(None, _run)
        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    async def _generate_asr(self, request: ModelRequest) -> ModelResponse:
        """Transcribe the first ``AudioPart`` via the speech-recognition pipeline."""
        audio_parts = [p for p in request.parts if isinstance(p, AudioPart)]
        if not audio_parts:
            raise ModelError("automatic-speech-recognition requires at least one AudioPart")

        part = audio_parts[0]
        if not (part.uri or part.data_b64):
            raise ModelError("AudioPart requires either uri or data_b64")

        self._ensure_model_loaded()
        loop = asyncio.get_event_loop()

        def _run() -> str:
            """Decode audio from URI or base64 and run the ASR pipeline."""
            pipeline = self._pipeline
            if pipeline is None:
                raise ConfigurationError("HuggingFace pipeline is not initialized")
            # Easiest supported path: uri points to a local file path.
            if part.uri:
                out = pipeline(part.uri)
                text_val = object_dict_get_str(out, "text")
                if text_val is not None:
                    return text_val
                return str(out)

            raw = base64.b64decode(part.data_b64 or "")
            # Write to a temp file; transformers ASR pipelines commonly expect a file path.
            suffix = ".wav" if (part.mime or "").endswith("wav") else ".audio"
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as f:
                _ = f.write(raw)
                _ = f.flush()
                out = pipeline(f.name)
            text_val = object_dict_get_str(out, "text")
            if text_val is not None:
                return text_val
            return str(out)

        text = await loop.run_in_executor(None, _run)
        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    async def _generate_vision_task(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        """Classify or detect objects in the first ``ImagePart`` via the vision pipeline."""
        image_parts = [p for p in request.parts if isinstance(p, ImagePart)]
        if not image_parts:
            raise ModelError(f"{self._task} requires at least one ImagePart")

        part = image_parts[0]
        self._ensure_model_loaded()
        loop = asyncio.get_event_loop()

        def _run() -> str:
            """Load the image from URI or base64 and run the vision pipeline."""
            pipeline = self._pipeline
            if pipeline is None:
                raise ConfigurationError("HuggingFace pipeline is not initialized")
            # Load image from URI or b64
            import io

            image_module = import_module("PIL.Image")
            open_fn_obj: object = getattr(image_module, "open", None)
            if not callable(open_fn_obj):
                raise ConfigurationError("PIL.Image.open is unavailable")

            if part.uri:
                img: object = open_fn_obj(part.uri)
            elif part.data_b64:
                img = open_fn_obj(io.BytesIO(base64.b64decode(part.data_b64)))
            else:
                raise ModelError("ImagePart must have uri or data_b64")

            out = pipeline(img)
            return str(out)

        text = await loop.run_in_executor(None, _run)
        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    def _generate_sync(self, prompt: str) -> str:
        """Execute the transformers pipeline synchronously and extract the generated text."""
        self._ensure_model_loaded()
        pipeline = self._pipeline
        if pipeline is None:
            raise ConfigurationError("HuggingFace pipeline is not initialized")

        try:
            # Generate with reasonable defaults for small models
            tokenizer = getattr(pipeline, "tokenizer", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            outputs_obj = pipeline(
                prompt,
                max_new_tokens=128,  # Conservative for CPU/small models
                temperature=0.7,
                do_sample=True,
                pad_token_id=eos_token_id,
                num_return_sequences=1,
            )

            if is_object_list(outputs_obj) and outputs_obj:
                generated_text = object_dict_get_str(outputs_obj[0], "generated_text")
                if generated_text is not None:
                    return generated_text
                return str(outputs_obj[0])
            return ""

        except Exception as e:
            logger.error("Error during HuggingFace generation: %s", e)
            raise ModelError("HuggingFace generation failed") from e

    @override
    def _stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream token deltas from the langchain HuggingFacePipeline."""

        async def _event_stream() -> AsyncIterator[ModelStreamEvent]:
            """Yield ``TextDelta`` / ``UsageEvent`` from the langchain HuggingFacePipeline stream."""
            result = await self._generate(request)
            yield TextDeltaEvent(delta=result.text)
            yield FinalTextEvent(text=result.text)

        return _event_stream()

    @override
    def get_token_count(self, text: str) -> int:
        """Tokenise *text* via the pipeline’s tokenizer; fall back to whitespace approximation."""
        if not text:
            return 0

        try:
            self._ensure_model_loaded()
            pipeline = self._pipeline
            if pipeline is None:
                return max(1, len(text.split()))
            # Use tokenizer if available
            tokenizer_obj: object | None = pipeline.tokenizer
            if tokenizer_obj is not None:
                token_count = tokenizer_encode_length(tokenizer_obj, text)
                if token_count is not None:
                    return token_count
        except Exception:
            logger.debug("Could not get accurate token count, using approximation")

        # Fallback: rough approximation
        return max(1, len(text.split()))

    def _clean_generated_text(self, prompt: str, generated: str) -> str:
        """Clean up common artifacts from generated text."""
        # Remove the original prompt if it was included
        if generated.startswith(prompt):
            generated = generated[len(prompt) :].lstrip()

        # Remove common stop sequences or artifacts
        generated = generated.strip()

        # Limit reasonable length
        if len(generated) > 1000:
            generated = generated[:1000] + "..."

        return generated or "I apologize, but I couldn't generate a response."


__all__ = ["HuggingFaceLLM"]
