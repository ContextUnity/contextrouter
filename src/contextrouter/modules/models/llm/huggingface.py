"""HuggingFace Transformers LLM provider.

⚠️  WARNING: This provider requires heavy dependencies (`torch`, `transformers`)
and is designed for local inference. It is NOT suitable for:
- High-throughput scenarios
- Very large models on limited hardware

Use cases:
- CPU-based development/testing
- Small specialized models
- Offline environments

Requires: `uv add contextrouter[hf-transformers]`
"""

from __future__ import annotations

import asyncio
import base64
import logging
import tempfile
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


@model_registry.register_llm("hf", "*")
class HuggingFaceLLM(BaseModel):
    """HuggingFace Transformers provider for local inference.

    ⚠️  WARNING: Requires 'transformers' and 'torch' packages.
    Not recommended for heavy models; start with small models and scale up carefully.
    """

    def __init__(
        self,
        config: Config,
        *,
        model_name: str | None = None,
        task: str | None = None,
        **_kwargs: object,
    ) -> None:
        self._cfg = config
        self._model_name = model_name or "distilgpt2"  # Default small model
        self._task = (task or "text-generation").strip() or "text-generation"

        # Lazy initialization - load model only when needed
        self._model = None
        self._tokenizer = None
        self._pipeline = None

        self._capabilities = self._capabilities_for_task(self._task)

        logger.warning(
            "HuggingFaceLLM initialized with model '%s'. "
            "This provider is for local transformers inference and may be slow for large models. "
            "For heavy models / high throughput prefer vLLM.",
            self._model_name,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _capabilities_for_task(self, task: str) -> ModelCapabilities:
        """Derive capabilities from HF pipeline task.

        Transformers can support many modalities, but a pipeline instance is task-bound.
        We only claim capabilities we actually implement.
        """
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

        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers not installed. "
                "HuggingFaceLLM requires `contextrouter[hf-transformers]`."
            ) from e

        try:
            logger.info("Loading HuggingFace model: %s", self._model_name)
            # Use a transformers pipeline for simple inference.
            # If CUDA is available, allow GPU usage; otherwise fallback to CPU.
            try:
                import torch

                device = 0 if torch.cuda.is_available() else -1
            except Exception:
                device = -1

            self._pipeline = pipeline(
                self._task,
                model=self._model_name,
                device=device,
                torch_dtype="auto",
                trust_remote_code=False,  # Security: don't run arbitrary code
            )
            logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            logger.error("Failed to load HuggingFace model '%s': %s", self._model_name, e)
            raise RuntimeError(f"Failed to load model {self._model_name}: {e}") from e

    async def generate(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> ModelResponse:
        if self._task == "automatic-speech-recognition":
            return await self._generate_asr(request, token=token)
        if self._task == "text-classification":
            return await self._generate_text_classification(request, token=token)
        if self._task == "image-classification" or self._task == "object-detection":
            return await self._generate_vision_task(request, token=token)

        # Extract text from request
        if not request.parts:
            raise ValueError("Request must contain at least one part")

        text_parts = [part.text for part in request.parts if isinstance(part, TextPart)]
        if not text_parts:
            raise ValueError("HuggingFaceLLM requires at least one text part")

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
        self, request: ModelRequest, *, token: BiscuitToken | None
    ) -> ModelResponse:
        _ = token
        prompt = request.to_text_prompt(include_system=True)
        if not prompt:
            raise ValueError("text-classification requires at least one TextPart")

        self._ensure_model_loaded()
        loop = asyncio.get_event_loop()

        def _run() -> str:
            out = self._pipeline(prompt)
            # Common output: list[dict(label, score)] or dict(label, score)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                label = out[0].get("label")
                score = out[0].get("score")
                return f'{{"label": "{label}", "score": {score}}}'
            if isinstance(out, dict):
                label = out.get("label")
                score = out.get("score")
                return f'{{"label": "{label}", "score": {score}}}'
            return str(out)

        text = await loop.run_in_executor(None, _run)
        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    async def _generate_asr(
        self, request: ModelRequest, *, token: BiscuitToken | None
    ) -> ModelResponse:
        _ = token
        audio_parts = [p for p in request.parts if isinstance(p, AudioPart)]
        if not audio_parts:
            raise ValueError("automatic-speech-recognition requires at least one AudioPart")

        part = audio_parts[0]
        if not (part.uri or part.data_b64):
            raise ValueError("AudioPart requires either uri or data_b64")

        self._ensure_model_loaded()
        loop = asyncio.get_event_loop()

        def _run() -> str:
            # Easiest supported path: uri points to a local file path.
            if part.uri:
                out = self._pipeline(part.uri)
                if isinstance(out, dict) and isinstance(out.get("text"), str):
                    return out["text"]
                return str(out)

            raw = base64.b64decode(part.data_b64 or "")
            # Write to a temp file; transformers ASR pipelines commonly expect a file path.
            suffix = ".wav" if (part.mime or "").endswith("wav") else ".audio"
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as f:
                f.write(raw)
                f.flush()
                out = self._pipeline(f.name)
            if isinstance(out, dict) and isinstance(out.get("text"), str):
                return out["text"]
            return str(out)

        text = await loop.run_in_executor(None, _run)
        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    async def _generate_vision_task(
        self, request: ModelRequest, *, token: BiscuitToken | None
    ) -> ModelResponse:
        _ = token
        image_parts = [p for p in request.parts if isinstance(p, ImagePart)]
        if not image_parts:
            raise ValueError(f"{self._task} requires at least one ImagePart")

        part = image_parts[0]
        self._ensure_model_loaded()
        loop = asyncio.get_event_loop()

        def _run() -> str:
            # Load image from URI or b64
            import io

            from PIL import Image

            img: Image.Image
            if part.uri:
                img = Image.open(part.uri)
            elif part.data_b64:
                img = Image.open(io.BytesIO(base64.b64decode(part.data_b64)))
            else:
                raise ValueError("ImagePart must have uri or data_b64")

            out = self._pipeline(img)
            return str(out)

        text = await loop.run_in_executor(None, _run)
        return ModelResponse(
            text=text,
            raw_provider=ProviderInfo(
                provider="hf", model_name=self._model_name, model_key=f"hf/{self._model_name}"
            ),
        )

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation in thread pool."""
        self._ensure_model_loaded()

        try:
            # Generate with reasonable defaults for small models
            outputs = self._pipeline(
                prompt,
                max_new_tokens=128,  # Conservative for CPU/small models
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            if outputs and isinstance(outputs[0], dict):
                generated_text = outputs[0].get("generated_text", "")
                return generated_text
            else:
                return str(outputs[0]) if outputs else ""

        except Exception as e:
            logger.error("Error during HuggingFace generation: %s", e)
            raise RuntimeError(f"HuggingFace generation failed: {e}") from e

    async def stream(
        self,
        request: ModelRequest,
        *,
        token: BiscuitToken | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Basic streaming by yielding the full result at once.

        True token-by-token streaming would require more complex implementation
        with transformers generate() method and custom streaming logic.
        """
        result = await self.generate(request, token=token)
        yield TextDeltaEvent(delta=result.text)
        yield FinalTextEvent(text=result.text)

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0

        try:
            self._ensure_model_loaded()
            # Use tokenizer if available
            if hasattr(self._pipeline, "tokenizer"):
                tokens = self._pipeline.tokenizer.encode(text)
                return len(tokens)
        except Exception:
            logger.debug("Could not get accurate token count, using approximation")

        # Fallback: rough approximation
        return max(1, len(text.split()))

    def _clean_generated_text(self, prompt: str, generated: str) -> str:
        """Clean up common artifacts from text generation."""
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
