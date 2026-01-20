"""Strongly-typed Pydantic models for multimodal model contracts.

This module defines the runtime entities used by the multimodal model interface.
All types are Pydantic for validation and runtime type safety.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contextrouter.core.types import StructDataValue

# ---- Part Types (Discriminated Union) ----


class ModelPart(BaseModel):
    """Base class for model input parts (text, image, audio)."""

    model_config = ConfigDict(extra="forbid")

    kind: str


class TextPart(ModelPart):
    """Text input part."""

    kind: Literal["text"] = "text"
    text: str


class ImagePart(ModelPart):
    """Image input part."""

    kind: Literal["image"] = "image"
    mime: str
    data_b64: str | None = None
    uri: str | None = None


class AudioPart(ModelPart):
    """Audio input part."""

    kind: Literal["audio"] = "audio"
    mime: str
    data_b64: str | None = None
    uri: str | None = None
    sample_rate_hz: int | None = None


class VideoPart(ModelPart):
    """Video input part."""

    kind: Literal["video"] = "video"
    mime: str
    data_b64: str | None = None
    uri: str | None = None


# ---- Request/Response Types ----


class ModelCapabilities(BaseModel):
    """Model capabilities declaration."""

    model_config = ConfigDict(extra="forbid")

    supports_text: bool = True
    supports_image: bool = False
    supports_audio: bool = False
    supports_video: bool = False

    def supports(self, required: set[str]) -> bool:
        """Check if this model supports all required modalities."""
        mapping = {
            "text": self.supports_text,
            "image": self.supports_image,
            "audio": self.supports_audio,
            "video": self.supports_video,
        }
        return all(mapping.get(mod, False) for mod in required)


class ModelRequest(BaseModel):
    """Multimodal model request."""

    model_config = ConfigDict(extra="forbid")

    parts: list[ModelPart] = Field(default_factory=list, min_length=1)
    system: str | None = None
    metadata: dict[str, StructDataValue] = Field(default_factory=dict)

    # Generation controls
    temperature: float | None = None
    max_output_tokens: int | None = None  # None = use model default
    timeout_sec: float | None = None
    max_retries: int | None = None

    def required_modalities(self) -> set[str]:
        """Extract the set of required modalities from parts."""
        return {part.kind for part in self.parts}

    def to_text_prompt(self, *, include_system: bool = False) -> str:
        """Best-effort conversion into a plain text prompt (text-only).

        Default behavior excludes `system` because providers that support system prompts
        should pass it separately. Use `include_system=True` only for providers that
        do not support a separate system prompt and need a merged prompt.
        """
        text_parts: list[str] = []
        if include_system and isinstance(self.system, str) and self.system.strip():
            text_parts.append(self.system.strip())
        for part in self.parts:
            if isinstance(part, TextPart) and part.text:
                text_parts.append(part.text)
        return "\n\n".join(text_parts).strip()


class UsageStats(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Provider-specific costs if available
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None


class ProviderInfo(BaseModel):
    """Normalized provider information."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model_name: str
    model_key: str


class ModelResponse(BaseModel):
    """Model response."""

    model_config = ConfigDict(extra="forbid")

    text: str
    usage: UsageStats | None = None
    raw_provider: ProviderInfo


# ---- Stream Event Types (Discriminated Union) ----


class ModelStreamEvent(BaseModel):
    """Base class for streaming events."""

    model_config = ConfigDict(extra="forbid")

    event_type: str


class TextDeltaEvent(ModelStreamEvent):
    """Incremental text delta."""

    event_type: Literal["text_delta"] = "text_delta"
    delta: str


class FinalTextEvent(ModelStreamEvent):
    """Final complete text."""

    event_type: Literal["final_text"] = "final_text"
    text: str


class UsageEvent(ModelStreamEvent):
    """Usage statistics."""

    event_type: Literal["usage"] = "usage"
    usage: UsageStats


class ErrorEvent(ModelStreamEvent):
    """Error during generation."""

    event_type: Literal["error"] = "error"
    error: str
    provider_info: ProviderInfo | None = None


# ---- Error Types ----


class ModelError(Exception):
    """Base exception for model-related errors."""

    def __init__(self, message: str, provider_info: ProviderInfo | None = None) -> None:
        super().__init__(message)
        self.provider_info = provider_info


class ModelCapabilityError(ModelError):
    """Raised when a model doesn't support required modalities."""

    pass


class ModelExhaustedError(ModelError):
    """Raised when all candidate models fail for reasons other than capability mismatch."""


class ModelTimeoutError(ModelError):
    """Raised on generation timeout."""

    pass


class ModelRateLimitError(ModelError):
    """Raised on rate limiting."""

    pass


# ---- Type Exports ----

__all__ = [
    "ModelPart",
    "TextPart",
    "ImagePart",
    "AudioPart",
    "VideoPart",
    "ModelCapabilities",
    "ModelRequest",
    "UsageStats",
    "ProviderInfo",
    "ModelResponse",
    "ModelStreamEvent",
    "TextDeltaEvent",
    "FinalTextEvent",
    "UsageEvent",
    "ErrorEvent",
    "ModelError",
    "ModelCapabilityError",
    "ModelExhaustedError",
    "ModelTimeoutError",
    "ModelRateLimitError",
]
