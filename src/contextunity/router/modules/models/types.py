"""Strongly-typed Pydantic models for multimodal model contracts.
This module defines the runtime entities used by the multimodal model interface.
All types are Pydantic for validation and runtime type safety.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Literal

from contextunity.core.exceptions import ContextUnityError, register_error
from contextunity.core.types import WireValue
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.core.types import StructDataValue

# ---- Model Type Discriminator ----


class ModelType(str, Enum):
    """Discriminator for model registry variants."""

    LLM = "llm"  # Text generation (chat, completion, reasoning)
    EMBEDDINGS = "embeddings"  # Vector embeddings
    # Future: CLASSIFIER = "classifier", ASR = "asr", VISION = "vision"


# ---- Response Format ----


class ResponseFormat(str, Enum):
    """Strict output contract for LLM nodes.

    When ``json_object`` is requested but the model returns invalid JSON,
    ``ModelResponseFormatError`` is raised.  If ``response_format`` is listed
    in the active ``RetryPolicy.retry_on``, the same model is retried first.
    After retry exhaustion the error propagates to ``FallbackModel`` which
    tries the next candidate.
    """

    TEXT = "text"
    JSON_OBJECT = "json_object"


# ---- Part Types (Discriminated Union) ----


class ModelPart(BaseModel):
    """Base class for model input parts (text, image, audio)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    kind: str


class TextPart(ModelPart):
    """Text input part."""

    kind: str = "text"
    text: str


class ImagePart(ModelPart):
    """Image input part."""

    kind: str = "image"
    mime: str
    data_b64: str | None = None
    uri: str | None = None


class AudioPart(ModelPart):
    """Audio input part."""

    kind: str = "audio"
    mime: str
    data_b64: str | None = None
    uri: str | None = None
    sample_rate_hz: int | None = None


class VideoPart(ModelPart):
    """Video input part."""

    kind: str = "video"
    mime: str
    data_b64: str | None = None
    uri: str | None = None


# ---- Request/Response Types ----


class ModelCapabilities(BaseModel):
    """Model capabilities declaration."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

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

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    parts: list[ModelPart] = Field(default_factory=list, min_length=1)
    system: str | None = None
    metadata: dict[str, StructDataValue] = Field(default_factory=dict)

    # Generation controls (legacy — prefer provider_config)
    temperature: float | None = None
    max_output_tokens: int | None = None  # None = use model default
    timeout_sec: float | None = None
    max_retries: int | None = None

    # Response format (for JSON mode)
    response_format: Literal["text", "json_object"] | None = None

    # Provider-specific config passed through from manifest node config.
    # Each provider validates this dict via its own Pydantic model
    # (see modules/models/llm/types.py).
    provider_config: dict[str, WireValue] = Field(default_factory=dict)

    def required_modalities(self) -> set[str]:
        """Extract the set of required modalities from parts."""
        return {part.kind for part in self.parts}

    def to_text_prompt(self, *, include_system: bool = False) -> str:
        """Concatenate text-only parts (and optionally the system prompt) into a single plain string."""
        text_parts: list[str] = []
        if include_system and isinstance(self.system, str) and self.system.strip():
            text_parts.append(self.system.strip())
        for part in self.parts:
            if isinstance(part, TextPart) and part.text:
                text_parts.append(part.text)
        return "\n\n".join(text_parts).strip()


class UsageStats(BaseModel):
    """Token usage statistics."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Provider-specific costs if available
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None

    # ── Cost estimation per 1M tokens: (input_usd, output_usd) ──
    # Source: https://pricepertoken.com  |  API: https://www.llm-prices.com/current-v1.json
    # Last verified: 2026-02-23
    # Only models actually configured in our providers (see modules/models/llm/).
    _PRICE_TABLE: dict[str, tuple[float, float]] = {
        # OpenAI  (openai.py default: gpt-5.1)
        "gpt-5.1": (1.25, 10.0),
        "gpt-5.1-mini": (0.25, 2.0),
        "gpt-5.1-nano": (0.05, 0.40),
        "gpt-5-mini": (0.25, 2.0),  # RLM default
        "gpt-5-nano": (0.05, 0.40),
        # Anthropic  (anthropic.py default: claude-sonnet-4.5)
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-sonnet-4.5": (3.0, 15.0),
        "claude-sonnet-4.6": (3.0, 15.0),
        "claude-3-5-haiku": (0.80, 4.0),
        # Google Vertex  (vertex.py default: gemini-2.5-flash)
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
        "gemini-2.5-pro": (1.25, 10.0),
        # Groq  (groq.py default: llama-3.3-70b-versatile)
        "llama-3.3-70b-versatile": (0.59, 0.79),
        # Mercury 2
        "mercury-2": (0.25, 0.75),
    }

    def estimate_cost(self, model_name: str) -> "UsageStats":
        """Fill in cost fields from token counts and model price table."""
        if self.total_cost is not None:
            return self  # Already set by provider

        # Fuzzy match: try exact, then prefix match
        bare_model = model_name.split("/")[-1]

        prices = self._PRICE_TABLE.get(model_name) or self._PRICE_TABLE.get(bare_model)
        if not prices:
            for key, val in self._PRICE_TABLE.items():
                if bare_model.startswith(key) or key.startswith(bare_model):
                    prices = val
                    break
        if not prices:
            return self

        in_price, out_price = prices
        inp = self.input_tokens or 0
        out = self.output_tokens or 0
        self.input_cost = round(inp * in_price / 1_000_000, 8)
        self.output_cost = round(out * out_price / 1_000_000, 8)
        self.total_cost = round(self.input_cost + self.output_cost, 8)
        return self


class ProviderInfo(BaseModel):
    """Normalized provider information."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    provider: str
    model_name: str
    model_key: str


class ModelResponse(BaseModel):
    """Complete generation result returned by a provider’s ``_generate()`` method."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    text: str
    usage: UsageStats | None = None
    raw_provider: ProviderInfo


# ---- Stream Event Types (Discriminated Union) ----


class ModelStreamEvent(BaseModel):
    """Base class for streaming events."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    event_type: str


class TextDeltaEvent(ModelStreamEvent):
    """Incremental text delta."""

    event_type: str = "text_delta"
    delta: str


class FinalTextEvent(ModelStreamEvent):
    """Final complete text."""

    event_type: str = "final_text"
    text: str


class UsageEvent(ModelStreamEvent):
    """Trailing stream event carrying token usage and cost data."""

    event_type: str = "usage"
    usage: UsageStats


class ErrorEvent(ModelStreamEvent):
    """Error during generation."""

    event_type: str = "error"
    error: str
    provider_info: ProviderInfo | None = None


# ---- Error Types ----


@register_error("ROUTER_MODEL_ERROR")
class ModelError(ContextUnityError):
    """Base exception for model-related errors."""

    code: str = "ROUTER_MODEL_ERROR"
    message: str = "Model invocation failed"

    def __init__(
        self,
        message: str,
        provider_info: ProviderInfo | None = None,
        code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Store the error *message* and optional *provider_info* for diagnostics."""
        super().__init__(message=message, code=code, **kwargs)
        self.provider_info: ProviderInfo | None = provider_info


@register_error("ROUTER_MODEL_CAPABILITY")
class ModelCapabilityError(ModelError):
    """Raised when a model doesn't support required modalities."""

    code: str = "ROUTER_MODEL_CAPABILITY"
    message: str = "Model does not support required modalities"


@register_error("ROUTER_MODEL_EXHAUSTED")
class ModelExhaustedError(ModelError):
    """Raised when all candidate models fail for reasons other than capability mismatch."""

    code: str = "ROUTER_MODEL_EXHAUSTED"
    message: str = "All candidate models failed"


@register_error("ROUTER_MODEL_TIMEOUT")
class ModelTimeoutError(ModelError):
    """Raised on generation timeout."""

    code: str = "ROUTER_MODEL_TIMEOUT"
    message: str = "Model generation timed out"


@register_error("ROUTER_MODEL_RATE_LIMIT")
class ModelRateLimitError(ModelError):
    """Raised on rate limiting (transient, can retry after delay)."""

    code: str = "ROUTER_MODEL_RATE_LIMIT"
    message: str = "Model rate limited, retry later"


@register_error("ROUTER_MODEL_QUOTA_EXHAUSTED")
class ModelQuotaExhaustedError(ModelError):
    """Raised when API quota/billing is exhausted (NOT transient, should fallback immediately)."""

    code: str = "ROUTER_MODEL_QUOTA_EXHAUSTED"
    message: str = "Model API quota exhausted"


@register_error("ROUTER_MODEL_RESPONSE_FORMAT")
class ModelResponseFormatError(ModelError):
    """Raised when the model returns a response that violates the requested format.

    E.g. ``response_format=json_object`` was requested but the output is not valid JSON.
    Extends ``ModelError`` so ``FallbackModel`` catches it and tries the next candidate.
    """

    code: str = "ROUTER_MODEL_RESPONSE_FORMAT"
    message: str = "Model response format violation"


@register_error("ROUTER_MODEL_BUDGET_EXCEEDED")
class ModelBudgetExceededError(ModelError):
    """Raised when cumulative cost exceeds the per-request ``budget_usd`` cap.

    Non-retryable — immediately propagates to ``FallbackModel``.
    """

    code: str = "ROUTER_MODEL_BUDGET_EXCEEDED"
    message: str = "Model budget exceeded"


# ---- Type Exports ----

__all__ = [
    "ModelType",
    "ResponseFormat",
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
    "ModelQuotaExhaustedError",
    "ModelResponseFormatError",
    "ModelBudgetExceededError",
]
