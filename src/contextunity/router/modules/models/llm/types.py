"""Typed LLM provider configurations.

Each provider validates its accepted parameters via a Pydantic model
that inherits from :class:`BaseLLMProviderConfig`. This keeps
provider-specific logic out of the graph executor (``llm.py``) and
ensures unknown keys are rejected early with clear errors.

The graph executor passes the full ``node_config`` (minus executor-level
keys) as ``ModelRequest.provider_config``. Each provider instantiates
its own config from that dict.
"""

from __future__ import annotations

from typing import ClassVar, Literal, get_args, override

from pydantic import BaseModel, ConfigDict

#: Valid reasoning effort levels for o-series and GPT-5 models.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
_REASONING_EFFORTS: frozenset[str] = frozenset(get_args(ReasoningEffort))
_REASONING_EFFORT_BY_KEY: dict[str, ReasoningEffort] = {
    "none": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}


def parse_reasoning_effort(raw: object) -> ReasoningEffort | None:
    """Validate and narrow a raw config value to ReasoningEffort.

    Returns None if the value is not a recognized effort level.
    """
    if isinstance(raw, str):
        return _REASONING_EFFORT_BY_KEY.get(raw)
    return None


# ── Base ──────────────────────────────────────────────────────────


class BaseLLMProviderConfig(BaseModel):
    """Universal LLM generation parameters shared across all providers.

    Providers that accept additional parameters extend this class.
    ``extra="ignore"`` on the base lets unknown keys pass through safely
    during the transition period; provider-specific subclasses tighten
    this to ``extra="forbid"`` once stabilised.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    temperature: float | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None  # alias used in some manifests
    timeout_sec: float | None = None
    response_format: Literal["text", "json_object"] | None = None

    def get_max_tokens(self) -> int | None:
        """Return whichever max-token field is set (prefer explicit)."""
        return self.max_output_tokens or self.max_tokens


class _OpenAICompletionTokensMixin(BaseLLMProviderConfig):
    """OpenAI-style providers that accept ``max_completion_tokens``."""

    max_completion_tokens: int | None = None

    @override
    def get_max_tokens(self) -> int | None:
        """Return max output tokens, including ``max_completion_tokens``."""
        return self.max_output_tokens or self.max_tokens or self.max_completion_tokens


# ── OpenAI ────────────────────────────────────────────────────────


class OpenAIProviderConfig(_OpenAICompletionTokensMixin):
    """OpenAI-specific generation parameters.

    Covers GPT-5 reasoning models (``reasoning_effort``,
    ``max_completion_tokens``) and classic models (``max_tokens``,
    ``temperature``).
    """

    reasoning_effort: ReasoningEffort | None = None


# ── Anthropic ─────────────────────────────────────────────────────


class AnthropicProviderConfig(BaseLLMProviderConfig):
    """Anthropic-specific generation parameters."""

    # Anthropic requires max_tokens (not optional) — but we keep it
    # optional here and let the provider apply a default.
    pass


# ── Vertex (Google) ───────────────────────────────────────────────


class VertexProviderConfig(BaseLLMProviderConfig):
    """Google Vertex AI generation parameters."""

    pass


# ── Groq ──────────────────────────────────────────────────────────


class GroqProviderConfig(BaseLLMProviderConfig):
    """Groq-specific generation parameters."""

    pass


# ── OpenRouter ────────────────────────────────────────────────────


class OpenRouterProviderConfig(BaseLLMProviderConfig):
    """OpenRouter-specific generation parameters."""

    pass


# ── Perplexity ────────────────────────────────────────────────────


class PerplexityProviderConfig(BaseLLMProviderConfig):
    """Perplexity-specific generation parameters."""

    pass


# ── Local (Ollama / vLLM) ─────────────────────────────────────────


class LocalProviderConfig(BaseLLMProviderConfig):
    """Local OpenAI-compatible server parameters."""

    pass


# ── Inception ─────────────────────────────────────────────────────


class InceptionProviderConfig(BaseLLMProviderConfig):
    """Inception (Mercury) provider parameters."""

    pass


# ── RunPod ────────────────────────────────────────────────────────


class RunPodProviderConfig(BaseLLMProviderConfig):
    """RunPod serverless provider parameters."""

    pass


# ── HuggingFace Hub ───────────────────────────────────────────────


class HFHubProviderConfig(BaseLLMProviderConfig):
    """HuggingFace Hub / Inference API parameters."""

    pass


__all__ = [
    "BaseLLMProviderConfig",
    "OpenAIProviderConfig",
    "AnthropicProviderConfig",
    "VertexProviderConfig",
    "GroqProviderConfig",
    "OpenRouterProviderConfig",
    "PerplexityProviderConfig",
    "LocalProviderConfig",
    "InceptionProviderConfig",
    "RunPodProviderConfig",
    "HFHubProviderConfig",
    "ReasoningEffort",
    "parse_reasoning_effort",
]
