"""Model and LLM configuration — Pydantic schemas for provider selection, fallback chains, and response formats."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelSelectionStrategy = Literal["fallback", "parallel", "cost-priority"]


class ModelSelector(BaseModel):
    """Model selection + fallback for a single RAG component."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    model: str
    fallback: list[str] = Field(default_factory=list)
    strategy: ModelSelectionStrategy = "fallback"


class ModelsConfig(BaseModel):
    """Default model and embedding selection with optional fallback chain."""

    # Accept both `default_llm` and canonical `default` from TOML/env.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", populate_by_name=True)

    default_llm: str = Field(default="openai/gpt-5-mini", alias="default")
    default_embeddings: str = "hf/sentence-transformers"

    # Fallback LLM chain - used when default_llm fails (e.g., quota exceeded)
    # Set via CU_ROUTER_FALLBACK_LLMS="anthropic/claude-sonnet-4,google/gemini-2.5-flash"
    fallback_llms: list[str] = Field(default_factory=list)
    allow_global_fallback: bool = False


class LLMConfig(BaseModel):
    """Provider-agnostic LLM request controls.

    These settings are shared across all LLM providers.
    Model selection is controlled by `models.default_llm`.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    temperature: float = 0.2
    max_output_tokens: int = 1024
    timeout_sec: float = 60.0
    max_retries: int = 2
    merge_system_prompt: bool = False


class PrivacyConfig(BaseModel):
    """Router-local privacy controls (PII session, encryption TTL)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    # Ephemeral AES key rotation for in-process PII mapping store (seconds).
    pii_encryption_ttl_seconds: int = Field(default=60, ge=10, le=86400)


class RouterSection(BaseModel):
    """Router service instance configuration (tenants, feature flags, graph)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    # ── Server settings ───────────────────────────────────────────
    instance_name: str = "default"
    tenants: list[str] = Field(default_factory=list)

    # ── Feature flags ─────────────────────────────────────────────
    gcs_default_bucket: str = ""
    brain_index_tools: bool = False

    # ── Graph Selection ───────────────────────────────────────────
    graph: str | None = None
    override_path: str | None = None
