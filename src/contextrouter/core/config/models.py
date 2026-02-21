"""Model and LLM configuration."""

from __future__ import annotations

from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelSelectionStrategy = Literal["fallback", "parallel", "cost-priority"]


class ModelSelector(BaseModel):
    """Model selection + fallback for a single RAG component."""

    model_config = ConfigDict(extra="ignore")

    model: str
    fallback: list[str] = Field(default_factory=list)
    strategy: ModelSelectionStrategy = "fallback"


def _selector(model: str) -> ModelSelector:
    # Used only for type inference/documentation; do not call directly as a default_factory.
    return ModelSelector(model=model)


def _selector_factory(model: str):
    # `default_factory` must be a zero-arg callable; `partial` is perfect for this.
    return partial(ModelSelector, model=model)


class _ModelsGroup(BaseModel):
    """Shared base for model-group configs (RAG, ingestion, etc.)."""

    model_config = ConfigDict(extra="ignore")


class RagModelsConfig(_ModelsGroup):
    """Per-RAG-component model configuration (canonical)."""

    intent: ModelSelector = Field(default_factory=_selector_factory("vertex/gemini-2.5-flash-lite"))
    suggestions: ModelSelector = Field(
        default_factory=_selector_factory("vertex/gemini-2.5-flash-lite")
    )
    generation: ModelSelector = Field(default_factory=_selector_factory("vertex/gemini-2.5-flash"))
    no_results: ModelSelector = Field(
        default_factory=_selector_factory("vertex/gemini-2.5-flash-lite")
    )


class ModelsConfig(BaseModel):
    # Accept both `default_llm` and canonical `default` from TOML/env.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    default_llm: str = Field(default="vertex/gemini-2.5-flash", alias="default")
    default_embeddings: str = "hf/sentence-transformers"

    # Fallback LLM chain - used when default_llm fails (e.g., quota exceeded)
    # Set via CONTEXTROUTER_FALLBACK_LLMS="anthropic/claude-sonnet-4,google/gemini-2.5-flash"
    fallback_llms: list[str] = Field(default_factory=list)

    # Canonical per-component configuration:
    rag: RagModelsConfig = Field(default_factory=RagModelsConfig)


class LLMConfig(BaseModel):
    """Provider-agnostic LLM request controls.

    These settings are shared across all LLM providers.
    Model selection is controlled by `models.default_llm`.
    """

    model_config = ConfigDict(extra="ignore")

    temperature: float = 0.2
    max_output_tokens: int = 1024
    timeout_sec: float = 60.0
    max_retries: int = 2
    merge_system_prompt: bool = False


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    override_path: str | None = None
    # Which cortex graph to run (lookup key in graph registry).
    # Examples: "rag_retrieval", "rag_ingestion", "brain" (if you register it)
    graph: str = "rag_retrieval"
    # Graph assembly mode:
    # - "agent": class-based nodes registered in `agent_registry` (default)
    # - "direct": function-based nodes (simple flows, no agent instantiation)
    mode: Literal["agent", "direct"] = "agent"

    # ── Server settings ───────────────────────────────────────────
    port: str = "50050"
    instance_name: str = "default"
    tenants: list[str] = Field(default_factory=list)

    # ── External service endpoints ────────────────────────────────
    worker_grpc_endpoint: str = "localhost:50052"
    contextzero_grpc_host: str = ""
    contextshield_grpc_host: str = ""
    gcs_default_bucket: str = ""
    brain_index_tools: bool = False


class GardenerConfig(BaseModel):
    """Configuration for Gardener enrichment agent.

    Gardener processes DealerProduct enrichment:
    - taxonomy classification
    - NER extraction (product_type, brand, model)
    - Knowledge Graph updates

    Note: Paths to prompts/ontology are commerce-specific.
    They should be passed from Worker config, not stored here.
    """

    model_config = ConfigDict(extra="ignore")

    # Processing
    batch_size: int = 50
    poll_interval_sec: int = 900  # 15 min default

    # LLM settings
    llm_model: str = "vertex/gemini-2.5-flash-lite"

    # Tenant - MUST be set via env/config
    tenant_id: str = ""


class NewsEngineConfig(BaseModel):
    """Configuration for News Engine graph.

    Controls news harvesting, generation, and post-processing.
    Set via NEWS_ENGINE_* environment variables.
    """

    model_config = ConfigDict(extra="ignore")

    # LanguageTool grammar/spell checking
    # Set via NEWS_ENGINE_LANGUAGE_TOOL_LANG=uk (or en, de, etc.)
    language_tool_enabled: bool = False
    language_tool_lang: str = "uk"  # Ukrainian by default
    language_tool_auto_correct: bool = True  # Auto-apply corrections

    # Generation settings
    max_posts_per_run: int = 10
    min_article_chars: int = 1100
    max_article_chars: int = 1800

    # Deduplication
    dedupe_similarity_threshold: float = 0.85
