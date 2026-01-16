"""Model and LLM configuration."""

from __future__ import annotations

from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RagConfig(BaseModel):
    """RAG datastore selection (compat for retrieval settings).

    Used by `contextrouter.modules.retrieval.rag.settings.resolve_data_store_id`.
    Values may be provided via TOML or env; env can still override at runtime.
    """

    model_config = ConfigDict(extra="ignore")

    # blue/green selector or full datastore id
    db_name: str = ""
    data_store_id_blue: str = ""
    data_store_id_green: str = ""


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


class IngestionModelsConfig(_ModelsGroup):
    """Per-ingestion-stage model configuration (canonical).

    Keep ingestion model choices in core config so ingestion TOML stays about ingestion behavior
    (paths/workers/filters), not model selection.
    """

    taxonomy: ModelSelector = Field(default_factory=_selector_factory("vertex/gemini-2.5-flash"))
    preprocess: ModelSelector = Field(
        default_factory=_selector_factory("vertex/gemini-2.5-flash-lite")
    )
    graph: ModelSelector = Field(default_factory=_selector_factory("vertex/gemini-2.5-pro"))
    persona: ModelSelector = Field(default_factory=_selector_factory("vertex/gemini-2.5-flash"))
    json_model: ModelSelector = Field(default_factory=_selector_factory("vertex/gemini-2.5-flash"))

    @field_validator("json_model")
    @classmethod
    def _require_json_model(cls, v: ModelSelector) -> ModelSelector:
        if not isinstance(v, ModelSelector) or not v.model.strip():
            raise ValueError("models.ingestion.json_model.model must be set")
        return v


class ModelsConfig(BaseModel):
    # Accept both `default_llm` and canonical `default` from TOML/env.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    default_llm: str = Field(default="vertex/gemini-2.5-flash", alias="default")
    default_embeddings: str = "hf/sentence-transformers"

    # Canonical per-component configuration:
    rag: RagModelsConfig = Field(default_factory=RagModelsConfig)
    ingestion: IngestionModelsConfig = Field(default_factory=IngestionModelsConfig)


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
