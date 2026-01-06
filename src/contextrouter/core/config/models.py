"""Model and LLM configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


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


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_llm: str = "vertex/gemini-2.5-flash"
    default_embeddings: str = "vertex/text-embedding"
    # Optional per-component overrides (keys are model registry keys).
    intent_llm: str = "vertex/gemini-2.5-flash-lite"
    suggestions_llm: str = "vertex/gemini-2.5-flash-lite"
    generation_llm: str = "vertex/gemini-2.5-flash"
    no_results_llm: str = "vertex/gemini-2.5-flash-lite"


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
