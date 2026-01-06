"""RAG / retrieval configuration (capability-specific).

This module owns *RAG-specific* defaults and env/runtime overrides:
- retrieval limits (per-source vs general mode)
- citation limits
- blue/green datastore selection and resolution

Core (`contextrouter.core`) should not contain these settings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from contextrouter.core import get_core_config
from contextrouter.core.config import get_env

from .runtime import get_runtime_settings
from .types import RuntimeRagSettings

RagDataset = Literal["blue", "green"]


class RagRetrievalSettings(BaseModel):
    """RAG retrieval and citation settings (UI-facing knobs)."""

    model_config = ConfigDict(extra="ignore")

    # Per-source retrieval (used when general_retrieval_enabled=False)
    max_books: int = 10
    max_videos: int = 10
    max_qa: int = 10
    max_knowledge: int = 5
    # Optional dynamic per-type limits (for modular source types).
    # If provided, this overrides legacy max_* fields.
    max_by_type: dict[str, int] = Field(default_factory=dict)

    # General retrieval mode (bypasses per-source selection)
    general_retrieval_enabled: bool = False
    general_retrieval_initial_count: int = 30  # Fetch top N chunks (10-200)
    general_retrieval_final_count: int = 15  # After reranking, take top M (1-100)

    # Reranking (optional)
    reranking_enabled: bool = False
    ranker_model: str = "semantic-ranker-default@latest"

    # Citation UI limits (per type)
    citations_enabled: bool = True
    citations_books: int = 10
    citations_videos: int = 10
    citations_qa: int = 10
    citations_web: int = 3

    # Retrieval sources (registry keys). These are loaded lazily on first use.
    # Providers must implement IRead; connectors must implement BaseConnector.
    providers: list[str] = Field(default_factory=lambda: ["vertex"])
    connectors: list[str] = Field(default_factory=lambda: ["web"])

    # Track which keys are locked by env vars (for host UI).
    env_locked: set[str] = Field(default_factory=set)

    def apply_runtime_overrides(self, runtime: RuntimeRagSettings) -> "RagRetrievalSettings":
        """Return a copy with runtime overrides applied, respecting env locks."""
        if not runtime:
            return self

        data = self.model_dump()
        locked = set(self.env_locked)

        def _set(key: str, val: Any) -> None:
            if key in locked:
                return
            data[key] = val

        for key in (
            "max_books",
            "max_videos",
            "max_qa",
            "max_knowledge",
            "max_by_type",
            "citations_enabled",
            "citations_books",
            "citations_videos",
            "citations_qa",
            "citations_web",
            "general_retrieval_enabled",
            "general_retrieval_initial_count",
            "general_retrieval_final_count",
            "reranking_enabled",
            "ranker_model",
        ):
            if key in runtime and runtime.get(key) is not None:
                _set(key, runtime.get(key))

        next_cfg = RagRetrievalSettings.model_validate(data)
        next_cfg.env_locked = locked
        return next_cfg


def _get_env_or_none(key: str) -> str | None:
    return get_env(key)


@lru_cache(maxsize=8)
def resolve_data_store_id(rag_db_name: str | None = None) -> str:
    """Resolve RAG datastore selection to an actual Vertex datastore id.

    `rag_db_name` can be:
    - "blue" -> resolves to DATA_STORE_ID_BLUE
    - "green" -> resolves to DATA_STORE_ID_GREEN
    - actual datastore id -> returned as-is
    """
    cfg = get_core_config()
    rag_cfg = getattr(cfg, "rag", None)
    if isinstance(rag_cfg, dict):
        cfg_db_name = str(rag_cfg.get("db_name") or "").strip()
        cfg_blue = str(rag_cfg.get("data_store_id_blue") or "").strip()
        cfg_green = str(rag_cfg.get("data_store_id_green") or "").strip()
    else:
        cfg_db_name = str(getattr(rag_cfg, "db_name", "") or "").strip()
        cfg_blue = str(getattr(rag_cfg, "data_store_id_blue", "") or "").strip()
        cfg_green = str(getattr(rag_cfg, "data_store_id_green", "") or "").strip()

    # Prefer explicit arg, then environment (host app style), then core config.
    raw = (rag_db_name or _get_env_or_none("RAG_DB_NAME") or cfg_db_name or "").strip()
    if not raw:
        raise ValueError("RAG datastore name (e.g. RAG_DB_NAME) is required")

    name = raw.lower()
    if name == "blue":
        resolved = (_get_env_or_none("DATA_STORE_ID_BLUE") or cfg_blue).strip()
        if not resolved:
            raise ValueError("RAG datastore is 'blue' but DATA_STORE_ID_BLUE is not set")
        return resolved
    if name == "green":
        resolved = (_get_env_or_none("DATA_STORE_ID_GREEN") or cfg_green).strip()
        if not resolved:
            raise ValueError("RAG datastore is 'green' but DATA_STORE_ID_GREEN is not set")
        return resolved
    return raw


def get_effective_data_store_id(*, runtime: RuntimeRagSettings | None = None) -> str:
    """Resolve datastore id using runtime override when provided."""
    rt = runtime if runtime is not None else get_runtime_settings()
    ds = rt.get("rag_dataset") if isinstance(rt, dict) else None
    if isinstance(ds, str) and ds.strip().lower() in {"blue", "green"}:
        return resolve_data_store_id(ds.strip().lower())
    return resolve_data_store_id()


def get_rag_retrieval_settings(
    *, runtime: RuntimeRagSettings | None = None
) -> RagRetrievalSettings:
    """Load RagRetrievalSettings from env + apply runtime overrides."""
    locked: set[str] = set()

    cfg = RagRetrievalSettings()

    def _lock(key: str) -> None:
        locked.add(key)

    # Per-source limits
    if v := _get_env_or_none("MAX_BOOKS"):
        try:
            cfg.max_books = max(1, min(20, int(v)))
            _lock("max_books")
        except ValueError:
            pass
    if v := _get_env_or_none("MAX_VIDEOS"):
        try:
            cfg.max_videos = max(1, min(20, int(v)))
            _lock("max_videos")
        except ValueError:
            pass
    if v := _get_env_or_none("MAX_QA"):
        try:
            cfg.max_qa = max(1, min(20, int(v)))
            _lock("max_qa")
        except ValueError:
            pass
    if v := _get_env_or_none("MAX_KNOWLEDGE"):
        try:
            cfg.max_knowledge = max(1, min(20, int(v)))
            _lock("max_knowledge")
        except ValueError:
            pass

    # Citation limits
    if v := _get_env_or_none("CITATIONS_BOOKS"):
        try:
            cfg.citations_books = max(0, min(20, int(v)))
            _lock("citations_books")
        except ValueError:
            pass
    if v := _get_env_or_none("CITATIONS_VIDEOS"):
        try:
            cfg.citations_videos = max(0, min(20, int(v)))
            _lock("citations_videos")
        except ValueError:
            pass
    if v := _get_env_or_none("CITATIONS_QA"):
        try:
            cfg.citations_qa = max(0, min(20, int(v)))
            _lock("citations_qa")
        except ValueError:
            pass
    if v := _get_env_or_none("CITATIONS_WEB"):
        try:
            cfg.citations_web = max(0, min(10, int(v)))
            _lock("citations_web")
        except ValueError:
            pass

    # General retrieval mode
    if v := _get_env_or_none("GENERAL_RETRIEVAL_ENABLED"):
        cfg.general_retrieval_enabled = v.lower() in {"1", "true", "yes", "on"}
        _lock("general_retrieval_enabled")
    if v := _get_env_or_none("GENERAL_RETRIEVAL_INITIAL_COUNT"):
        try:
            cfg.general_retrieval_initial_count = max(10, min(200, int(v)))
            _lock("general_retrieval_initial_count")
        except ValueError:
            pass
    if v := _get_env_or_none("GENERAL_RETRIEVAL_FINAL_COUNT"):
        try:
            cfg.general_retrieval_final_count = max(1, min(100, int(v)))
            _lock("general_retrieval_final_count")
        except ValueError:
            pass

    # Reranking
    if v := _get_env_or_none("RERANKING_ENABLED"):
        cfg.reranking_enabled = v.lower() in {"1", "true", "yes", "on"}
        _lock("reranking_enabled")
    if v := _get_env_or_none("RANKER_MODEL"):
        cfg.ranker_model = v
        _lock("ranker_model")

    cfg.env_locked = locked

    rt = runtime if runtime is not None else get_runtime_settings()
    return cfg.apply_runtime_overrides(rt if isinstance(rt, dict) else {})


__all__ = [
    "RagRetrievalSettings",
    "get_rag_retrieval_settings",
    "get_effective_data_store_id",
    "resolve_data_store_id",
]
