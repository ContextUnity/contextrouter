"""RAG retrieval module (orchestration + ranking + formatting).

This package is intentionally *lazy-imported* to avoid cycles with `contextrouter.cortex`.
Import from here for DX; heavy modules are loaded on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextrouter.modules.retrieval.rag.citations import CitationBuilder, build_citations
    from contextrouter.modules.retrieval.rag.formatting.citations import format_citations_to_ui
    from contextrouter.modules.retrieval.rag.models import Citation, RetrievedDoc
    from contextrouter.modules.retrieval.rag.pipeline import RetrievalPipeline as RagPipeline
    from contextrouter.modules.retrieval.rag.pipeline import RetrievalResult as RagResult
    from contextrouter.modules.retrieval.rag.ranking import rerank_documents
    from contextrouter.modules.retrieval.rag.runtime import (
        get_runtime_settings,
        use_runtime_settings,
    )
    from contextrouter.modules.retrieval.rag.settings import (
        RagRetrievalSettings,
        get_effective_data_store_id,
        get_rag_retrieval_settings,
        resolve_data_store_id,
    )
    from contextrouter.modules.retrieval.rag.types import (
        RawCitation,
        RuntimeRagSettings,
        SourceType,
        UICitation,
    )

__all__ = [
    "RagPipeline",
    "RagResult",
    "rerank_documents",
    "RagRetrievalSettings",
    "get_rag_retrieval_settings",
    "resolve_data_store_id",
    "get_effective_data_store_id",
    "use_runtime_settings",
    "get_runtime_settings",
    "build_citations",
    "CitationBuilder",
    "format_citations_to_ui",
    "SourceType",
    "UICitation",
    "RetrievedDoc",
    "RawCitation",
    "RuntimeRagSettings",
    "Citation",
]

_EXPORTS: dict[str, str] = {
    "RagPipeline": "contextrouter.modules.retrieval.rag.pipeline.RetrievalPipeline",
    "RagResult": "contextrouter.modules.retrieval.rag.pipeline.RetrievalResult",
    "rerank_documents": "contextrouter.modules.retrieval.rag.ranking.rerank_documents",
    "RagRetrievalSettings": "contextrouter.modules.retrieval.rag.settings.RagRetrievalSettings",
    "get_rag_retrieval_settings": "contextrouter.modules.retrieval.rag.settings.get_rag_retrieval_settings",
    "resolve_data_store_id": "contextrouter.modules.retrieval.rag.settings.resolve_data_store_id",
    "get_effective_data_store_id": "contextrouter.modules.retrieval.rag.settings.get_effective_data_store_id",
    "use_runtime_settings": "contextrouter.modules.retrieval.rag.runtime.use_runtime_settings",
    "get_runtime_settings": "contextrouter.modules.retrieval.rag.runtime.get_runtime_settings",
    "build_citations": "contextrouter.modules.retrieval.rag.citations.build_citations",
    "CitationBuilder": "contextrouter.modules.retrieval.rag.citations.CitationBuilder",
    "format_citations_to_ui": "contextrouter.modules.retrieval.rag.formatting.citations.format_citations_to_ui",
    "SourceType": "contextrouter.modules.retrieval.rag.types.SourceType",
    "UICitation": "contextrouter.modules.retrieval.rag.types.UICitation",
    "RetrievedDoc": "contextrouter.modules.retrieval.rag.models.RetrievedDoc",
    "RawCitation": "contextrouter.modules.retrieval.rag.types.RawCitation",
    "RuntimeRagSettings": "contextrouter.modules.retrieval.rag.types.RuntimeRagSettings",
    "Citation": "contextrouter.modules.retrieval.rag.models.Citation",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    path = _EXPORTS[name]
    mod_name, attr = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)
