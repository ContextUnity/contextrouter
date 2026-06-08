"""RAG retrieval module (orchestration + ranking + formatting).

This package is intentionally *lazy-imported* to avoid cycles with `contextunity.router.cortex`.
Import from here for DX; heavy modules are loaded on first attribute access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextunity.router.modules.retrieval.rag.citations import CitationBuilder, build_citations
    from contextunity.router.modules.retrieval.rag.formatting.citations import (
        format_citations_to_ui,
    )
    from contextunity.router.modules.retrieval.rag.models import Citation, RetrievedDoc
    from contextunity.router.modules.retrieval.rag.pipeline import RetrievalPipeline as RagPipeline
    from contextunity.router.modules.retrieval.rag.pipeline import RetrievalResult as RagResult
    from contextunity.router.modules.retrieval.rag.ranking import rerank_documents
    from contextunity.router.modules.retrieval.rag.runtime import (
        get_runtime_settings,
        use_runtime_settings,
    )
    from contextunity.router.modules.retrieval.rag.settings import (
        RagRetrievalSettings,
        get_effective_data_store_id,
        get_rag_retrieval_settings,
        resolve_data_store_id,
    )
    from contextunity.router.modules.retrieval.rag.types import (
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
    "RagPipeline": "contextunity.router.modules.retrieval.rag.pipeline.RetrievalPipeline",
    "RagResult": "contextunity.router.modules.retrieval.rag.pipeline.RetrievalResult",
    "rerank_documents": "contextunity.router.modules.retrieval.rag.ranking.rerank_documents",
    "RagRetrievalSettings": "contextunity.router.modules.retrieval.rag.settings.RagRetrievalSettings",
    "get_rag_retrieval_settings": "contextunity.router.modules.retrieval.rag.settings.get_rag_retrieval_settings",
    "resolve_data_store_id": "contextunity.router.modules.retrieval.rag.settings.resolve_data_store_id",
    "get_effective_data_store_id": "contextunity.router.modules.retrieval.rag.settings.get_effective_data_store_id",
    "use_runtime_settings": "contextunity.router.modules.retrieval.rag.runtime.use_runtime_settings",
    "get_runtime_settings": "contextunity.router.modules.retrieval.rag.runtime.get_runtime_settings",
    "build_citations": "contextunity.router.modules.retrieval.rag.citations.build_citations",
    "CitationBuilder": "contextunity.router.modules.retrieval.rag.citations.CitationBuilder",
    "format_citations_to_ui": "contextunity.router.modules.retrieval.rag.formatting.citations.format_citations_to_ui",
    "SourceType": "contextunity.router.modules.retrieval.rag.types.SourceType",
    "UICitation": "contextunity.router.modules.retrieval.rag.types.UICitation",
    "RetrievedDoc": "contextunity.router.modules.retrieval.rag.models.RetrievedDoc",
    "RawCitation": "contextunity.router.modules.retrieval.rag.types.RawCitation",
    "RuntimeRagSettings": "contextunity.router.modules.retrieval.rag.types.RuntimeRagSettings",
    "Citation": "contextunity.router.modules.retrieval.rag.models.Citation",
}


def __getattr__(name: str) -> object:
    if name == "RagPipeline":
        from contextunity.router.modules.retrieval.rag.pipeline import RetrievalPipeline

        return RetrievalPipeline
    if name == "RagResult":
        from contextunity.router.modules.retrieval.rag.pipeline import RetrievalResult

        return RetrievalResult
    if name == "rerank_documents":
        from contextunity.router.modules.retrieval.rag.ranking import rerank_documents

        return rerank_documents
    if name == "RagRetrievalSettings":
        from contextunity.router.modules.retrieval.rag.settings import RagRetrievalSettings

        return RagRetrievalSettings
    if name == "get_rag_retrieval_settings":
        from contextunity.router.modules.retrieval.rag.settings import get_rag_retrieval_settings

        return get_rag_retrieval_settings
    if name == "resolve_data_store_id":
        from contextunity.router.modules.retrieval.rag.settings import resolve_data_store_id

        return resolve_data_store_id
    if name == "get_effective_data_store_id":
        from contextunity.router.modules.retrieval.rag.settings import get_effective_data_store_id

        return get_effective_data_store_id
    if name == "use_runtime_settings":
        from contextunity.router.modules.retrieval.rag.runtime import use_runtime_settings

        return use_runtime_settings
    if name == "get_runtime_settings":
        from contextunity.router.modules.retrieval.rag.runtime import get_runtime_settings

        return get_runtime_settings
    if name == "build_citations":
        from contextunity.router.modules.retrieval.rag.citations import build_citations

        return build_citations
    if name == "CitationBuilder":
        from contextunity.router.modules.retrieval.rag.citations import CitationBuilder

        return CitationBuilder
    if name == "format_citations_to_ui":
        from contextunity.router.modules.retrieval.rag.formatting.citations import (
            format_citations_to_ui,
        )

        return format_citations_to_ui
    if name == "SourceType":
        from contextunity.router.modules.retrieval.rag.types import SourceType

        return SourceType
    if name == "UICitation":
        from contextunity.router.modules.retrieval.rag.types import UICitation

        return UICitation
    if name == "RetrievedDoc":
        from contextunity.router.modules.retrieval.rag.models import RetrievedDoc

        return RetrievedDoc
    if name == "RawCitation":
        from contextunity.router.modules.retrieval.rag.types import RawCitation

        return RawCitation
    if name == "RuntimeRagSettings":
        from contextunity.router.modules.retrieval.rag.types import RuntimeRagSettings

        return RuntimeRagSettings
    if name == "Citation":
        from contextunity.router.modules.retrieval.rag.models import Citation

        return Citation
    raise AttributeError(name)
