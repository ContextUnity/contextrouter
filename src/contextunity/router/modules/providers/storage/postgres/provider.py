"""Postgres provider (storage + retrieval).
Uses ContextUnit protocol for data transport.
"""

from __future__ import annotations

from typing import override

from contextunity.core import ContextUnit
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.sdk.payload import get_optional_str, get_str
from contextunity.core.types import JsonDict, JsonValue

from contextunity.router.core import get_core_config
from contextunity.router.core.exceptions import RouterStorageError
from contextunity.router.core.interfaces import BaseProvider, IRead, IWrite
from contextunity.router.core.types import coerce_struct_data
from contextunity.router.modules.models import model_registry
from contextunity.router.modules.retrieval.rag.models import RetrievedDoc
from contextunity.router.modules.retrieval.rag.settings import get_rag_retrieval_settings

from .models import GraphNode
from .store import PostgresKnowledgeStore


def _flatten_keywords(metadata: JsonDict) -> str | None:
    """Merge ``keywords`` and ``keyphrase_texts`` from metadata into a single deduplicated string."""
    keywords = metadata.get("keywords")
    keyphrases = metadata.get("keyphrase_texts")
    parts: list[str] = []
    for raw in (keywords, keyphrases):
        if isinstance(raw, list):
            for item in raw:
                text = str(item).strip()
                if text:
                    parts.append(text)
    if not parts:
        return None
    seen: set[str] = set()
    uniq: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return " ".join(uniq)


class PostgresProvider(BaseProvider, IRead, IWrite):
    """ContextUnit facade over ``PostgresKnowledgeStore`` for hybrid search and ingestion."""

    _store: PostgresKnowledgeStore

    def __init__(self, *, store: PostgresKnowledgeStore | None = None) -> None:
        """Accept an existing store or create one from ``RouterConfig.postgres``."""
        cfg = get_core_config()
        if store is not None:
            self._store = store
        else:
            if not getattr(cfg, "postgres", None):
                raise ConfigurationError("Postgres config is missing from core config")
            self._store = PostgresKnowledgeStore(
                dsn=cfg.postgres.dsn,
                pool_min_size=cfg.postgres.pool_min_size,
                pool_max_size=cfg.postgres.pool_max_size,
            )

    @override
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: JsonDict | None = None,
    ) -> list[ContextUnit]:
        """Read."""
        cfg = get_core_config()
        rag_cfg = get_rag_retrieval_settings()

        tenant_raw = (filters or {}).get("tenant_id")
        if not isinstance(tenant_raw, str) or not tenant_raw.strip():
            from contextunity.core.exceptions import SecurityError

            raise SecurityError("tenant_id is required for Postgres retrieval")
        tenant_id = tenant_raw

        user_id: str | None = None
        user_raw = (filters or {}).get("user_id")
        if isinstance(user_raw, str) and user_raw.strip():
            user_id = user_raw

        source_types: list[str] | None = None
        if filters and (st := filters.get("source_type")):
            source_types = [str(st)]

        embeddings_key = rag_cfg.embeddings_model or cfg.models.default_embeddings
        embedder = model_registry.get_embeddings(embeddings_key, config=cfg)
        query_vec = await embedder.embed_query(query)

        candidate_k = max(rag_cfg.candidate_k, int(limit))
        results = await self._store.hybrid_search(
            query_text="" if not rag_cfg.enable_fts else query,
            query_vec=query_vec,
            candidate_k=candidate_k,
            limit=max(1, limit),
            scope=None,
            source_types=source_types,
            fusion=rag_cfg.hybrid_fusion,
            rrf_k=rag_cfg.rrf_k,
            vector_weight=rag_cfg.hybrid_vector_weight,
            text_weight=rag_cfg.hybrid_text_weight,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        units: list[ContextUnit] = []
        for res in results:
            doc = self._to_retrieved_doc(res.node, score=res.score)
            unit = ContextUnit(
                payload={"content": doc},
                provenance=["provider:postgres"],
            )
            units.append(unit)
        return units

    @override
    async def write(self, data: ContextUnit) -> None:
        """Upsert a ``RetrievedDoc`` (from ``payload.content``) as a graph node.

        Raises:
            SecurityError: If ``tenant_id`` is missing from the payload.
            ValueError: If the payload content is not a ``RetrievedDoc``.
        """
        payload = data.payload or {}
        content = payload.get("content")

        if isinstance(content, RetrievedDoc):
            doc = content
        elif isinstance(content, dict):
            doc = RetrievedDoc.model_validate(content)
        else:
            raise RouterStorageError(
                "PostgresProvider.write expects RetrievedDoc content in payload"
            )

        tenant_id = get_str(payload, "tenant_id")
        if not tenant_id:
            from contextunity.core.exceptions import SecurityError

            raise SecurityError("tenant_id is required for Postgres write")
        user_id = get_optional_str(payload, "user_id")

        node_id = str(payload.get("id", "")).strip()
        if not node_id:
            raise RouterStorageError("PostgresProvider.write requires payload.id")
        metadata = coerce_struct_data(doc.metadata or {})
        if not isinstance(metadata, dict):
            metadata = {}
        keywords_text = _flatten_keywords(metadata)
        node = GraphNode(
            id=node_id,
            content=str(doc.content or ""),
            node_kind="chunk",
            source_type=str(doc.source_type or "unknown"),
            source_id=str(doc.url or ""),
            title=doc.title,
            metadata=metadata,
            keywords_text=keywords_text,
            tenant_id=str(tenant_id),
            user_id=str(user_id) if user_id else None,
        )
        await self._store.upsert_graph([node], [], tenant_id=str(tenant_id), user_id=user_id)

    @override
    async def sink(self, unit: ContextUnit) -> None:
        """Sink."""
        await self.write(unit)
        return None

    def _to_retrieved_doc(self, node: GraphNode, *, score: float) -> RetrievedDoc:
        """Map a ``GraphNode`` and its search score to a ``RetrievedDoc``, extracting known fields from metadata."""
        metadata = coerce_struct_data(node.metadata or {})
        if not isinstance(metadata, dict):
            metadata = {}
        doc_data: dict[str, JsonValue] = {
            "source_type": node.source_type or "unknown",
            "content": node.content,
            "title": node.title,
            "metadata": metadata,
            "relevance": score,
        }
        for key in (
            "url",
            "snippet",
            "keywords",
            "summary",
            "quote",
            "book_title",
            "chapter",
            "chapter_number",
            "page_start",
            "page_end",
            "video_id",
            "video_url",
            "video_name",
            "timestamp",
            "timestamp_seconds",
            "session_title",
            "question",
            "answer",
            "filename",
            "description",
        ):
            if key in metadata:
                value = metadata[key]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    doc_data[key] = value
                elif isinstance(value, list):
                    doc_data[key] = [str(v) for v in value]
        return RetrievedDoc.model_validate(doc_data)


__all__ = ["PostgresProvider"]
