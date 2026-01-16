"""Postgres provider (storage + retrieval)."""

from __future__ import annotations

from typing import Any

from contextrouter.core import get_core_config
from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseProvider, IRead, IWrite
from contextrouter.core.tokens import AccessManager, BiscuitToken
from contextrouter.core.types import coerce_struct_data
from contextrouter.modules.models import model_registry
from contextrouter.modules.retrieval.rag.models import RetrievedDoc
from contextrouter.modules.retrieval.rag.settings import get_rag_retrieval_settings

from .models import GraphNode
from .store import PostgresKnowledgeStore


def _flatten_keywords(metadata: dict[str, Any]) -> str | None:
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
    def __init__(self, *, store: PostgresKnowledgeStore | None = None) -> None:
        cfg = get_core_config()
        self._access = AccessManager.from_core_config()
        if store is not None:
            self._store = store
        else:
            if not getattr(cfg, "postgres", None):
                raise RuntimeError("Postgres config is missing from core config")
            self._store = PostgresKnowledgeStore(
                dsn=cfg.postgres.dsn,
                pool_min_size=cfg.postgres.pool_min_size,
                pool_max_size=cfg.postgres.pool_max_size,
            )

    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: BiscuitToken,
    ) -> list[BisquitEnvelope]:
        self._access.verify_read(token)
        cfg = get_core_config()
        rag_cfg = get_rag_retrieval_settings()

        tenant_id = (filters or {}).get("tenant_id")
        user_id = (filters or {}).get("user_id")
        if cfg.security.enabled and not tenant_id:
            raise PermissionError("tenant_id is required for Postgres retrieval")
        if not tenant_id:
            tenant_id = "public"

        source_types: list[str] | None = None
        if filters and (st := filters.get("source_type")):
            source_types = [str(st)]

        embeddings_key = rag_cfg.embeddings_model or cfg.models.default_embeddings
        embedder = model_registry.get_embeddings(embeddings_key, config=cfg)
        query_vec = await embedder.embed_query(query, token=token)

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
            tenant_id=str(tenant_id),
            user_id=str(user_id) if user_id else None,
        )
        envelopes: list[BisquitEnvelope] = []
        for res in results:
            doc = self._to_retrieved_doc(res.node, score=res.score)
            envelopes.append(BisquitEnvelope(content=doc).add_trace("provider:postgres"))
        return envelopes

    async def write(self, data: BisquitEnvelope, *, token: BiscuitToken) -> None:
        self._access.verify_envelope_write(data, token)
        cfg = get_core_config()
        content = data.content
        if isinstance(content, RetrievedDoc):
            doc = content
        elif isinstance(content, dict):
            doc = RetrievedDoc.model_validate(content)
        else:
            raise ValueError("PostgresProvider.write expects RetrievedDoc content")

        tenant_id = data.metadata.get("tenant_id") if isinstance(data.metadata, dict) else None
        if cfg.security.enabled and not tenant_id:
            raise PermissionError("tenant_id is required for Postgres write")
        if not tenant_id:
            tenant_id = "public"
        user_id = data.metadata.get("user_id") if isinstance(data.metadata, dict) else None

        node_id = str(data.id or "").strip()
        if not node_id:
            raise ValueError("PostgresProvider.write requires envelope.id")
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

    async def sink(self, envelope: BisquitEnvelope, *, token: BiscuitToken) -> Any:
        await self.write(envelope, token=token)
        return None

    def _to_retrieved_doc(self, node: GraphNode, *, score: float) -> RetrievedDoc:
        metadata = coerce_struct_data(node.metadata or {})
        if not isinstance(metadata, dict):
            metadata = {}
        doc_data = {
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
                doc_data[key] = metadata[key]
        return RetrievedDoc.model_validate(doc_data)


__all__ = ["PostgresProvider"]
