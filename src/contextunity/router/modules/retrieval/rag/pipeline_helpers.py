"""Pipeline helper functions for post-processing and utilities."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import ContextUnitPayload, is_object_dict, is_object_list

from contextunity.router.cortex.services import get_graph_service
from contextunity.router.cortex.types import GraphState

from .models import RetrievedDoc
from .settings import RagRetrievalSettings

logger = get_contextunit_logger(__name__)


def deduplicate_docs(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    """Deduplicate retrieved documents by content hash."""
    seen: set[str] = set()
    out: list[RetrievedDoc] = []
    for d in docs:
        raw_key = f"{(d.url or '')}::{(d.snippet or '')}::{(d.content or '')}"
        key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def select_top_per_type(
    ranked: list[RetrievedDoc], cfg: RagRetrievalSettings
) -> list[RetrievedDoc]:
    """Select top documents per source type based on config limits."""
    limits = {
        "book": cfg.max_books,
        "video": cfg.max_videos,
        "qa": cfg.max_qa,
        "knowledge": cfg.max_knowledge,
    }
    buckets: dict[str, list[RetrievedDoc]] = {k: [] for k in limits}
    for d in ranked:
        st = str(getattr(d, "source_type", "") or "")
        if st in buckets and len(buckets[st]) < limits.get(st, 0):
            buckets[st].append(d)

    out: list[RetrievedDoc] = []
    for _, docs in buckets.items():
        out.extend(docs)
    return out


def normalize_queries(state: GraphState, user_query: str) -> list[str]:
    """Normalize and limit retrieval queries."""
    dyn = state.get("dynamic", {})
    queries_raw: object = dyn.get("retrieval_queries") or state.get("retrieval_queries", []) or []
    queries = [q for q in queries_raw if isinstance(q, str)] if is_object_list(queries_raw) else []
    out = [q.strip() for q in queries if q.strip()]
    if not out:
        out = [user_query]
    return out[:3]


def get_type_limits(cfg: RagRetrievalSettings) -> dict[str, int]:
    """Resolve per-type limits for RAG retrieval.

    This is modular: new source types can be added via cfg.max_by_type.
    """
    if cfg.max_by_type:
        return {str(k): int(v) for k, v in cfg.max_by_type.items() if int(v) > 0}
    out = {
        "book": int(cfg.max_books),
        "video": int(cfg.max_videos),
        "qa": int(cfg.max_qa),
        "knowledge": int(cfg.max_knowledge),
    }
    return {k: v for k, v in out.items() if v > 0}


def get_graph_facts(state: GraphState) -> list[str]:
    """Get graph facts from taxonomy concepts in state."""
    concepts_raw: object = state.get("taxonomy_concepts") or []
    concepts = (
        [concept for concept in concepts_raw if isinstance(concept, str)]
        if is_object_list(concepts_raw)
        else []
    )
    if not concepts:
        logger.debug(
            "Graph facts SKIPPED: taxonomy_concepts is empty or missing in state. Available state keys: %s",
            list(state.keys()),
        )
        return []

    logger.debug(
        "Graph facts lookup START: concepts=%d concepts_list=%s",
        len(concepts),
        concepts[:5] if concepts else [],
    )

    service = get_graph_service()
    try:
        facts = service.get_facts(concepts)[:50]
        logger.debug(
            "Graph facts lookup COMPLETE: concepts=%d facts=%d",
            len(concepts),
            len(facts),
        )
        return facts
    except Exception:
        logger.exception("Graph facts lookup failed: concepts=%s", concepts[:5])
        return []


@runtime_checkable
class ContentEnvelope(Protocol):
    @property
    def content(self) -> object: ...


def _doc_from_payload(payload: ContextUnitPayload) -> RetrievedDoc | None:
    payload_copy: dict[str, object] = dict(payload)
    meta_raw = payload_copy.get("metadata")
    if is_object_dict(meta_raw):
        struct_raw = meta_raw.get("structData", meta_raw)
        if is_object_dict(struct_raw):
            for key, value in struct_raw.items():
                if key not in payload_copy:
                    payload_copy[key] = value
    try:
        return RetrievedDoc.model_validate(payload_copy)
    except Exception as e:
        logger.debug("Failed to validate RetrievedDoc from payload: %s", e)
    return None


def coerce_doc_from_envelope(env: object) -> RetrievedDoc | None:
    """Convert ContextUnit envelope into RetrievedDoc when possible."""
    if isinstance(env, RetrievedDoc):
        return env

    from contextunity.core import ContextUnit

    if isinstance(env, ContextUnit):
        return _doc_from_payload(env.payload)

    content = env.content if isinstance(env, ContentEnvelope) else None

    if isinstance(content, RetrievedDoc):
        return content
    if is_object_dict(content):
        try:
            return RetrievedDoc.model_validate(content)
        except Exception as e:
            logger.debug("Failed to validate RetrievedDoc from dict: %s", e)
    if isinstance(content, str) and content.strip():
        return RetrievedDoc(source_type="unknown", content=content)

    return None


__all__ = [
    "deduplicate_docs",
    "select_top_per_type",
    "normalize_queries",
    "get_type_limits",
    "get_graph_facts",
    "coerce_doc_from_envelope",
]
