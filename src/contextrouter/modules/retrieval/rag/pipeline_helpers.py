"""Pipeline helper functions for post-processing and utilities."""

from __future__ import annotations

import hashlib
import logging
from typing import List

from contextrouter.cortex.services import get_graph_service
from contextrouter.cortex.state import AgentState

from .models import RetrievedDoc
from .settings import RagRetrievalSettings

logger = logging.getLogger(__name__)


def deduplicate_docs(docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    """Deduplicate retrieved documents by content hash."""
    seen: set[str] = set()
    out: List[RetrievedDoc] = []
    for d in docs:
        raw_key = f"{(d.url or '')}::{(d.snippet or '')}::{(d.content or '')}"
        key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def select_top_per_type(
    ranked: List[RetrievedDoc], cfg: RagRetrievalSettings
) -> List[RetrievedDoc]:
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

    out: List[RetrievedDoc] = []
    for _, docs in buckets.items():
        out.extend(docs)
    return out


def normalize_queries(state: AgentState, user_query: str) -> List[str]:
    """Normalize and limit retrieval queries."""
    queries = state.get("retrieval_queries") or []
    out = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
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


def get_graph_facts(state: AgentState) -> List[str]:
    """Get graph facts from taxonomy concepts in state."""
    concepts = state.get("taxonomy_concepts") or []
    if not concepts:
        logger.debug(
            "Graph facts SKIPPED: taxonomy_concepts is empty or missing in state. "
            "Available state keys: %s",
            list(state.keys()) if isinstance(state, dict) else "not a dict",
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


def coerce_doc_from_envelope(env: object) -> RetrievedDoc | None:
    """Convert ContextUnit.content into RetrievedDoc when possible."""
    try:
        content = getattr(env, "content", None)
    except Exception:
        content = None
    if isinstance(content, RetrievedDoc):
        return content
    if isinstance(content, dict):
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
