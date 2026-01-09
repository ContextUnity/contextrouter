"""Vertex AI Ranking API integration.

Reranks documents using Google's Discovery Engine Ranking API.
Improves relevance when merging results from multiple retrieval queries.

Usage:
    from contextrouter.modules.retrieval.rag.ranking import rerank_documents

    ranked_docs = await rerank_documents(query="What is leadership?", documents=docs)
"""

from __future__ import annotations

import logging
import time

from contextrouter.core import get_core_config
from contextrouter.cortex import RetrievedDoc
from contextrouter.utils.retry import retry_with_backoff_async

from .settings import get_rag_retrieval_settings

logger = logging.getLogger(__name__)

_rank_client = None


def _get_rank_client():
    """Singleton async RankServiceAsyncClient (reuses gRPC channel)."""
    global _rank_client
    if _rank_client is None:
        from google.cloud import discoveryengine_v1 as discoveryengine

        _rank_client = discoveryengine.RankServiceAsyncClient()
    return _rank_client


def _is_reranking_enabled() -> bool:
    """Check if reranking is enabled via config (default: off)."""
    return get_rag_retrieval_settings().reranking_enabled


def _get_ranker_model() -> str:
    return get_rag_retrieval_settings().ranker_model


async def rerank_documents(
    query: str,
    documents: list[RetrievedDoc],
    *,
    top_n: int | None = None,
    source_type: str | None = None,
) -> list[RetrievedDoc]:
    """Rerank documents using Vertex AI Ranking API (async).

    Args:
        query: The user query to rank documents against.
        documents: List of documents to rerank. Each doc should have content.
        top_n: Maximum number of documents to return. If None, returns all.
        source_type: Optional filter to only rerank docs of this type.

    Returns:
        Documents sorted by relevance score (highest first), with score added.
        If reranking fails or is disabled, returns original documents unchanged.
    """
    from contextrouter.modules.observability.langfuse import retrieval_span

    if not _is_reranking_enabled():
        logger.debug("Reranking disabled, returning original documents")
        return documents[:top_n] if top_n else documents

    if not documents:
        return []

    if not query or not query.strip():
        return documents[:top_n] if top_n else documents

    # Filter by source_type if specified
    docs_to_rank = documents
    if source_type:
        docs_to_rank = [d for d in documents if d.source_type == source_type]
        if not docs_to_rank:
            return []

    # Vertex AI Ranking API limit: 1000 records per request
    MAX_RANK_RECORDS = 1000
    if len(docs_to_rank) > MAX_RANK_RECORDS:
        logger.warning(
            "Truncating %d docs to %d for reranking API limit", len(docs_to_rank), MAX_RANK_RECORDS
        )
        docs_to_rank = docs_to_rank[:MAX_RANK_RECORDS]

    with retrieval_span(
        name="rerank",
        input_data={"query": query, "source_type": source_type, "doc_count": len(docs_to_rank)},
    ) as span_ctx:
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine

            cfg = get_core_config()
            project_id = cfg.vertex.project_id

            if not project_id:
                logger.warning("vertex.project_id not set, skipping reranking")
                return docs_to_rank[:top_n] if top_n else docs_to_rank

            client = _get_rank_client()

            ranking_config = client.ranking_config_path(
                project=project_id,
                location="global",
                ranking_config="default_ranking_config",
            )

            # Build ranking records from documents
            records: list[discoveryengine.RankingRecord] = []
            doc_id_map: dict[str, RetrievedDoc] = {}

            for i, doc in enumerate(docs_to_rank):
                doc_id = f"doc_{i}"
                doc_id_map[doc_id] = doc

                # Extract title and content based on source_type
                st = doc.source_type
                title = ""
                content = ""

                if st == "book":
                    title = doc.book_title or doc.title or ""
                    if doc.chapter:
                        title = f"{title} - {doc.chapter}"
                    content = doc.content or ""
                elif st == "video":
                    title = doc.title or ""
                    content = doc.content or ""
                elif st == "qa":
                    title = doc.session_title or "Q&A"
                    content = (
                        f"Q: {doc.question}\nA: {doc.answer or doc.content}"
                        if doc.question
                        else (doc.answer or doc.content)
                    )
                elif st == "web":
                    title = doc.title or ""
                    content = doc.content or ""
                elif st == "knowledge":
                    # For knowledge we still use getattr for fields not in Pydantic model if they exist in extra
                    filename = getattr(doc, "filename", "")
                    description = getattr(doc, "description", "")
                    title = filename
                    content = f"{description}\n{doc.content or ''}"
                else:
                    title = doc.title or ""
                    content = doc.content or ""

                # Skip docs with no content
                if not content and not title:
                    continue

                records.append(
                    discoveryengine.RankingRecord(
                        id=doc_id,
                        title=title[:500] if title else "",
                        content=content[:1500] if content else "",
                    )
                )

            if not records:
                logger.warning("No valid records to rerank")
                return docs_to_rank[:top_n] if top_n else docs_to_rank

            t0 = time.perf_counter()

            request = discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model=_get_ranker_model(),
                top_n=top_n if top_n else len(records),
                query=query,
                records=records,
            )

            response = await retry_with_backoff_async(
                lambda: client.rank(request=request), attempts=3
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                "Reranking completed: %d docs -> %d ranked in %.1fms (source_type=%s)",
                len(records),
                len(response.records),
                elapsed_ms,
                source_type or "all",
            )

            # Build result list with scores
            ranked_docs: list[RetrievedDoc] = []
            for ranked_record in response.records:
                doc_id = ranked_record.id
                if doc_id not in doc_id_map:
                    continue
                doc = doc_id_map[doc_id]
                doc.relevance = ranked_record.score
                ranked_docs.append(doc)

            span_ctx["output"] = {"ranked_count": len(ranked_docs), "elapsed_ms": elapsed_ms}
            return ranked_docs

        except ImportError:
            logger.warning("google-cloud-discoveryengine not installed, skipping reranking")
            return docs_to_rank[:top_n] if top_n else docs_to_rank
        except Exception:
            logger.exception("Reranking failed, returning original documents")
            return docs_to_rank[:top_n] if top_n else docs_to_rank


__all__ = ["rerank_documents"]
