"""Vertex AI Search low-level async client (storage provider I/O).

Per `.cursorrules`: Vertex Search infrastructure belongs under
`modules/providers/storage/` (NOT under retrieval logic).

This module preserves the existing parsing behavior (no logic loss).
"""

from __future__ import annotations

import logging
import time

from contextrouter.core import get_core_config
from contextrouter.core.exceptions import ProviderError
from contextrouter.core.types import coerce_struct_data
from contextrouter.modules.observability.langfuse import retrieval_span
from contextrouter.modules.retrieval.rag.models import RetrievedDoc
from contextrouter.modules.retrieval.rag.settings import get_effective_data_store_id

logger = logging.getLogger(__name__)

_async_clients: dict[str, object] = {}


def _endpoint_for_location(location: str) -> str:
    """Resolve Discovery Engine API endpoint for a given location.

    Discovery Engine requires regional endpoints for regional resources:
    - location="us"  -> "us-discoveryengine.googleapis.com"
    - location="eu"  -> "eu-discoveryengine.googleapis.com"
    - location="global" -> "discoveryengine.googleapis.com"
    """
    loc = (location or "").strip().lower()
    if not loc or loc == "global":
        return "discoveryengine.googleapis.com"
    return f"{loc}-discoveryengine.googleapis.com"


def _proto_to_dict(proto_obj: object) -> object:
    """Convert protobuf object to dict."""
    from typing import cast

    from google.protobuf.json_format import (
        MessageToDict,  # type: ignore[import-not-found]
    )

    if proto_obj is None:
        return None
    if hasattr(proto_obj, "DESCRIPTOR"):
        try:
            return MessageToDict(proto_obj, preserving_proto_field_name=True)
        except Exception as e:
            logger.debug("MessageToDict failed, trying fallback conversion: %s", e)
    if hasattr(proto_obj, "items"):
        try:
            items = cast("dict[object, object]", proto_obj).items()
        except Exception:
            items = []
        return {str(k): _proto_to_dict(v) for k, v in items}
    if hasattr(proto_obj, "__iter__") and not isinstance(proto_obj, (str, bytes)):
        try:
            it = cast("list[object]", proto_obj)
        except Exception:
            it = []
        return [_proto_to_dict(item) for item in it]
    return proto_obj


def _parse_search_result(result: object) -> RetrievedDoc | None:
    """Parse Discovery Engine search result to RetrievedDoc."""
    try:
        doc = getattr(result, "document", None)
        if doc is None:
            return None

        derived_obj = _proto_to_dict(getattr(doc, "derived_struct_data", None))
        struct_obj = _proto_to_dict(getattr(doc, "struct_data", None))
        derived_data: dict[str, object] = derived_obj if isinstance(derived_obj, dict) else {}
        struct_data: dict[str, object] = struct_obj if isinstance(struct_obj, dict) else {}
        doc_data: dict[str, object] = {**struct_data, **derived_data}

        content = ""
        ea = doc_data.get("extractive_answers")
        if isinstance(ea, list) and ea and isinstance(ea[0], dict):
            c = ea[0].get("content")
            content = c if isinstance(c, str) else ""
        else:
            snippets = doc_data.get("snippets")
            if isinstance(snippets, list) and snippets and isinstance(snippets[0], dict):
                s = snippets[0].get("snippet")
                content = s if isinstance(s, str) else ""

        if not content:
            c = doc_data.get("content")
            t = doc_data.get("text")
            if isinstance(c, str) and c:
                content = c
            elif isinstance(t, str) and t:
                content = t

        source_type = doc_data.get("source_type", "")
        source_type = source_type if isinstance(source_type, str) else ""
        if not source_type or source_type == "unknown":
            if (
                doc_data.get("book_title")
                or doc_data.get("chapter_title")
                or doc_data.get("page_start") is not None
            ):
                source_type = "book"
            elif (
                doc_data.get("video_id") or doc_data.get("video_url") or doc_data.get("video_name")
            ):
                source_type = "video"
            elif (
                doc_data.get("session_title") or doc_data.get("question") or doc_data.get("answer")
            ):
                source_type = "qa"
            else:
                source_type = "unknown"

        parsed = RetrievedDoc(
            content=content,
            source_type=source_type,
            relevance=float(getattr(result, "relevance_score", 0.0) or 0.0),
        )

        if source_type == "book":
            parsed.book_title = doc_data.get("book_title") or ""
            parsed.chapter = doc_data.get("chapter_title") or doc_data.get("chapter") or ""
            parsed.chapter_number = doc_data.get("chapter_number")
            parsed.page_start = doc_data.get("page_start")
            parsed.page_end = doc_data.get("page_end")
            parsed.keywords = doc_data.get("keywords") or []
            quote = doc_data.get("quote")
            if isinstance(quote, str) and quote.strip():
                parsed.quote = quote
                parsed.content = quote
        elif source_type == "video":
            parsed.video_id = doc_data.get("video_id") or ""
            parsed.video_url = doc_data.get("video_url") or ""
            parsed.title = doc_data.get("video_name") or doc_data.get("title") or ""
            parsed.timestamp = doc_data.get("timestamp") or ""
            parsed.timestamp_seconds = doc_data.get("timestamp_seconds")
            parsed.keywords = doc_data.get("keywords") or []
            quote = doc_data.get("quote")
            if isinstance(quote, str) and quote.strip():
                parsed.quote = quote
                parsed.content = quote
        elif source_type == "qa":
            parsed.session_title = doc_data.get("session_title") or ""
            parsed.question = doc_data.get("question") or ""
            parsed.answer = doc_data.get("answer") or ""
            parsed.keywords = doc_data.get("keywords") or []

        parsed.metadata = coerce_struct_data(doc_data) if isinstance(doc_data, dict) else {}
        return parsed
    except Exception:
        return None


def _get_async_client(*, api_endpoint: str):
    """Get or create async Discovery Engine client."""
    global _async_clients
    ep = (api_endpoint or "").strip() or "discoveryengine.googleapis.com"
    if ep not in _async_clients:
        from google.cloud import discoveryengine_v1 as discoveryengine

        _async_clients[ep] = discoveryengine.SearchServiceAsyncClient(
            client_options={"api_endpoint": ep}
        )
        logger.debug("Created async SearchServiceAsyncClient (endpoint=%s)", ep)
    return _async_clients[ep]


def _build_search_request(
    query: str,
    max_results: int,
    source_type_filter: str | None,
    serving_config: str,
):
    from google.cloud import discoveryengine_v1 as discoveryengine

    filter_expr = f'source_type: ANY("{source_type_filter}")' if source_type_filter else ""
    return discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=max_results,
        filter=filter_expr,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
            ),
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=1,  # Reduced from 3 for lower latency
                max_extractive_segment_count=1,  # Reduced from 3 for lower latency
            ),
        ),
    )


async def search_vertex_ai_async(
    *,
    query: str,
    max_results: int,
    source_type_filter: str | None = None,
) -> list[RetrievedDoc]:
    cfg = get_core_config()
    project_id = cfg.vertex.project_id
    # IMPORTANT: Discovery Engine location is often "global" even when Vertex LLM is regional.
    # Prefer explicit provider config names.
    location = (
        (getattr(cfg.vertex, "data_store_location", "") or "").strip()
        or (getattr(cfg.vertex, "discovery_engine_location", "") or "").strip()
        or "global"
    )
    if not project_id:
        logger.error("vertex.project_id must be set (TOML or env)")
        return []

    try:
        data_store_id = get_effective_data_store_id()
    except ValueError as e:
        logger.error("Failed to resolve RAG data_store_id: %s", e)
        return []

    # Discovery Engine resource name format:
    # projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}/servingConfigs/{servingConfig}
    serving_config = (
        f"projects/{project_id}/locations/{location}"
        f"/collections/default_collection"
        f"/dataStores/{data_store_id}/servingConfigs/default_search"
    )
    api_endpoint = _endpoint_for_location(location)

    logger.info(
        "Vertex AI Search START: query=%r datastore=%s location=%s endpoint=%s max_results=%d source_type=%s",
        query[:80],
        data_store_id,
        location,
        api_endpoint,
        max_results,
        source_type_filter or "all",
    )

    with retrieval_span(
        name="vertex_search",
        input_data={
            "query": query,
            "source_type": source_type_filter,
            "max_results": max_results,
            "location": location,
            "api_endpoint": api_endpoint,
            "data_store_id": data_store_id,
        },
    ) as span_ctx:
        try:
            t0 = time.perf_counter()
            client = _get_async_client(api_endpoint=api_endpoint)
            request = _build_search_request(query, max_results, source_type_filter, serving_config)

            response = await client.search(request)
            results: list[RetrievedDoc] = []
            async for result in response:
                if doc := _parse_search_result(result):
                    results.append(doc)
                if len(results) >= max_results:
                    break

            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Calculate relevance stats
            relevance_scores = [d.relevance for d in results if hasattr(d, "relevance")]
            avg_relevance = (
                sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            )
            max_relevance = max(relevance_scores) if relevance_scores else 0.0
            non_zero_count = sum(1 for r in relevance_scores if r > 0.0)

            logger.debug(
                "Vertex AI Search COMPLETE: query=%r datastore=%s results=%d elapsed_ms=%.1f "
                "avg_relevance=%.3f max_relevance=%.3f non_zero_relevance=%d/%d",
                query[:80],
                data_store_id,
                len(results),
                elapsed_ms,
                avg_relevance,
                max_relevance,
                non_zero_count,
                len(relevance_scores),
            )
            span_ctx["output"] = {"count": len(results), "elapsed_ms": elapsed_ms}
            return results
        except Exception as e:
            logger.exception("Vertex AI Search failed for query: %s", query[:50])
            raise ProviderError(
                f"Vertex AI Search failed: {str(e)}",
                code="VERTEX_SEARCH_ERROR",
                query=query[:50],
            ) from e


__all__ = ["search_vertex_ai_async"]
