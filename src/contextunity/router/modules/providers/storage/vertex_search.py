"""Vertex AI Search — low-level async Discovery Engine client.

Pure storage I/O layer:
- gRPC client lifecycle (``_get_async_client``)
- Request construction (``_build_search_request``)
- Endpoint resolution (``_endpoint_for_location``)

Result parsing is delegated to ``modules/retrieval/rag/vertex_parser.py``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterable
from typing import Protocol

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ProviderError

from contextunity.router.modules.providers._async_iterate import async_iterate
from contextunity.router.modules.retrieval.rag.models import RetrievedDoc

logger = get_contextunit_logger(__name__)


class DiscoveryEngineSearchClient(Protocol):
    def search(self, request: object) -> AsyncIterable[object]: ...


def _safe_getattr(obj: object, name: str, default: object = "") -> object:
    return getattr(obj, name, default)


class _DiscoveryEngineSearchClientAdapter:
    """Adapter narrowing vendor Discovery Engine client at the import boundary."""

    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    async def search(self, request: object) -> AsyncIterable[object]:
        search_fn: object = _safe_getattr(self._inner, "search")
        if not callable(search_fn):
            raise ProviderError(
                "Discovery Engine client.search is not callable",
                code="VERTEX_SEARCH_IMPORT_ERROR",
            )
        search_response: object = search_fn(request)
        from contextunity.core.narrowing import await_object

        stream_source: object = await await_object(search_response)
        async for item in async_iterate(stream_source):
            yield item


class _SearchRequestFactory(Protocol):
    def __call__(
        self,
        *,
        serving_config: str,
        query: str,
        page_size: int,
        filter: str,
        content_search_spec: object,
    ) -> object: ...


class _ContentSearchSpecFactory(Protocol):
    def __call__(
        self,
        *,
        snippet_spec: object,
        extractive_content_spec: object,
    ) -> object: ...


class _NestedSpecFactory(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


_async_clients: dict[str, DiscoveryEngineSearchClient] = {}


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


def _load_search_client_factory() -> object:
    import importlib

    discoveryengine = importlib.import_module("google.cloud.discoveryengine_v1")
    client_factory_obj: object = getattr(discoveryengine, "SearchServiceAsyncClient", None)
    if not callable(client_factory_obj):
        raise ProviderError(
            "Discovery Engine SearchServiceAsyncClient is not callable",
            code="VERTEX_SEARCH_IMPORT_ERROR",
        )
    return client_factory_obj


def _get_async_client(*, api_endpoint: str) -> DiscoveryEngineSearchClient:
    """Get or create async Discovery Engine client."""
    global _async_clients
    ep = (api_endpoint or "").strip() or "discoveryengine.googleapis.com"
    if ep not in _async_clients:
        client_factory_obj = _load_search_client_factory()
        if not callable(client_factory_obj):
            raise ProviderError(
                "Discovery Engine SearchServiceAsyncClient is not callable",
                code="VERTEX_SEARCH_IMPORT_ERROR",
            )
        inner_client: object = client_factory_obj(client_options={"api_endpoint": ep})
        client: DiscoveryEngineSearchClient = _DiscoveryEngineSearchClientAdapter(inner_client)
        _async_clients[ep] = client
        logger.debug("Created async SearchServiceAsyncClient (endpoint=%s)", ep)
    return _async_clients[ep]


def _build_search_request(
    query: str,
    max_results: int,
    source_type_filter: str | None,
    serving_config: str,
) -> object:
    import importlib

    discoveryengine = importlib.import_module("google.cloud.discoveryengine_v1")

    filter_expr = f'source_type: ANY("{source_type_filter}")' if source_type_filter else ""

    search_request_cls_obj: object = getattr(discoveryengine, "SearchRequest", None)
    if not callable(search_request_cls_obj):
        raise ProviderError(
            "Discovery Engine SearchRequest is not callable",
            code="VERTEX_SEARCH_IMPORT_ERROR",
        )
    search_request_factory: _SearchRequestFactory = search_request_cls_obj

    content_search_spec_cls_obj: object = getattr(search_request_cls_obj, "ContentSearchSpec", None)
    if not callable(content_search_spec_cls_obj):
        raise ProviderError(
            "Discovery Engine ContentSearchSpec is not callable",
            code="VERTEX_SEARCH_IMPORT_ERROR",
        )
    content_search_spec_factory: _ContentSearchSpecFactory = content_search_spec_cls_obj

    snippet_spec_cls_obj: object = getattr(content_search_spec_cls_obj, "SnippetSpec", None)
    extractive_spec_cls_obj: object = getattr(
        content_search_spec_cls_obj, "ExtractiveContentSpec", None
    )
    if not callable(snippet_spec_cls_obj) or not callable(extractive_spec_cls_obj):
        raise ProviderError(
            "Discovery Engine nested search specs are not callable",
            code="VERTEX_SEARCH_IMPORT_ERROR",
        )
    snippet_factory: _NestedSpecFactory = snippet_spec_cls_obj
    extractive_factory: _NestedSpecFactory = extractive_spec_cls_obj

    content_spec = content_search_spec_factory(
        snippet_spec=snippet_factory(return_snippet=True),
        extractive_content_spec=extractive_factory(
            max_extractive_answer_count=1,
            max_extractive_segment_count=1,
        ),
    )

    return search_request_factory(
        serving_config=serving_config,
        query=query,
        page_size=max_results,
        filter=filter_expr,
        content_search_spec=content_spec,
    )


async def search_vertex_ai_async(
    *,
    query: str,
    max_results: int,
    source_type_filter: str | None = None,
    project_id: str,
    location: str = "global",
    data_store_id: str,
) -> list[RetrievedDoc]:
    """Execute a Discovery Engine search and return parsed results.

    Args:
        query: Search query text.
        max_results: Maximum number of results to return.
        source_type_filter: Optional source_type filter value.
        project_id: GCP project ID.
        location: Discovery Engine location (default: "global").
        data_store_id: Discovery Engine data store ID.

    Returns:
        List of ``RetrievedDoc`` objects (parsed by vertex_parser).
    """
    from contextunity.router.modules.retrieval.rag.vertex_parser import parse_search_result

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

    try:
        t0 = time.perf_counter()
        client = _get_async_client(api_endpoint=api_endpoint)
        request = _build_search_request(query, max_results, source_type_filter, serving_config)

        results: list[RetrievedDoc] = []
        async for result in client.search(request):
            if doc := parse_search_result(result):
                results.append(doc)
            if len(results) >= max_results:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        relevance_scores = [d.relevance for d in results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        max_relevance = max(relevance_scores) if relevance_scores else 0.0
        non_zero_count = sum(1 for r in relevance_scores if r > 0.0)

        logger.debug(
            (
                "Vertex AI Search COMPLETE: query=%r datastore=%s results=%d elapsed_ms=%.1f "
                "avg_relevance=%.3f max_relevance=%.3f non_zero_relevance=%d/%d"
            ),
            query[:80],
            data_store_id,
            len(results),
            elapsed_ms,
            avg_relevance,
            max_relevance,
            non_zero_count,
            len(relevance_scores),
        )
        return results
    except Exception as e:
        logger.exception("Vertex AI Search failed for query: %s", query[:50])
        raise ProviderError(
            "Vertex AI Search failed",
            code="VERTEX_SEARCH_ERROR",
        ) from e


__all__ = ["search_vertex_ai_async", "_endpoint_for_location"]
