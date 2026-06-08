"""Web search platform tool — internet search via Tavily/SerpAPI with result formatting."""

from __future__ import annotations

import importlib
import logging
import time
from typing import ClassVar, Literal, Protocol, runtime_checkable

from contextunity.core.types import JsonDict, is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.compiler.types import CompilerDataSourceSpec
from contextunity.router.cortex.types import GraphState, StateUpdate, extract_message_content
from contextunity.router.cortex.utils.pipeline import pipeline_log
from contextunity.router.modules.observability import retrieval_span

logger = logging.getLogger(__name__)


@runtime_checkable
class _GoogleSearchWrapper(Protocol):
    """Runtime surface used from the google search wrapper."""

    def results(self, *, query: str, num_results: int) -> list[dict[str, object]]: ...


@runtime_checkable
class _GoogleSearchWrapperFactory(Protocol):
    """Constructor surface for the google search wrapper."""

    def __call__(self, *, google_api_key: str, google_cse_id: str) -> _GoogleSearchWrapper: ...


class WebSearchConfig(BaseModel):
    """Configuration for web search data source."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["google", "serper", "duckduckgo"] = Field(
        default="google", description="Search provider"
    )
    max_results: int = Field(default=5, ge=1, le=20)
    search_kwargs: JsonDict = Field(default_factory=dict)
    output_mode: str = "direct"


def _data_sources_from_state(state: GraphState) -> list[CompilerDataSourceSpec]:
    """Extract configured data sources from graph state."""
    raw_state_config: object = state.get("config")
    if not is_object_dict(raw_state_config):
        return []
    raw_sources: object = raw_state_config.get("data_sources", [])
    if not is_object_list(raw_sources):
        return []

    sources: list[CompilerDataSourceSpec] = []
    for raw_source in raw_sources:
        if not is_object_dict(raw_source):
            continue
        source: CompilerDataSourceSpec = {}
        source_type = raw_source.get("type")
        if source_type == "vector":
            source["type"] = "vector"
        elif source_type == "sql":
            source["type"] = "sql"
        elif source_type == "federated":
            source["type"] = "federated"
        elif source_type == "web":
            source["type"] = "web"
        binding = raw_source.get("binding")
        if isinstance(binding, str):
            source["binding"] = binding
        provider = raw_source.get("provider")
        if isinstance(provider, str):
            source["provider"] = provider
        max_results = raw_source.get("max_results")
        if isinstance(max_results, int):
            source["max_results"] = max_results
        search_kwargs = raw_source.get("search_kwargs")
        if is_object_dict(search_kwargs):
            source["search_kwargs"] = {str(key): value for key, value in search_kwargs.items()}
        if source:
            sources.append(source)
    return sources


def _as_object_dict(value: object) -> dict[str, object]:
    """Return a string-keyed object dict when possible."""
    if not is_object_dict(value):
        return {}
    return {str(key): item for key, item in value.items()}


async def perform_web_search(state: GraphState) -> StateUpdate:
    """Execute a web search using the provided configuration."""
    from contextunity.router.cortex.config_resolution import get_node_manifest_config

    node_config = get_node_manifest_config(state, "web_search")

    # If called via fanout, __source_binding__ dictates the concrete configured data source.
    source_binding = state.get("__source_binding__")
    dynamic_cfg = dict(node_config)

    if source_binding:
        for ds in _data_sources_from_state(state):
            if ds.get("binding") == source_binding:
                if "provider" in ds:
                    dynamic_cfg["provider"] = ds["provider"]
                if "max_results" in ds:
                    dynamic_cfg["max_results"] = ds["max_results"]
                if "search_kwargs" in ds:
                    dynamic_cfg["search_kwargs"] = ds["search_kwargs"]
                break

    try:
        validated_cfg = WebSearchConfig.model_validate(dynamic_cfg)
        provider = validated_cfg.provider
        max_results = validated_cfg.max_results
    except Exception as e:  # graceful-degrade: tool failure returns empty result
        from contextunity.core.exceptions import ConfigurationError

        raise ConfigurationError(
            message=f"Invalid web search configuration for binding '{source_binding}': {str(e)}",
            component="router",
        ) from e

    intermediate = state.get("intermediate_results", {})
    detect_intent = _as_object_dict(intermediate.get("detect_intent"))
    raw_queries = detect_intent.get("retrieval_queries", [])
    raw_queries_list = raw_queries if is_object_list(raw_queries) else []
    queries = [q for q in raw_queries_list if isinstance(q, str) and q.strip()]
    if not queries:
        messages = state.get("messages", [])
        if messages:
            queries = [extract_message_content(messages[-1])]

    with retrieval_span(name=f"web_search_{provider}", input_data={"queries": queries}):
        t0 = time.perf_counter()

        results: list[str] = []
        if provider == "google":
            from contextunity.router.core import get_core_config

            cfg = get_core_config()
            if not cfg.google_cse.enabled or not cfg.google_cse.api_key or not cfg.google_cse.cx:
                logger.error("Google CSE is not enabled or lacks credentials in CoreConfig.")
                results = ["Search Error: Google provider not configured."]
            else:
                try:
                    search_mod = importlib.import_module("langchain_google_community.search")
                    wrapper_factory = getattr(search_mod, "GoogleSearchAPIWrapper", None)
                    if not isinstance(wrapper_factory, _GoogleSearchWrapperFactory):
                        raise RuntimeError("GoogleSearchAPIWrapper is unavailable")

                    wrapper = wrapper_factory(
                        google_api_key=cfg.google_cse.api_key,
                        google_cse_id=cfg.google_cse.cx,
                    )
                    for q in queries:
                        res = wrapper.results(query=q, num_results=max_results)
                        if is_object_list(res):
                            snippets = [
                                str(result.get("snippet", ""))
                                for result in res
                                if is_object_dict(result)
                            ]
                            results.append(f"🔍 Google [{q}]:\n" + "\n".join(snippets))
                except Exception as e:  # graceful-degrade: tool failure returns empty result
                    logger.exception("Google CSE failed during web_search platform tool")
                    results.append(f"Search Error: {e}")
        elif provider == "serper":
            results = [f"🔍 SERPER stub search results for '{q}': NotImplemented" for q in queries]
        elif provider == "duckduckgo":
            results = [f"🔍 DDG stub search results for '{q}': NotImplemented" for q in queries]
        else:
            results = [f"🔍 Unknown provider '{provider}' for '{q}'" for q in queries]

        formatted_result = "\n".join(results)

        pipeline_log(
            "web_search.out",
            provider=provider,
            query_count=len(queries),
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )

        route = node_config.get("next_node") or "synthesize"

        return {
            "intermediate_results": {
                "web_search_results": formatted_result,
                "fanout_outputs": [formatted_result],
            },
            "web_route": route,
        }
