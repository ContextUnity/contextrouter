"""Parallel LLM execution helpers for platform tools.

Generic ``asyncio.gather``-based runner for concurrent LLM sub-calls.
Used by ``sql_visualizer`` and available for future ``mode: "parallel"``
implementations.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypedDict

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from contextunity.router.cortex.types import GraphState, extract_message_content
from contextunity.router.modules.models.base import BaseLLM as LLMBaseModel

from .sql import TokenUsageDict, extract_json, invoke_model

logger = get_contextunit_logger(__name__)


def _coerce_component(value: object) -> ParsedComponent | None:
    """Coerce a raw object into a parsed component dict."""
    if not is_object_dict(value):
        return None
    component_type = value.get("type")
    if not isinstance(component_type, str):
        return None

    component: ParsedComponent = {"type": component_type}

    title = value.get("title")
    if isinstance(title, str):
        component["title"] = title

    content = value.get("content")
    if isinstance(content, str):
        component["content"] = content

    raw_rows = value.get("rows")
    if is_object_list(raw_rows):
        component["rows"] = [
            {str(key): item for key, item in row.items()} for row in raw_rows if is_object_dict(row)
        ]

    raw_columns = value.get("columns")
    if is_object_list(raw_columns):
        component["columns"] = [
            {str(key): str(item) for key, item in column.items()}
            for column in raw_columns
            if is_object_dict(column)
        ]

    raw_cards = value.get("cards")
    if is_object_list(raw_cards):
        component["cards"] = [
            {str(key): item for key, item in card.items()}
            for card in raw_cards
            if is_object_dict(card)
        ]

    echarts_option = value.get("echarts_option")
    if is_object_dict(echarts_option):
        component["echarts_option"] = {str(key): item for key, item in echarts_option.items()}

    raw_links = value.get("links")
    if is_object_list(raw_links):
        component["links"] = [
            {str(key): str(item) for key, item in link.items()}
            for link in raw_links
            if is_object_dict(link)
        ]

    return component


class ParsedComponent(TypedDict, total=False):
    """Component dict parsed from LLM JSON response.

    Every component has at minimum a ``type`` field.  Other fields are
    optional and depend on the component type (table, chart, text, etc.).
    """

    type: str  # text, table, bar_chart, line_chart, pie_chart, kpi_cards, conclusion
    title: str
    content: str  # markdown text content
    rows: list[dict[str, object]]
    columns: list[dict[str, str]]  # [{key, label, format?, link_template?}]
    cards: list[dict[str, object]]  # KPI cards [{label, value, format?}]
    echarts_option: dict[str, object]  # ECharts option for chart types
    links: list[dict[str, str]]  # link_list [{label, url}]


# Result of a single parallel sub-call: (parsed_components, token_usage)
ParallelCallResult = tuple[list[ParsedComponent], TokenUsageDict]
ModelInvokeFunc = Callable[..., Awaitable[tuple[AIMessage, TokenUsageDict]]]


async def run_parallel_llm_calls(
    llm: LLMBaseModel,
    tasks: dict[str, tuple[str, str | None]],
    data_context: str,
    config: RunnableConfig,
    *,
    invoke_model_fn: ModelInvokeFunc = invoke_model,
    state: GraphState | None = None,
    node_name: str | None = None,
) -> dict[str, ParallelCallResult]:
    """Run multiple LLM calls in parallel via asyncio.gather.

    Args:
        llm: Model instance (from model_registry).
        tasks: ``{name: (system_prompt, prompt_version)}`` mapping.
        data_context: Shared user context passed as HumanMessage.
        config: LangChain RunnableConfig for callbacks.

    Returns:
        ``{name: (parsed_components, usage)}`` for each successful call.
        Failed calls are logged and skipped.
    """

    async def _one_call(
        name: str, system_prompt: str, prompt_version: str | None
    ) -> tuple[str, list[ParsedComponent], TokenUsageDict]:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=data_context)]
        response, usage = await invoke_model_fn(
            llm,
            messages,
            config=config,
            prompt_version=prompt_version,
            node_name=node_name,
            state=state,
        )
        raw_content = extract_message_content(response)
        data = extract_json(raw_content)
        raw_comps = data.get("components", [])
        comps = (
            [
                component
                for raw_component in raw_comps
                if (component := _coerce_component(raw_component)) is not None
            ]
            if is_object_list(raw_comps)
            else []
        )
        # Fallback: LLM returned a raw component without {"components": [...]} wrapper
        if not comps:
            fallback = _coerce_component(data)
            if fallback is not None:
                comps = [fallback]
        return name, comps, usage

    coros = [_one_call(name, prompt, pv) for name, (prompt, pv) in tasks.items()]
    results_list = await asyncio.gather(*coros, return_exceptions=True)

    results: dict[str, ParallelCallResult] = {}
    for item in results_list:
        if isinstance(item, Exception):
            logger.warning("Parallel LLM sub-call failed: %s", item)
            continue
        if not isinstance(item, tuple):
            continue
        try:
            name, comps, usage = item
        except ValueError:
            logger.warning("Parallel LLM sub-call returned unexpected result shape")
            continue
        results[name] = (comps, usage)
    return results


__all__ = ["ParsedComponent", "ParallelCallResult", "run_parallel_llm_calls"]
