"""Builder for the SQL analytics graph.

Assembles nodes into the LangGraph pipeline:

    planner (anon→LLM→deanon) → execute_sql
    → verifier (anon→LLM→deanon) → visualizer (anon→LLM→deanon)
    → reflect → END

Each LLM node handles PII atomically: anonymize input, call LLM, deanonymize output.
No separate privacy nodes needed.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from contextrouter.cortex.graphs.sql_analytics.nodes import (
    make_execute_node,
    make_planner_node,
    make_reflect_node,
    make_verifier_node,
    make_visualizer_node,
)
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.tools import discover_all_tools

logger = logging.getLogger(__name__)


def build_sql_analytics_graph(config: dict) -> Any:
    """Build the SQL analytics graph based on registration config.

    Args:
        config: Graph configuration dict with keys:
            - tool_bindings: list of SQL tool names to try
            - model_key: default model key
            - max_retries: max retry attempts (default: 2)
            - pii_masking: whether to mask PII (default: False)
    """
    tool_names = config.get("tool_bindings", [])
    default_model_key = config.get("model_key")
    max_retries = config.get("max_retries", 2)
    pii_masking = config.get("pii_masking", False)

    # 1. Resolve prompts from project config
    planner_prompt = config.get("planner_prompt", "You are a helpful analyst.")
    verifier_prompt = config.get("verifier_prompt", "")
    visualizer_prompt = config.get("visualizer_prompt", "")
    visualizer_sub_prompts = config.get("visualizer_sub_prompts")

    # 2. Resolve tools
    all_tools = {t.name: t for t in discover_all_tools()}
    sql_tool = None
    for name in tool_names:
        if name in all_tools:
            sql_tool = all_tools[name]
            break

    # Resolve PII masking tools
    anonymize_tool = None
    deanonymize_tool = None
    if pii_masking:
        anonymize_tool = all_tools.get("anonymize_text")
        deanonymize_tool = all_tools.get("deanonymize_text")
        if not anonymize_tool or not deanonymize_tool:
            logger.warning("pii_masking=True but privacy tools not found — PII masking disabled")
            pii_masking = False

    if sql_tool:
        logger.info("SQL tool resolved: %s", sql_tool.name)
    elif tool_names:
        logger.warning("SQL Tool '%s' not found in registry", tool_names[0])

    logger.info(
        "Building sql_analytics graph: tools=%s, retries=%d, verifier=%s, visualizer=%s, pii=%s",
        tool_names,
        max_retries,
        bool(verifier_prompt),
        bool(visualizer_prompt),
        pii_masking,
    )

    # 3. Create nodes — each LLM node handles PII atomically
    planner = make_planner_node(
        planner_prompt=planner_prompt,
        default_model_key=default_model_key,
        pii_masking=pii_masking,
        anonymize_tool=anonymize_tool,
        deanonymize_tool=deanonymize_tool,
    )
    executor = make_execute_node(sql_tool=sql_tool, tool_names=tool_names)
    verifier = make_verifier_node(
        verifier_prompt=verifier_prompt,
        default_model_key=default_model_key,
        pii_masking=pii_masking,
        anonymize_tool=anonymize_tool,
        deanonymize_tool=deanonymize_tool,
    )
    visualizer = make_visualizer_node(
        visualizer_prompt=visualizer_prompt,
        visualizer_sub_prompts=visualizer_sub_prompts,
        default_model_key=default_model_key,
        pii_masking=pii_masking,
        anonymize_tool=anonymize_tool,
        deanonymize_tool=deanonymize_tool,
    )
    reflect = make_reflect_node()

    # 4. Graph structure
    workflow = StateGraph(SqlAnalyticsState)

    workflow.add_node("planner", planner)
    workflow.add_node("execute_sql", executor)
    workflow.add_node("verifier", verifier)
    workflow.add_node("visualizer", visualizer)
    workflow.add_node("reflect", reflect)

    workflow.set_entry_point("planner")

    # 5. Edges and routing
    def after_planner(state):
        if state.get("error") and not state.get("sql"):
            return "reflect"
        return "execute_sql"

    def after_execute(state):
        if state.get("error"):
            count = state.get("retry_count", 0)
            if count >= max_retries:
                sql_result = state.get("sql_result") or {}
                if isinstance(sql_result, dict) and sql_result.get("rows"):
                    return "visualizer"
                return "reflect"
            return "planner"
        return "verifier"

    def after_verifier(state):
        val = state.get("validation", {})
        if not val.get("valid", True):
            count = state.get("retry_count", 0)
            if count >= max_retries:
                return "visualizer"
            return "planner"
        return "visualizer"

    workflow.add_conditional_edges("planner", after_planner)
    workflow.add_conditional_edges("execute_sql", after_execute)
    workflow.add_conditional_edges("verifier", after_verifier)
    workflow.add_edge("visualizer", "reflect")
    workflow.add_edge("reflect", END)

    return workflow.compile()


__all__ = ["build_sql_analytics_graph"]
