"""Builder for the SQL analytics graph.

Assembles nodes into the LangGraph pipeline:

    planner (anon→LLM→deanon) → execute_sql
    → verifier (anon→LLM→deanon) → visualizer (anon→LLM→deanon)
    → reflect → END

Each LLM node handles PII atomically: anonymize input, call LLM, deanonymize output.
No separate privacy nodes needed.
"""

from __future__ import annotations

from typing import Any

from contextcore import get_context_unit_logger
from langgraph.graph import END, StateGraph

from contextrouter.cortex.graphs.config_resolution import get_node_attr, make_shield_path
from contextrouter.cortex.graphs.sql_analytics.nodes import (
    make_execute_node,
    make_planner_node,
    make_verifier_node,
    make_visualizer_node,
)
from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.tools import discover_all_tools

logger = get_context_unit_logger(__name__)


def build_sql_analytics_graph(config: dict) -> Any:
    """Build the SQL analytics graph based on registration config.

    Args:
        config: Graph configuration dict with keys:
            - tool_bindings: list of SQL tool names to try
            - model_key: default model key
            - max_retries: max retry attempts (default: 2)
            - pii_masking: whether to mask PII (default: False)
    """
    node_bindings = config.get("node_tool_bindings", {})

    # Consolidate all tool names required by any node to discover the master sql_tool
    tool_names_set = set(config.get("tool_bindings", []))
    for tools in node_bindings.values():
        tool_names_set.update(tools.keys())
    tool_names = list(tool_names_set)

    default_model_key = config.get("model_key")
    fallback_keys = config.get("fallback_keys")

    # Per-node model overrides (falls back to default)
    planner_model_key = get_node_attr(config, "planner", "model", default_model_key)
    verifier_model_key = get_node_attr(config, "verifier", "model", default_model_key)
    visualizer_model_key = get_node_attr(config, "visualizer", "model", default_model_key)

    # Compute Shield key from node's model_secret_ref, if it exists
    planner_secret = get_node_attr(config, "planner", "model_secret_ref")
    verifier_secret = get_node_attr(config, "verifier", "model_secret_ref")
    visualizer_secret = get_node_attr(config, "visualizer", "model_secret_ref")

    planner_shield_key = make_shield_path("planner") if planner_secret else None
    verifier_shield_key = make_shield_path("verifier") if verifier_secret else None
    visualizer_shield_key = make_shield_path("visualizer") if visualizer_secret else None

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
            raise RuntimeError(
                "pii_masking=True but privacy tools (anonymize_text, deanonymize_text) not found. Cannot guarantee PII protection."
            )
        else:
            # Check if Zero backend is reachable (resolved at config startup)
            from contextrouter.modules.tools.privacy_tools import _get_grpc_stub

            if _get_grpc_stub() is not None:
                logger.info("PII masking enabled (gRPC mode)")
            else:
                # Fallback: local mode — check if contextzero package is importable
                try:
                    import contextzero  # noqa: F401

                    logger.info("PII masking enabled (local mode)")
                except ImportError:
                    # Both gRPC and Local modes are unavailable
                    raise RuntimeError(
                        "pii_masking=True but ContextZero is unreachable "
                        "(no gRPC endpoint resolved and local contextzero package not installed). "
                        "Strict PII isolation cannot be guaranteed."
                    )

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

    if planner_model_key != default_model_key or visualizer_model_key != default_model_key:
        logger.info(
            "Per-node models: planner=%s, verifier=%s, visualizer=%s",
            planner_model_key,
            verifier_model_key,
            visualizer_model_key,
        )

    # 3. Create nodes — each LLM node handles PII atomically
    planner = make_planner_node(
        planner_prompt=planner_prompt,
        default_model_key=planner_model_key,
        fallback_keys=fallback_keys,
        shield_key_name=planner_shield_key,
        pii_masking=pii_masking,
        anonymize_tool=anonymize_tool,
        deanonymize_tool=deanonymize_tool,
    )
    executor = make_execute_node(sql_tool=sql_tool, tool_names=tool_names)
    verifier = make_verifier_node(
        verifier_prompt=verifier_prompt,
        default_model_key=verifier_model_key,
        fallback_keys=fallback_keys,
        shield_key_name=verifier_shield_key,
        pii_masking=pii_masking,
        anonymize_tool=anonymize_tool,
        deanonymize_tool=deanonymize_tool,
    )
    visualizer = make_visualizer_node(
        visualizer_prompt=visualizer_prompt,
        visualizer_sub_prompts=visualizer_sub_prompts,
        default_model_key=visualizer_model_key,
        fallback_keys=fallback_keys,
        shield_key_name=visualizer_shield_key,
        pii_masking=pii_masking,
        anonymize_tool=anonymize_tool,
        deanonymize_tool=deanonymize_tool,
    )

    # 4. Graph structure — wrap nodes in SecureNode for provenance + capability stripping
    from contextrouter.cortex.graphs.secure_node import make_secure_node

    workflow = StateGraph(SqlAnalyticsState)

    # PII tools (anonymize_text, deanonymize_text, check_pii) are NOT
    # added as execute_tools — they are internal Zero wrappers authorized
    # via zero: permission scopes (handled by pii_masking= in make_secure_node).

    def _get_tools(node_key: str, mode: str) -> list[str]:
        return [t for t, m in node_bindings.get(node_key, {}).items() if m == mode]

    workflow.add_node(
        "planner",
        make_secure_node(
            "planner",
            planner,
            pii_masking=pii_masking,
            model_secret_ref=planner_shield_key,
            prompt_signature=get_node_attr(config, "planner", "prompt_signature"),
            schema_tools=_get_tools("planner", "schema"),
            execute_tools=_get_tools("planner", "execute"),
        ),
    )
    workflow.add_node(
        "execute_sql",
        make_secure_node(
            "execute_sql",
            executor,
            requires_llm=False,
            schema_tools=_get_tools("tool_execution", "schema"),
            execute_tools=_get_tools("tool_execution", "execute"),
        ),
    )
    workflow.add_node(
        "verifier",
        make_secure_node(
            "verifier",
            verifier,
            pii_masking=pii_masking,
            model_secret_ref=verifier_shield_key,
            prompt_signature=get_node_attr(config, "verifier", "prompt_signature"),
            schema_tools=_get_tools("verifier", "schema"),
            execute_tools=_get_tools("verifier", "execute"),
        ),
    )
    workflow.add_node(
        "visualizer",
        make_secure_node(
            "visualizer",
            visualizer,
            pii_masking=pii_masking,
            model_secret_ref=visualizer_shield_key,
            prompt_signature=get_node_attr(config, "visualizer", "prompt_signature"),
            schema_tools=_get_tools("visualizer", "schema"),
            execute_tools=_get_tools("visualizer", "execute"),
        ),
    )

    workflow.set_entry_point("planner")

    # 5. Edges and routing
    def after_planner(state):
        if state.get("error") and not state.get("sql"):
            return END
        return "execute_sql"

    def after_execute(state):
        if state.get("error"):
            count = state.get("retry_count", 0)
            if count >= max_retries:
                sql_result = state.get("sql_result") or {}
                if isinstance(sql_result, dict) and sql_result.get("rows"):
                    return "visualizer"
                return END
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
    workflow.add_edge("visualizer", END)

    return workflow.compile()


__all__ = ["build_sql_analytics_graph"]
