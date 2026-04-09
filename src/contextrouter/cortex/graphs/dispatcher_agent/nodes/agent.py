"""Agent node — LLM reasoning with tool binding."""

from __future__ import annotations

import time
from typing import Any

from contextcore import get_context_unit_logger

from contextrouter.core import get_core_config
from contextrouter.cortex.graphs.config_resolution import get_node_manifest_config
from contextrouter.cortex.graphs.dispatcher_agent.prompts import SYSTEM_PROMPT
from contextrouter.cortex.graphs.dispatcher_agent.state import (
    DispatcherState,
    DispatcherStateUpdate,
)
from contextrouter.modules.models import model_registry
from contextrouter.modules.tools import discover_all_tools

logger = get_context_unit_logger(__name__)


async def agent_node(state: DispatcherState) -> DispatcherStateUpdate:
    """LLM reasoning node — decides which tools to use."""
    # Record pipeline start time on first iteration
    updates: dict[str, Any] = {}
    if not state.get("_start_ts"):
        updates["_start_ts"] = time.monotonic()

    node_config = get_node_manifest_config(state, "agent")
    config = get_core_config()
    model_name = node_config.get("model", config.models.default_llm)

    llm = model_registry.create_llm(model_name)

    # Self-healing trigger
    if state.get("error_detected") and not state.get("healing_triggered"):
        from contextrouter.cortex.graphs.self_healing import build_self_healing_graph

        healing_graph = build_self_healing_graph()
        healing_result = await healing_graph.ainvoke({})

        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
                    content=f"Self-healing triggered. Result: {healing_result.get('healing_report', {})}"
                )
            ],
            "iteration": state["iteration"] + 1,
            "healing_triggered": True,
        }

    # Discover and optionally filter tools
    tools = discover_all_tools()
    allowed = state.get("allowed_tools", [])
    if allowed and "*" not in allowed:
        allowed_set = set(allowed)
        tools = [t for t in tools if t.name in allowed_set]

    logger.info(
        "Dispatcher agent loaded %d tools (allowed_filter=%d entries)",
        len(tools),
        len(allowed),
    )

    if not tools:
        logger.warning("No tools available for dispatcher agent")
        from langchain_core.messages import AIMessage

        return {
            **updates,
            "messages": [
                AIMessage(
                    content="No tools are currently available. Please check tool registration."
                )
            ],
            "iteration": state["iteration"] + 1,
        }

    llm_with_tools = llm.bind_tools(tools)

    from langchain_core.messages import SystemMessage

    system_prompt = (state.get("meta") or {}).get("system_prompt") or SYSTEM_PROMPT
    full_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    response = await llm_with_tools.ainvoke(full_messages)

    return {**updates, "messages": [response], "iteration": state["iteration"] + 1}


__all__ = ["agent_node"]
