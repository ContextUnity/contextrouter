"""Agent node — LLM reasoning with tool binding."""

from __future__ import annotations

import logging
import time
from typing import Any

from contextrouter.cortex.graphs.dispatcher_agent.prompts import SYSTEM_PROMPT
from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState
from contextrouter.modules.models import model_registry
from contextrouter.modules.tools import discover_all_tools

logger = logging.getLogger(__name__)


async def agent_node(state: DispatcherState) -> dict[str, Any]:
    """LLM reasoning node — decides which tools to use."""
    # Record pipeline start time on first iteration
    updates: dict[str, Any] = {}
    if not state.get("_start_ts"):
        updates["_start_ts"] = time.monotonic()

    meta = state.get("metadata") or {}
    model_key = meta.get("model_key")
    llm = model_registry.get_llm_with_fallback(key=model_key)

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

    system_prompt = meta.get("system_prompt") or SYSTEM_PROMPT
    full_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    response = await llm_with_tools.ainvoke(full_messages)

    return {**updates, "messages": [response], "iteration": state["iteration"] + 1}


__all__ = ["agent_node"]
