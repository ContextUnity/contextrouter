"""Agent node — LLM reasoning with tool binding."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from contextunity.router.core import get_core_config
from contextunity.router.cortex.config_resolution import get_node_manifest_config
from contextunity.router.cortex.dispatcher_agent.prompts import SYSTEM_PROMPT
from contextunity.router.cortex.types import GraphState, StateUpdate, extract_message_content
from contextunity.router.modules.models import model_registry

logger = get_contextunit_logger(__name__)


@runtime_checkable
class _RunnableToolBinding(Protocol):
    """Bound-tool runnable surface used by dispatcher."""

    async def ainvoke(self, input: object, /, **kwargs: object) -> object: ...


@runtime_checkable
class _LLMWithToolBind(Protocol):
    """Runtime-checkable dispatcher subset for tool-capable LLMs."""

    def bind_tools(self, tools: Sequence[BaseTool], /) -> _RunnableToolBinding: ...


def _message_text(message: BaseMessage) -> str:
    """Normalize LangChain message content to plain text."""
    return extract_message_content(message)


def _resolve_system_prompt(state: GraphState) -> str:
    """Read the dispatcher system prompt from execution metadata when present."""
    metadata = state.get("metadata")
    system_prompt = metadata.get("system_prompt") if is_object_dict(metadata) else None
    if not isinstance(system_prompt, str) or not system_prompt:
        meta = state.get("meta")
        system_prompt = meta.get("system_prompt") if is_object_dict(meta) else None
    if isinstance(system_prompt, str) and system_prompt:
        return system_prompt
    return SYSTEM_PROMPT


def _coerce_ai_message(response: object) -> AIMessage:
    """Narrow model output to the ``AIMessage`` contract used by dispatcher state."""
    if isinstance(response, AIMessage):
        return response
    if isinstance(response, BaseMessage):
        return AIMessage(content=_message_text(response))
    return AIMessage(content=str(response))


async def agent_node(state: GraphState, config: RunnableConfig) -> StateUpdate:
    """LLM reasoning node — decides which tools to use."""
    _ = config
    # Record pipeline start time on first iteration
    updates: dict[str, object] = {}
    if not state.get("_start_ts"):
        updates["_start_ts"] = time.monotonic()

    node_config = get_node_manifest_config(state, "agent")
    core_config = get_core_config()
    model_name = node_config.get("model", core_config.models.default_llm)

    from contextunity.router.cortex.config_resolution import metadata_project_id

    llm = model_registry.create_llm(
        model_name,
        project_id=metadata_project_id(state),
        tenant_id=state.get("tenant_id", ""),
    )
    if not isinstance(llm, _LLMWithToolBind):
        logger.warning("Model '%s' does not support tool binding", model_name)
        return {
            **updates,
            "messages": [AIMessage(content="The selected model does not support tool usage.")],
            "iteration": state["iteration"] + 1,
            "error_detected": True,
        }

    # Self-healing trigger (stub — self_healing graph removed, will be reimplemented as compiled graph)
    if state.get("error_detected") and not state.get("healing_triggered"):
        logger.warning(
            "Self-healing requested but self_healing graph is not yet reimplemented. Skipping."
        )
        return {
            "messages": [
                AIMessage(
                    content="Self-healing is not yet available. Error detected but healing skipped."
                )
            ],
            "iteration": state["iteration"] + 1,
            "healing_triggered": True,
        }

    from contextunity.router.cortex.dispatcher_agent.tool_resolution import (
        dispatcher_tools_for_state,
    )

    tools = dispatcher_tools_for_state(state)
    allowed = state.get("allowed_tools", [])

    logger.info(
        "Dispatcher agent loaded %d tools (allowed_filter=%d entries)",
        len(tools),
        len(allowed),
    )

    if not tools:
        logger.warning("No tools available for dispatcher agent")
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
    system_prompt = _resolve_system_prompt(state)
    full_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    response = await llm_with_tools.ainvoke(full_messages)

    return {
        **updates,
        "messages": [_coerce_ai_message(response)],
        "iteration": state["iteration"] + 1,
    }


__all__ = ["agent_node"]
