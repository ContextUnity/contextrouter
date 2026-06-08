"""Agent Node Executor for compiled graphs.
Runs an LLM with an explicit allowlist of LangChain tools. Tool execution
uses the normal SecureTool path, so ContextToken permissions and tenant
binding remain enforced by the existing tool security layer.
"""

from __future__ import annotations

from typing import Protocol, TypeGuard

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.manifest.helpers import parse_tool_ref as _parse_tool_ref
from contextunity.core.types import is_json_dict, is_object_dict, is_object_list
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig

from contextunity.router.core.exceptions import RouterLLMError

from ...compiler.node_config import NodeConfig
from ...compiler.types import CompilerNodeSpec, ProjectManifest
from ...types import GraphState, NodeFunc, StateUpdate

logger = get_contextunit_logger(__name__)

DEFAULT_AGENT_GOAL = (
    "Help the user accomplish their task using the approved tools available to this agent. "
    "Prefer concrete, verifiable progress and clearly report blockers."
)


class _ToolBoundChatModel(Protocol):
    """Bound chat model surface used by the agent executor."""

    async def ainvoke(self, input: list[BaseMessage]) -> BaseMessage: ...


class _ToolBindingChatModel(Protocol):
    """Chat model surface that can bind tools."""

    def bind_tools(self, tools: list[object], **kwargs: object) -> _ToolBoundChatModel: ...


def _is_tool_binding_chat_model(value: object) -> TypeGuard[_ToolBindingChatModel]:
    """Runtime guard for chat models that expose bind_tools()."""
    return callable(getattr(value, "bind_tools", None))


def _tool_calls_from_message(message: object) -> list[object]:
    """Return tool call payloads from a LangChain-style assistant message."""
    raw = getattr(message, "tool_calls", None)
    return list(raw) if is_object_list(raw) else []


def _message_from_item(item: object) -> BaseMessage:
    """Coerce an arbitrary item into a LangChain BaseMessage.

    Handles three input shapes:
    - Already a ``BaseMessage`` → returned as-is.
    - A ``dict`` with ``role`` / ``content`` keys → ``SystemMessage`` or ``HumanMessage``.
    - Anything else → stringified into a ``HumanMessage``.

    Args:
        item: Raw message object from graph state.

    Returns:
        A LangChain BaseMessage suitable for LLM invocation.
    """
    if isinstance(item, BaseMessage):
        return item
    if is_object_dict(item):
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        if role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)
    return HumanMessage(content=str(item))


def _messages_from_state(state: GraphState, input_key: str) -> list[BaseMessage]:
    """Extract and coerce messages from a state field into BaseMessage list.

    Reads ``state[input_key]`` and normalizes it:
    - ``list`` → each item coerced via ``_message_from_item``.
    - Non-empty scalar → wrapped in a single-element list.
    - Missing / empty → a single empty ``HumanMessage`` (keeps LLM happy).

    Args:
        state: Current graph execution state.
        input_key: State field name containing raw messages.

    Returns:
        Ordered list of LangChain messages ready for LLM invocation.
    """
    raw_messages = state.get(input_key, [])
    if is_object_list(raw_messages):
        return [_message_from_item(item) for item in raw_messages]
    if raw_messages:
        return [_message_from_item(raw_messages)]
    return [HumanMessage(content="")]


def _resolve_allowed_tool_names(
    node_spec: CompilerNodeSpec, graph_config: ProjectManifest
) -> list[str]:
    """Build the deduplicated allowlist of tool names for an agent node.

    Each ``tools`` ref is parsed via ``parse_tool_ref``; federated refs are
    resolved through the manifest ``federated_tool_map``.

    Args:
        node_spec: Compiled node specification containing explicit ``tools``.
        graph_config: Project manifest with ``federated_tool_map``.

    Returns:
        Ordered list of unique tool names the agent is permitted to call.
    """
    resolved_raw = graph_config.get("federated_tool_map", {})
    resolved_federated: dict[str, str] = {}
    if is_object_dict(resolved_raw):
        for key, value in resolved_raw.items():
            if isinstance(value, str):
                resolved_federated[str(key)] = value

    names: list[str] = []
    for ref in node_spec.get("tools") or []:
        kind, name = _parse_tool_ref(ref)
        if kind == "federated":
            name = resolved_federated.get(name, name)
        if name and name not in names:
            names.append(name)

    if node_spec.get("toolkits"):
        raise ConfigurationError(
            "Agent node toolkits must be expanded to explicit tools by the SDK bundle generator"
        )

    return names


def _resolve_goal_prompt(
    node_spec: CompilerNodeSpec,
    node_config: dict[str, object],
    manifest_config: ProjectManifest,
) -> str:
    """Resolve goal instructions for agent mode using a cascade.

    Priority (first non-empty wins):
    ``node_spec["goal"]`` → ``node_config["goal"]`` →
    ``manifest_config["goal"]`` → ``manifest_config["config"]["goal"]`` →
    ``DEFAULT_AGENT_GOAL``.

    Args:
        node_spec: Compiled node specification.
        node_config: Validated per-node configuration dict.
        manifest_config: Full project manifest.

    Returns:
        Non-empty goal prompt string.
    """
    inner_config = manifest_config.get("config", manifest_config)

    candidates = (
        node_spec.get("goal"),
        node_config.get("goal"),
        manifest_config.get("goal"),
        inner_config.get("goal"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return DEFAULT_AGENT_GOAL


def make_agent_node(node_spec: CompilerNodeSpec, manifest_config: ProjectManifest) -> NodeFunc:
    """Create a LangGraph node that runs an LLM with explicit tool-calling.

    The returned closure executes a multi-turn tool-calling loop:
    1. Resolve the tool allowlist and bind tools to the LLM.
    2. Inject goal + system prompt + optional ``prompt_ref`` extension.
    3. Invoke the LLM and dispatch each ``tool_call`` via SecureTool.
    4. Feed ``ToolMessage`` results back until no more calls or
       ``max_tool_calls`` is reached.

    Args:
        node_spec: Compiled node specification (model, tools, goal, config).
        manifest_config: Full project manifest for prompt / tool resolution.

    Returns:
        Async ``NodeFunc`` closure suitable for LangGraph node registration.

    Raises:
        ConfigurationError: If the node has no resolved tools.
        RouterLLMError: If the LLM cannot be instantiated or invoked.
    """
    node_name = node_spec.get("name", "unnamed_agent")
    model_name = node_spec.get("model", "")
    prompt_ref = node_spec.get("prompt_ref")
    _raw_cfg = node_spec.get("config")
    _cfg_dict = dict(_raw_cfg) if is_json_dict(_raw_cfg) else {}
    nc = NodeConfig.model_validate(_cfg_dict)
    node_cfg_map: dict[str, object] = nc.model_dump(mode="python", exclude_none=False)
    input_key = nc.state_input_key or "messages"
    output_key = nc.state_output_key or "final_output"
    tool_choice = nc.tool_choice
    max_tool_calls = int(nc.max_tool_calls if nc.max_tool_calls is not None else 5)
    max_tool_calls = min(max(max_tool_calls, 0), 20)

    allowed_tool_names = _resolve_allowed_tool_names(node_spec, manifest_config)
    if not allowed_tool_names:
        raise ConfigurationError(
            message=f"Agent node '{node_name}' requires non-empty tools or toolkits.",
            node_name=node_name,
        )

    async def agent_executor(state: GraphState, config: RunnableConfig) -> StateUpdate:
        """Run the multi-turn tool-calling loop for this agent node.

        Args:
            state: Graph execution state with ``__token__`` and messages.
            config: LangGraph runnable config carrying callback handlers.

        Returns:
            State update with ``output_key`` content and accumulated messages.

        Raises:
            ConfigurationError: If referenced tools are unavailable.
            RouterLLMError: If the LLM cannot be created or is not a
                ``BaseChatModel``.
        """
        from contextunity.router.cortex.config_resolution import metadata_project_id
        from contextunity.router.modules.models import model_registry
        from contextunity.router.modules.tools import discover_tools_for_project

        try:
            from contextunity.router.cortex.config_resolution import metadata_project_id

            llm = model_registry.create_llm(
                model_name,
                project_id=metadata_project_id(state),
                tenant_id=state.get("tenant_id", ""),
            )
        except Exception as exc:  # wraps-to-domain: re-raises as typed exception
            raise RouterLLMError(
                message=f"Agent node '{node_name}' failed to create LLM '{model_name}': {exc}",
                node_name=node_name,
                model=model_name,
            ) from exc

        available = {
            tool.name: tool for tool in discover_tools_for_project(metadata_project_id(state))
        }
        selected_tools = [available[name] for name in allowed_tool_names if name in available]
        missing = sorted(set(allowed_tool_names) - {tool.name for tool in selected_tools})
        if missing:
            raise ConfigurationError(
                message=f"Agent node '{node_name}' references unavailable tools: {missing}",
                node_name=node_name,
            )

        messages = _messages_from_state(state, input_key)

        # Track delta messages to return (for add_messages reducer compatibility).
        # LangGraph's add_messages reducer appends returned messages to existing state.
        # Returning the full list causes exponential duplication.
        new_messages: list[BaseMessage] = []

        # ── Goal injection: agent mode uses task goal, not persona identity ──
        goal_prompt = _resolve_goal_prompt(node_spec, node_cfg_map, manifest_config)
        system_sections = [f"Goal:\n{goal_prompt}"]

        explicit_system_prompt = nc.system_prompt
        if explicit_system_prompt:
            system_sections.append(str(explicit_system_prompt))

        system_prompt = "\n\n".join(system_sections)

        # prompt_ref extends the base (goal or explicit)
        if prompt_ref:
            inner_config = manifest_config.get("config", manifest_config)
            ref_prompt = inner_config.get(f"{node_name}_prompt")
            if ref_prompt:
                ref_prompt_str = str(ref_prompt)
                system_prompt = (
                    f"{system_prompt}\n\n{ref_prompt_str}" if system_prompt else ref_prompt_str
                )

        if system_prompt:
            system_msg = SystemMessage(content=system_prompt)
            messages = [system_msg, *messages]
            new_messages.append(system_msg)

        from langchain_core.language_models.chat_models import BaseChatModel

        if not isinstance(llm, BaseChatModel):
            raise RouterLLMError(
                message=f"Agent node '{node_name}' requires a LangChain BaseChatModel, got {type(llm).__name__}",
                node_name=node_name,
                model=model_name,
            )
        if not _is_tool_binding_chat_model(llm):
            raise RouterLLMError(
                message=f"Agent node '{node_name}' requires a chat model with bind_tools().",
                node_name=node_name,
                model=model_name,
            )

        bind_kwargs: dict[str, str] = {}
        if isinstance(tool_choice, str) and tool_choice:
            bind_kwargs["tool_choice"] = tool_choice
        tools_for_bind: list[object] = list(selected_tools)
        llm_with_tools = llm.bind_tools(tools_for_bind, **bind_kwargs)
        tool_map = {tool.name: tool for tool in selected_tools}
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        new_messages.append(response)

        tool_call_count = 0
        tool_calls = _tool_calls_from_message(response)
        while tool_calls and tool_call_count < max_tool_calls:
            executed_this_pass = 0
            for tool_call_obj in tool_calls:
                if not is_json_dict(tool_call_obj):
                    call_id = "unknown"
                    tool_call_count += 1
                    executed_this_pass += 1
                    err_msg = ToolMessage(
                        tool_call_id=call_id,
                        content="Error: malformed tool call payload (expected JSON object).",
                    )
                    messages.append(err_msg)
                    new_messages.append(err_msg)
                    continue
                tool_name_obj = tool_call_obj.get("name") or tool_call_obj.get("tool")
                tool_name = tool_name_obj if isinstance(tool_name_obj, str) else None
                call_id_obj = tool_call_obj.get("id", tool_name or "unknown")
                call_id = call_id_obj if isinstance(call_id_obj, str) else str(call_id_obj)
                if not isinstance(tool_name, str) or tool_name not in tool_map:
                    logger.warning(
                        "Agent '%s': unknown tool call '%s' (allowed: %s)",
                        node_name,
                        tool_name,
                        sorted(tool_map.keys()),
                    )
                    # Return error ToolMessage so LLM can self-correct
                    err_msg = ToolMessage(
                        tool_call_id=call_id,
                        content=(
                            f"Error: tool '{tool_name}' is not available. "
                            f"Available tools: {sorted(tool_map.keys())}"
                        ),
                    )
                    messages.append(err_msg)
                    new_messages.append(err_msg)
                    tool_call_count += 1
                    executed_this_pass += 1
                    continue
                tool_args_raw = tool_call_obj.get("args")
                tool_args = tool_args_raw if is_object_dict(tool_args_raw) else {}
                from contextunity.router.langchain_boundaries import invoke_tool_arun

                result = await invoke_tool_arun(tool_map[tool_name], tool_args, config=config)
                tool_msg = ToolMessage(
                    tool_call_id=call_id,
                    content=str(result),
                )
                messages.append(tool_msg)
                new_messages.append(tool_msg)
                tool_call_count += 1
                executed_this_pass += 1
                if tool_call_count >= max_tool_calls:
                    break
            if executed_this_pass == 0:
                break
            if tool_call_count >= max_tool_calls:
                break
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            new_messages.append(response)
            tool_calls = _tool_calls_from_message(response)

        return {
            str(output_key): getattr(response, "content", response),
            "messages": new_messages,
        }

    return agent_executor


__all__ = ["make_agent_node"]
