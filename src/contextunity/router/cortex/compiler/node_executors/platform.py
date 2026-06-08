"""Platform Node Executor for the Graph Compiler.
Creates a LangGraph node function that dispatches to Brain/Shield/Worker/Privacy
platform services via PlatformToolRegistry. Registry-based dispatch replaces
the Phase 1 stubs.
Dispatch flow:
1. Lookup tool_binding in PlatformToolRegistry → get (executor, schema, scopes)
2. Validate config against Pydantic schema
3. Check token scopes → SecurityError if missing
4. Execute tool with validated config
5. Route output to state_output_key
"""

from __future__ import annotations

import asyncio

from contextunity.core import get_contextunit_logger
from contextunity.core.manifest.helpers import parse_tool_ref
from contextunity.core.types import is_json_dict, is_object_dict
from langchain_core.runnables import RunnableConfig

from contextunity.router.cortex.compiler.node_executors.telemetry import tool_telemetry

from ...compiler.types import CompilerNodeSpec, NodeMeta, ProjectManifest
from ...types import GraphState, NodeFunc, StateUpdate

logger = get_contextunit_logger(__name__)

_registry_initialized = False


def _get_platform_registry():
    """Get the module-level PlatformToolRegistry singleton.

    Lazy-initializes platform tools on first access. This ensures all
    platform tools (brain_*, shield_*, worker_*, router_*, privacy_*) are
    registered before any platform node attempts to use them.

    Separate function enables test patching without import-time side effects.
    """
    global _registry_initialized

    from contextunity.router.cortex.compiler.platform_registry import (
        platform_registry,
    )

    if not _registry_initialized:
        from contextunity.router.cortex.compiler.platform_tools import (
            register_all_platform_tools,
        )

        register_all_platform_tools(platform_registry)
        _registry_initialized = True

    return platform_registry


def get_platform_required_scopes(tool_name: str) -> list[str]:
    """Return the ContextToken scopes required to invoke a platform tool.

    Delegates to the ``PlatformToolRegistry`` singleton, which holds
    per-tool scope declarations from registration time.

    Args:
        tool_name: Registered platform tool identifier.

    Returns:
        List of required scope strings (e.g., ``["brain:read"]``).
    """
    registry = _get_platform_registry()
    return list(registry.get(tool_name).required_scopes)


def make_platform_node(node_spec: CompilerNodeSpec, config: ProjectManifest) -> NodeFunc:
    """Create a LangGraph node for platform service dispatch.

    Dispatch flow:
    1. Lookup ``tool_binding`` in ``PlatformToolRegistry``.
    2. Validate ContextToken scopes against tool requirements.
    3. Validate ``tool_config`` against the tool’s Pydantic schema.
    4. Execute with ``tool_telemetry`` wrapper for observability.
    5. Route output: ``direct`` mode passes dict fields through;
       ``wrapped`` mode (default) nests result under ``state_output_key``.

    Args:
        node_spec: Compiled node specification with ``tool_binding``.
        config: Project manifest (used for federated tool map fallback).

    Returns:
        Async ``NodeFunc`` closure for LangGraph registration.

    Raises:
        PlatformServiceError: If tool is unknown, scopes are insufficient,
            or execution fails.
    """
    node_name = node_spec.get("name", "unnamed_platform")
    tool_binding = node_spec.get("tool_binding", "")
    _ = config
    _kind, parsed_tool_name = parse_tool_ref(tool_binding)
    tool_name_value = node_spec.get("tool_name")
    tool_name = tool_name_value if isinstance(tool_name_value, str) else parsed_tool_name
    tool_kind = node_spec.get("tool_kind", "platform")
    node_config = node_spec.get("config", {})
    node_meta: NodeMeta = node_spec.get("meta") or {}
    if tool_name.startswith("platform:"):
        tool_name = tool_name[len("platform:") :]
    elif tool_name.startswith("federated:"):
        tool_name = tool_name[len("federated:") :]

    # Extract routing keys from config
    state_output_key = node_config.get("state_output_key", "final_output")
    output_mode = node_config.get("output_mode", "wrapped")

    # Routing/output keys stripped from tool config validation
    _ROUTING_KEYS = ("state_input_key", "state_output_key", "output_mode")

    async def platform_executor(state: GraphState, config: RunnableConfig) -> StateUpdate:
        """Execute a platform tool with scope validation and telemetry.

        Args:
            state: Graph execution state (must include ``__token__``).
            config: LangGraph runnable config with callback handlers.

        Returns:
            State update with tool result and ``_last_node`` marker.

        Raises:
            PlatformServiceError: On scope violation, config validation
                failure, or execution timeout.
        """
        _ = config
        registry = _get_platform_registry()

        logger.debug("Executing Platform Node: %s -> %s", node_name, tool_binding)

        # 1. Lookup tool in registry (raises PlatformServiceError if unknown)
        registration = registry.get(tool_name)

        token = state.get("__token__")
        if token is None:
            from contextunity.core.exceptions import PlatformServiceError

            raise PlatformServiceError("Context token missing in state")
        registry.check_scopes(tool_name, token)

        # 3. Validate config against tool's Pydantic schema
        tool_config_dict = node_config.get("tool_config", {})
        validated_config = registry.validate_config(
            tool_name,
            dict(tool_config_dict) if is_json_dict(tool_config_dict) else {},
        )

        # 4. Execute tool with telemetry
        async def _execute():
            """Invoke the registered tool executor with validated config."""
            return await registration.executor(state, validated_config)

        try:
            result = await tool_telemetry(
                node_name=node_name,
                tool_binding=tool_binding,
                tool_kind=tool_kind,
                node_meta=node_meta,
                state=state,
                execute_fn=_execute,
            )
        except asyncio.TimeoutError as exc:
            from contextunity.core.exceptions import PlatformServiceError

            raise PlatformServiceError(
                message=f"Platform tool '{tool_name}' timed out on node '{node_name}'",
                node_name=node_name,
                tool_binding=tool_binding,
            ) from exc

        # 5. Route output — direct mode for router_* tools (state field updates),
        #    wrapped mode (default) for brain/shield/worker/privacy tools.
        if output_mode == "direct":
            if not is_object_dict(result):
                from contextunity.core.exceptions import PlatformServiceError

                raise PlatformServiceError(
                    message=f"Platform tool '{tool_name}' returned non-object result",
                    node_name=node_name,
                    tool_binding=tool_binding,
                )
            direct_update: StateUpdate = dict(result)
            direct_update["_last_node"] = node_name
            direct_update["intermediate_results"] = {node_name: result}
            return direct_update
        return {
            str(state_output_key): result,
            "intermediate_results": {node_name: result},
            "_last_node": node_name,
        }

    return platform_executor


__all__ = ["get_platform_required_scopes", "make_platform_node"]
