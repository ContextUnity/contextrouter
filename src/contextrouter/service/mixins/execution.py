"""Execution mixin — ExecuteAgent, StreamAgent, ExecuteDispatcher, StreamDispatcher."""

from __future__ import annotations

import time

from contextcore import get_context_unit_logger

from contextrouter.core.registry import graph_registry
from contextrouter.cortex.runners.dispatcher import invoke_dispatcher, stream_dispatcher
from contextrouter.cortex.runtime_context import (
    reset_current_access_token,
    set_current_access_token,
)
from contextrouter.modules.observability import (
    get_langfuse_callbacks,
    get_langfuse_trace_id,
    get_langfuse_trace_url,
    trace_context,
)
from contextrouter.service.decorators import grpc_error_handler, grpc_stream_error_handler
from contextrouter.service.helpers import make_response, parse_unit
from contextrouter.service.payloads import (
    DispatcherResponsePayload,
    ExecuteAgentPayload,
    ExecuteDispatcherPayload,
    StreamDispatcherEventPayload,
)
from contextrouter.service.security import sanitize_for_struct, validate_dispatcher_access

logger = get_context_unit_logger(__name__)


def _resolve_tenant_id(token) -> str:
    """Derive tenant_id from token. Token is the single point of truth."""
    if token and getattr(token, "allowed_tenants", ()):
        return token.allowed_tenants[0]
    return "default"


class ExecutionMixin:
    """Mixin providing ExecuteAgent, ExecuteDispatcher, StreamDispatcher handlers."""

    @grpc_error_handler
    async def ExecuteAgent(self, request, context):
        """Execute a specific named agent/graph.

        Request payload: ExecuteAgentPayload (tenant_id, agent_id, input, config)
        """
        unit = parse_unit(request)

        try:
            params = ExecuteAgentPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        graph_name = params.agent_id

        # Resolve graph from registry
        if not graph_registry.has(graph_name):
            mapped_name = self._project_graphs.get(tenant_id)
            if mapped_name and mapped_name == graph_name:
                pass
            elif project_graph := self._project_graphs.get(graph_name):
                graph_name = project_graph
            else:
                if not graph_registry.has(graph_name):
                    default_graph = self._project_graphs.get(tenant_id)
                    if default_graph:
                        logger.info(
                            "Using default graph '%s' for tenant '%s'",
                            default_graph,
                            tenant_id,
                        )
                        graph_name = default_graph
                    else:
                        raise ValueError(f"Graph '{graph_name}' not found")

        builder = graph_registry.get(graph_name)
        graph = builder()

        # Extract user input for Shield firewall check
        execution_input_copy = params.input.copy()
        last_user_msg = ""

        # Check messages list format
        if "messages" in execution_input_copy and isinstance(
            execution_input_copy["messages"], list
        ):
            for m in reversed(execution_input_copy["messages"]):
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
                content = getattr(m, "content", None) or (
                    m.get("content") if isinstance(m, dict) else None
                )
                if role == "user" and content:
                    last_user_msg = str(content)
                    break
        elif "input" in execution_input_copy and isinstance(execution_input_copy["input"], str):
            last_user_msg = execution_input_copy["input"]

        if last_user_msg:
            guard_result = await self._guard.check_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise PermissionError(f"Shield blocked: {guard_result.reason}")

        # Inject project config to preserve agent persona
        project_config = self._project_configs.get(tenant_id, {})
        execution_input = params.input.copy()
        metadata = execution_input.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        if not metadata.get("system_prompt") and project_config.get("planner_prompt"):
            metadata["system_prompt"] = project_config["planner_prompt"]
        # Ensure trace node sees tenant/agent from top-level params
        metadata.setdefault("tenant_id", tenant_id)
        metadata.setdefault("agent_id", params.agent_id or "")
        execution_input["metadata"] = metadata

        effective_user_id = getattr(token, "user_id", None) if token else None
        # Ensure metadata carries the authoritative user_id from the token
        if effective_user_id:
            metadata["user_id"] = effective_user_id
        callbacks = get_langfuse_callbacks(
            session_id=metadata.get("session_id", ""),
            user_id=effective_user_id,
            platform=metadata.get("platform", ""),
        )

        langfuse_trace_id = ""
        langfuse_trace_url = ""
        t0 = time.monotonic()

        token_ref = set_current_access_token(token)
        try:
            with trace_context(
                session_id=metadata.get("session_id", ""),
                platform=metadata.get("platform", "grpc"),
                name=f"agent:{graph_name}",
                user_id=effective_user_id,
                trace_id=str(unit.trace_id),
                trace_input=execution_input,
                trace_metadata=metadata,
            ):
                # Capture Langfuse trace ID for downstream storage
                langfuse_trace_id = get_langfuse_trace_id()
                langfuse_trace_url = get_langfuse_trace_url()

                # Inject Langfuse IDs into metadata so trace node saves them
                if langfuse_trace_id:
                    metadata["langfuse_trace_id"] = langfuse_trace_id
                    metadata["langfuse_trace_url"] = langfuse_trace_url
                    execution_input["metadata"] = metadata

                config = params.config.copy() if params.config else {}
                config["callbacks"] = callbacks
                result = await graph.ainvoke(execution_input, config=config)

            if isinstance(result, dict) and "messages" in result:
                serialized = []
                for m in result["messages"]:
                    if hasattr(m, "model_dump"):
                        serialized.append(m.model_dump())
                    elif hasattr(m, "dict"):
                        serialized.append(m.dict())
                    else:
                        serialized.append(m)
                result["messages"] = serialized
        finally:
            reset_current_access_token(token_ref)

        wall_ms = int((time.monotonic() - t0) * 1000)

        if isinstance(result, dict):
            # Inject observability metadata into response
            if langfuse_trace_id:
                result["langfuse_trace_id"] = langfuse_trace_id
                result["langfuse_trace_url"] = langfuse_trace_url
            result["wall_ms"] = wall_ms
            result = sanitize_for_struct(result)

        return make_response(
            payload=result if isinstance(result, dict) else {"output": result},
            trace_id=str(unit.trace_id),
            provenance=list(unit.provenance) + [f"router:agent:{graph_name}"],
            security=unit.security,
        )

    @grpc_stream_error_handler
    async def StreamAgent(self, request, context):
        """Stream agent/graph execution with real-time progress events.

        Yields ContextUnit events:
          - event_type="progress": {node, step} for each completed node
          - event_type="result":   full graph state (same as ExecuteAgent response)
        """
        unit = parse_unit(request)

        try:
            params = ExecuteAgentPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        graph_name = params.agent_id

        # Resolve graph (same logic as ExecuteAgent)
        if not graph_registry.has(graph_name):
            mapped_name = self._project_graphs.get(tenant_id)
            if mapped_name and mapped_name == graph_name:
                pass
            elif project_graph := self._project_graphs.get(graph_name):
                graph_name = project_graph
            else:
                if not graph_registry.has(graph_name):
                    default_graph = self._project_graphs.get(tenant_id)
                    if default_graph:
                        graph_name = default_graph
                    else:
                        raise ValueError(f"Graph '{graph_name}' not found")

        builder = graph_registry.get(graph_name)
        graph = builder()

        # Extract user input for Shield firewall check
        execution_input_copy = params.input.copy()
        last_user_msg = ""

        # Check messages list format
        if "messages" in execution_input_copy and isinstance(
            execution_input_copy["messages"], list
        ):
            for m in reversed(execution_input_copy["messages"]):
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
                content = getattr(m, "content", None) or (
                    m.get("content") if isinstance(m, dict) else None
                )
                if role == "user" and content:
                    last_user_msg = str(content)
                    break
        elif "input" in execution_input_copy and isinstance(execution_input_copy["input"], str):
            last_user_msg = execution_input_copy["input"]

        if last_user_msg:
            guard_result = await self._guard.check_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise PermissionError(f"Shield blocked: {guard_result.reason}")

        # Inject project config
        project_config = self._project_configs.get(tenant_id, {})
        execution_input = params.input.copy()
        metadata = execution_input.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if not metadata.get("system_prompt") and project_config.get("planner_prompt"):
            metadata["system_prompt"] = project_config["planner_prompt"]
        # Ensure trace node sees tenant/agent from top-level params
        metadata.setdefault("tenant_id", tenant_id)
        metadata.setdefault("agent_id", params.agent_id or "")
        execution_input["metadata"] = metadata

        effective_user_id = getattr(token, "user_id", None) if token else None
        # Ensure metadata carries the authoritative user_id from the token
        # (mirrors ExecuteAgent logic — trace node reads metadata["user_id"])
        if effective_user_id:
            metadata["user_id"] = effective_user_id
        callbacks = get_langfuse_callbacks(
            session_id=metadata.get("session_id", ""),
            user_id=effective_user_id,
            platform=metadata.get("platform", ""),
        )

        langfuse_trace_id = ""
        langfuse_trace_url = ""
        t0 = time.monotonic()

        token_ref = set_current_access_token(token)
        final_state = {}
        try:
            with trace_context(
                session_id=metadata.get("session_id", ""),
                platform=metadata.get("platform", "grpc"),
                name=f"agent:{graph_name}:stream",
                user_id=effective_user_id,
                trace_id=str(unit.trace_id),
                trace_input=execution_input,
                trace_metadata=metadata,
            ):
                # Capture Langfuse trace ID
                langfuse_trace_id = get_langfuse_trace_id()
                langfuse_trace_url = get_langfuse_trace_url()

                # Inject into metadata for trace node
                if langfuse_trace_id:
                    metadata["langfuse_trace_id"] = langfuse_trace_id
                    metadata["langfuse_trace_url"] = langfuse_trace_url
                    execution_input["metadata"] = metadata

                config = params.config.copy() if params.config else {}
                config["callbacks"] = callbacks

                # astream_events gives lifecycle visibility:
                # on_chain_start → instant SSE progress when node begins
                # on_chain_end   → accumulate final_state from output
                graph_nodes = set(graph.nodes.keys()) if hasattr(graph, "nodes") else set()
                async for event in graph.astream_events(
                    execution_input, config=config, version="v2"
                ):
                    kind = event.get("event", "")
                    name = event.get("name", "")

                    # Only top-level graph nodes, skip internal sub-chains
                    if name not in graph_nodes:
                        continue

                    if kind == "on_chain_start":
                        # Node STARTING — send progress immediately
                        yield make_response(
                            payload=sanitize_for_struct(
                                {
                                    "event_type": "progress",
                                    "node": name,
                                    "step": {},
                                }
                            ),
                            trace_id=str(unit.trace_id),
                            provenance=list(unit.provenance)
                            + [f"router:agent:{graph_name}:stream"],
                            security=unit.security,
                        )
                    elif kind == "on_chain_end":
                        # Node finished — merge output into state
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            final_state.update(output)
                            # Send step details (row_count, timing, etc.)
                            new_steps = output.get("_steps", [])
                            if new_steps:
                                yield make_response(
                                    payload=sanitize_for_struct(
                                        {
                                            "event_type": "progress",
                                            "node": name,
                                            "step": new_steps[-1],
                                        }
                                    ),
                                    trace_id=str(unit.trace_id),
                                    provenance=list(unit.provenance)
                                    + [f"router:agent:{graph_name}:stream"],
                                    security=unit.security,
                                )
        finally:
            reset_current_access_token(token_ref)

        wall_ms = int((time.monotonic() - t0) * 1000)

        # Serialize messages in final state
        if "messages" in final_state:
            serialized = []
            for m in final_state["messages"]:
                if hasattr(m, "model_dump"):
                    serialized.append(m.model_dump())
                elif hasattr(m, "dict"):
                    serialized.append(m.dict())
                else:
                    serialized.append(m)
            final_state["messages"] = serialized

        if isinstance(final_state, dict):
            if langfuse_trace_id:
                final_state["langfuse_trace_id"] = langfuse_trace_id
                final_state["langfuse_trace_url"] = langfuse_trace_url
            final_state["wall_ms"] = wall_ms
            final_state = sanitize_for_struct(final_state)

        # Final result event
        yield make_response(
            payload={"event_type": "result", **final_state},
            trace_id=str(unit.trace_id),
            provenance=list(unit.provenance) + [f"router:agent:{graph_name}"],
            security=unit.security,
        )

    @grpc_error_handler
    async def ExecuteDispatcher(self, request, context):
        """Execute dispatcher agent (non-streaming).

        Security: Requires "dispatcher:execute" read scope.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteDispatcherPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        messages = [{"role": msg.role, "content": msg.content} for msg in params.messages]

        # Shield firewall check
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if last_user_msg:
            guard_result = await self._guard.check_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise PermissionError(f"Shield blocked: {guard_result.reason}")

        # Inject model_key and project prompt
        dispatch_metadata = dict(params.metadata)
        if params.model_key:
            dispatch_metadata["model_key"] = params.model_key

        project_config = self._project_configs.get(tenant_id, {})
        if project_config.get("planner_prompt"):
            dispatch_metadata["system_prompt"] = project_config["planner_prompt"]

        token_ref = set_current_access_token(token)
        try:
            result = await invoke_dispatcher(
                messages=messages,
                tenant_id=tenant_id,
                session_id=params.session_id,
                platform=params.platform,
                metadata=dispatch_metadata,
                max_iterations=params.max_iterations,
                allowed_tools=params.allowed_tools,
                denied_tools=params.denied_tools,
                access_token=token,
            )
        finally:
            reset_current_access_token(token_ref)

        # Extract messages from result
        response_messages = []
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "type"):
                    role = "assistant"
                    if msg.type == "human":
                        role = "user"
                    elif msg.type == "system":
                        role = "system"
                    elif msg.type == "tool":
                        role = "tool"

                    content = getattr(msg, "content", "")
                    if not isinstance(content, (str, list)):
                        content = str(content)

                    response_messages.append({"role": role, "content": content})
                elif isinstance(msg, dict):
                    response_messages.append(
                        {
                            "role": msg.get("role") or msg.get("type") or "assistant",
                            "content": msg.get("content") or "",
                        }
                    )

        response_payload = DispatcherResponsePayload(
            messages=response_messages,
            session_id=params.session_id,
            metadata={
                "iteration": result.get("iteration", 0),
                "platform": params.platform,
                **result.get("metadata", {}),
            },
        )

        return make_response(
            payload=response_payload.model_dump(),
            trace_id=str(unit.trace_id),
            provenance=list(unit.provenance) + ["router:dispatcher:execute"],
            security=unit.security,
        )

    @grpc_stream_error_handler
    async def StreamDispatcher(self, request, context):
        """Stream dispatcher agent execution.

        Security: Requires "dispatcher:execute" read scope.

        Yields: ContextUnit events with event_type and data.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteDispatcherPayload(**unit.payload)
        except Exception as e:
            raise ValueError(f"Invalid payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        messages = [{"role": msg.role, "content": msg.content} for msg in params.messages]

        # Shield firewall check
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if last_user_msg:
            guard_result = await self._guard.check_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise PermissionError(f"Shield blocked: {guard_result.reason}")

        token_ref = set_current_access_token(token)
        try:
            async for event in stream_dispatcher(
                messages=messages,
                tenant_id=tenant_id,
                session_id=params.session_id,
                platform=params.platform,
                metadata=params.metadata,
                max_iterations=params.max_iterations,
                allowed_tools=params.allowed_tools,
                denied_tools=params.denied_tools,
                access_token=token,
            ):
                event_payload = StreamDispatcherEventPayload(
                    event_type=event.get("event_type", "unknown"),
                    data=event.get("data", event),
                )

                yield make_response(
                    payload=event_payload.model_dump(),
                    trace_id=str(unit.trace_id),
                    provenance=list(unit.provenance) + ["router:dispatcher:stream"],
                    security=unit.security,
                )
        finally:
            reset_current_access_token(token_ref)


__all__ = ["ExecutionMixin"]
