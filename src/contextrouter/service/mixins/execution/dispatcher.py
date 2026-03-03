"""ExecuteDispatcher and StreamDispatcher mixin."""

from __future__ import annotations

from contextcore import get_context_unit_logger

from contextrouter.cortex.runners.dispatcher import invoke_dispatcher, stream_dispatcher
from contextrouter.cortex.runtime_context import (
    reset_current_access_token,
    set_current_access_token,
)
from contextrouter.service.decorators import grpc_error_handler, grpc_stream_error_handler
from contextrouter.service.helpers import make_response, parse_unit
from contextrouter.service.payloads import (
    DispatcherResponsePayload,
    ExecuteDispatcherPayload,
    StreamDispatcherEventPayload,
)
from contextrouter.service.security import validate_dispatcher_access

from .helpers import _resolve_tenant_id

logger = get_context_unit_logger(__name__)


class DispatcherExecutionMixin:
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
