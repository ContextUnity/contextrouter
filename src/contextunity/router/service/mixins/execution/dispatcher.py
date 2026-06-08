"""ExecuteDispatcher and StreamDispatcher mixin."""

from __future__ import annotations

from collections.abc import AsyncIterator

from contextunity.core import ContextToken, contextunit_pb2, get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, SecurityError
from contextunity.core.types import JsonDict, is_json_dict
from grpc.aio import ServicerContext

from contextunity.router.core.context import (
    reset_current_access_token,
    set_current_access_token,
)
from contextunity.router.cortex.dispatcher import invoke_dispatcher, stream_dispatcher
from contextunity.router.cortex.dispatcher_agent.types import DispatcherState
from contextunity.router.cortex.types import (
    MessageDict,
    RegisteredProjectConfig,
    extract_message_content,
    extract_message_role,
    serialize_message_object,
)
from contextunity.router.service.decorators import grpc_error_handler, grpc_stream_error_handler
from contextunity.router.service.helpers import make_response, parse_unit
from contextunity.router.service.mixins.execution.metadata_helpers import (
    execution_metadata_from_payload,
    merge_json_metadata,
)
from contextunity.router.service.payloads import (
    DispatcherResponsePayload,
    ExecuteDispatcherPayload,
    StreamDispatcherEventPayload,
)
from contextunity.router.service.security import validate_dispatcher_access
from contextunity.router.service.shield_check import check_user_input

from .helpers import (
    _intersect_tenant_with_project,
    _resolve_tenant_id,
    get_registered_project_config,
)
from .types import ProjectConfigMap

logger = get_contextunit_logger(__name__)

ContextUnit = contextunit_pb2.ContextUnit


def _authorized_dispatcher_project_config(
    configs: ProjectConfigMap,
    token: ContextToken,
    tenant_id: str,
    project_id: str | None,
) -> tuple[RegisteredProjectConfig, str]:
    """Resolve a dispatcher project and enforce its tenant intersection."""
    project_config = get_registered_project_config(configs, tenant_id, project_id=project_id)
    if project_id and not project_config:
        raise ConfigurationError(f"Unknown dispatcher project_id '{project_id}'")
    if project_config:
        tenant_id = _intersect_tenant_with_project(token, tenant_id, project_config)
    return project_config, tenant_id


def _response_messages_from_result(
    result: DispatcherState,
    request_messages: list[MessageDict],
) -> list[JsonDict]:
    """Serialize only messages produced after the current dispatcher request."""
    messages_raw = result.get("messages")
    if not isinstance(messages_raw, list):
        return []
    response_messages: list[JsonDict] = []
    for msg in messages_raw:
        serialized = serialize_message_object(msg)
        if is_json_dict(serialized):
            role_raw = serialized.get("role") or serialized.get("type") or "assistant"
            content_raw = serialized.get("content", "")
            response_messages.append(
                {"role": str(role_raw), "content": str(content_raw)},
            )
            continue
        role = extract_message_role(msg) or "assistant"
        response_messages.append(
            {"role": role, "content": extract_message_content(msg)},
        )
    if not request_messages:
        return response_messages
    last_request = request_messages[-1]
    request_role = last_request.get("role", "")
    request_content = last_request.get("content", "")
    for index in range(len(response_messages) - 1, -1, -1):
        response = response_messages[index]
        if response.get("role") == request_role and response.get("content") == request_content:
            return response_messages[index + 1 :]
    return response_messages


class DispatcherExecutionMixin:
    """Mixin providing ``ExecuteDispatcher`` and ``StreamDispatcher`` gRPC handlers."""

    _project_configs: ProjectConfigMap = {}

    @grpc_error_handler
    async def ExecuteDispatcher(
        self,
        request: ContextUnit,
        context: ServicerContext[ContextUnit, ContextUnit],
    ) -> ContextUnit:
        """Execute the dispatcher agent (unary RPC).

        Validates access, runs Shield, invokes the dispatcher graph,
        and serializes the conversation response.

        Raises:
            ConfigurationError: If the payload is invalid.
            SecurityError: If the token or Shield check fails.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteDispatcherPayload.model_validate(unit.payload or {})
        except Exception as e:  # graceful-degrade: dispatch error logged and returned
            raise ConfigurationError(f"Invalid dispatcher payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        messages: list[MessageDict] = [
            MessageDict(role=msg.role, content=msg.content) for msg in params.messages
        ]

        # Shield firewall check
        last_user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if last_user_msg:
            guard_result = await check_user_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise SecurityError(f"Shield blocked input: {guard_result.reason}")

        # Inject model_key, project prompt and project_config
        project_id_raw = (
            params.metadata.get("project_id") if is_json_dict(params.metadata) else None
        )
        project_id = project_id_raw if isinstance(project_id_raw, str) and project_id_raw else None
        project_config, tenant_id = _authorized_dispatcher_project_config(
            self._project_configs,
            token,
            tenant_id,
            project_id,
        )
        trusted_project_id = project_config.get("project_id")
        planner_prompt: str | None = None
        for graph_entry in (project_config.get("graph") or {}).values():
            graph_cfg = graph_entry.get("config")
            if is_json_dict(graph_cfg):
                candidate = graph_cfg.get("planner_prompt")
                if isinstance(candidate, str) and candidate:
                    planner_prompt = candidate
                    break
        dispatch_metadata = execution_metadata_from_payload(
            params.metadata,
            model_key=params.model_key or None,
            project_id=trusted_project_id if isinstance(trusted_project_id, str) else None,
            project_config=project_config,
            system_prompt=planner_prompt,
        )

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
        response_messages = _response_messages_from_result(result, messages)

        result_meta = result.get("metadata", {})
        extra: JsonDict = result_meta if is_json_dict(result_meta) else {}
        response_payload = DispatcherResponsePayload(
            messages=response_messages,
            session_id=params.session_id,
            metadata=merge_json_metadata(
                {"iteration": int(result.get("iteration", 0)), "platform": params.platform},
                extra,
            ),
        )

        return make_response(
            payload=response_payload.model_dump(),
            trace_id=str(unit.trace_id),
            security=unit.security,
        )

    @grpc_stream_error_handler
    async def StreamDispatcher(
        self,
        request: ContextUnit,
        context: ServicerContext[ContextUnit, ContextUnit],
    ) -> AsyncIterator[ContextUnit]:
        """Stream dispatcher agent execution with per-event progress.

        Yields event payloads for each dispatcher iteration, then
        delegates cleanup to the token context manager.

        Raises:
            ConfigurationError: If the payload is invalid.
            SecurityError: If the token or Shield check fails.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteDispatcherPayload.model_validate(unit.payload or {})
        except Exception as e:  # graceful-degrade: dispatch error logged and returned
            logger.exception("Dispatcher payload validation failed: %s", type(e).__name__)
            raise ConfigurationError("Invalid dispatcher payload") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        messages: list[MessageDict] = [
            MessageDict(role=msg.role, content=msg.content) for msg in params.messages
        ]

        # Shield firewall check
        last_user_msg = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if last_user_msg:
            guard_result = await check_user_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise SecurityError(f"Shield blocked input: {guard_result.reason}")

        # Inject project_config, model_key, and project prompt. Mirrors
        # ExecuteDispatcher so StreamDispatcher inherits the same model
        # selection, system prompt, and project_config wiring — without
        # this the stream path falls back to the dispatcher hardcoded
        # SYSTEM_PROMPT and the default model.
        project_id_raw = (
            params.metadata.get("project_id") if is_json_dict(params.metadata) else None
        )
        project_id = project_id_raw if isinstance(project_id_raw, str) and project_id_raw else None
        project_config, tenant_id = _authorized_dispatcher_project_config(
            self._project_configs,
            token,
            tenant_id,
            project_id,
        )
        trusted_project_id = project_config.get("project_id")
        planner_prompt: str | None = None
        for graph_entry in (project_config.get("graph") or {}).values():
            graph_cfg = graph_entry.get("config")
            if is_json_dict(graph_cfg):
                candidate = graph_cfg.get("planner_prompt")
                if isinstance(candidate, str) and candidate:
                    planner_prompt = candidate
                    break
        dispatch_metadata = execution_metadata_from_payload(
            params.metadata,
            model_key=params.model_key or None,
            project_id=trusted_project_id if isinstance(trusted_project_id, str) else None,
            project_config=project_config,
            system_prompt=planner_prompt,
        )

        token_ref = set_current_access_token(token)
        try:

            def _to_json_dict(obj: object) -> JsonDict:
                """Coerce event or payload data to L2 JSON dict."""
                if is_json_dict(obj):
                    return dict(obj)
                return {"output": str(obj)}

            async for event in stream_dispatcher(
                messages=messages,
                tenant_id=tenant_id,
                session_id=params.session_id,
                platform=params.platform,
                metadata=dispatch_metadata,
                max_iterations=params.max_iterations,
                allowed_tools=params.allowed_tools,
                denied_tools=params.denied_tools,
                access_token=token,
            ):
                event_type = event.get("event_type", "unknown")
                event_data = event.get("data", event)

                event_payload = StreamDispatcherEventPayload(
                    event_type=str(event_type),
                    data=_to_json_dict(event_data),
                    timestamp=None,
                )

                yield make_response(
                    payload=_to_json_dict(event_payload.model_dump()),
                    trace_id=str(unit.trace_id),
                    security=unit.security,
                )
        finally:
            reset_current_access_token(token_ref)
