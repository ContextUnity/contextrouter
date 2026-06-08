"""ExecuteAgent and StreamAgent mixin -- handles unary and streaming agent invocation RPCs."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass

from contextunity.core import contextunit_pb2, get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, SecurityError
from contextunity.core.types import is_json_dict, is_object_dict, is_object_list
from grpc.aio import ServicerContext

from contextunity.router.core.context import (
    get_accumulated_provenance,
    init_provenance_accumulator,
    reset_current_access_token,
    reset_provenance_accumulator,
    set_current_access_token,
)
from contextunity.router.modules.observability import (
    get_langfuse_trace_id,
    get_langfuse_trace_url,
    trace_context,
)
from contextunity.router.modules.observability.astream_tracer_replay import (
    replay_astream_event_to_tracer,
)
from contextunity.router.service.decorators import grpc_error_handler, grpc_stream_error_handler
from contextunity.router.service.helpers import make_response, parse_unit
from contextunity.router.service.payloads import ExecuteAgentPayload
from contextunity.router.service.security import sanitize_for_struct, validate_dispatcher_access
from contextunity.router.service.shield_check import check_user_input

from .helpers import (
    _resolve_tenant_id,
    build_execution_token,
    build_run_config,
    extract_last_user_msg,
    extract_state_update_from_chain_output,
    log_execution_trace,
    merge_graph_state_update,
    merge_token_usage,
    prepare_execution,
    resolve_graph,
    resolve_recursion_limit,
    serialize_messages,
)
from .metadata_helpers import execution_metadata_for_trace
from .types import ExecutionMetadata, GraphResult, ProjectConfigMap, ProjectGraphMap

logger = get_contextunit_logger(__name__)

ContextUnit = contextunit_pb2.ContextUnit


def _payload_dict(obj: object) -> dict[str, object]:
    """Coerce an object into a ``dict[str, object]`` for protobuf."""
    if is_json_dict(obj):
        return dict(obj)
    return {"output": obj}


def _metadata_has_pii_masking(metadata: ExecutionMetadata) -> bool:
    """Return True when any registered node enables PII masking."""
    project_cfg = metadata.get("project_config")
    if not is_json_dict(project_cfg):
        return False
    nodes_raw = project_cfg.get("nodes")
    if not isinstance(nodes_raw, list):
        return False
    for node_cfg in nodes_raw:
        if is_json_dict(node_cfg) and node_cfg.get("pii_masking") is True:
            return True
    return False


async def _destroy_pii_session(metadata: ExecutionMetadata) -> None:
    """Destroy ephemeral PII session keys when masking was active."""
    if not _metadata_has_pii_masking(metadata):
        return
    from contextunity.router.cortex.utils.pii import PiiSession

    session_id = str(metadata.get("session_id", "default"))
    PiiSession(session_id).destroy()


class AgentExecutionMixin:
    """Mixin providing ``ExecuteAgent`` and ``StreamAgent`` gRPC handlers."""

    _project_graphs: ProjectGraphMap = {}
    _project_configs: ProjectConfigMap = {}

    @grpc_error_handler
    async def ExecuteAgent(
        self, request: ContextUnit, context: ServicerContext[ContextUnit, ContextUnit]
    ) -> ContextUnit:
        """Execute a specific named agent/graph (unary RPC).

        Validates access, runs Shield input check, invokes the graph,
        and returns the serialized result with trace metadata.

        Raises:
            ConfigurationError: If the payload or graph config is invalid.
            SecurityError: If the token or Shield check fails.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteAgentPayload.model_validate(unit.payload or {})
        except Exception as e:  # graceful-degrade: execution cleanup must not mask primary error
            raise ConfigurationError(f"Invalid agent payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        resolved = resolve_graph(params.agent_id, tenant_id, self._project_graphs)
        graph_name, graph = resolved.name, resolved.graph

        # Extract user input for Shield
        last_user_msg = extract_last_user_msg(params.input)
        guard_result = None
        metadata: ExecutionMetadata = {}

        if last_user_msg:
            guard_result = await check_user_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise SecurityError(f"Shield blocked input: {guard_result.reason}")

        execution_input, metadata, effective_user_id, callbacks, auto_tracer, langfuse_ctx = (
            prepare_execution(
                params,
                tenant_id,
                token,
                self._project_configs,
            )
        )

        langfuse_trace_id = ""
        langfuse_trace_url = ""
        t0 = time.monotonic()
        result: GraphResult = {}
        graph_error = ""

        # Augment token with PII permissions and shield secrets so SecureTool checks pass
        execution_token = build_execution_token(
            token,
            agent_id=params.agent_id,
            platform=str(metadata.get("platform", "grpc")),
        )
        token_ref = set_current_access_token(execution_token)
        accum_ref = init_provenance_accumulator()
        final_accum = []
        try:
            with trace_context(
                session_id=str(metadata.get("session_id", "")),
                platform=str(metadata.get("platform", "grpc")),
                name=f"agent:{graph_name}",
                user_id=effective_user_id,
                trace_id=str(unit.trace_id),
                trace_input=execution_input,
                trace_metadata=execution_metadata_for_trace(metadata),
                tenant_id=tenant_id,
                langfuse_ctx=langfuse_ctx,
            ):
                langfuse_trace_id = get_langfuse_trace_id()
                langfuse_trace_url = get_langfuse_trace_url(langfuse_ctx=langfuse_ctx)

                if langfuse_trace_id:
                    metadata["langfuse_trace_id"] = langfuse_trace_id
                    metadata["langfuse_trace_url"] = langfuse_trace_url

                execution_input["metadata"] = metadata

                run_config = build_run_config(
                    params.graph_run_config,
                    callbacks,
                    default_recursion_limit=resolve_recursion_limit(
                        metadata.get("project_config"), graph_name, graph
                    ),
                )
                from contextunity.router.service.mixins.execution.helpers import invoke_graph

                result_raw = await invoke_graph(
                    graph,
                    execution_input,
                    run_config=run_config,
                )
                if is_json_dict(result_raw):
                    normalized_result: GraphResult = dict(result_raw)
                    result = serialize_messages(normalized_result)
                else:
                    result = {}

            final_accum = get_accumulated_provenance()
        except Exception as exc:  # graceful-degrade: execution cleanup must not mask primary error
            final_accum = get_accumulated_provenance()
            graph_error = str(exc)
            logger.error_exc("Graph '%s' failed", graph_name)
            raise
        finally:
            reset_current_access_token(token_ref)
            reset_provenance_accumulator(accum_ref)

            # PII Session cleanup: destroy ephemeral keys if pii_masking was active
            try:
                await _destroy_pii_session(metadata)
            except (
                Exception
            ) as cleanup_e:  # graceful-degrade: execution cleanup must not mask primary error
                logger.warning("Failed to destroy PII session in cleanup: %s", cleanup_e)

            # Pass the aggregated inner provenances out to trace logger
            metadata["_inner_provenance"] = final_accum

            # Always log trace — even on error (partial trace is better than no trace)
            wall_ms = int((time.monotonic() - t0) * 1000)
            await log_execution_trace(
                auto_tracer=auto_tracer,
                result=result,
                token=execution_token,
                tenant_id=tenant_id,
                params=params,
                metadata=metadata,
                effective_user_id=getattr(execution_token, "user_id", None) or effective_user_id,
                graph_name=graph_name,
                wall_ms=wall_ms,
                last_user_msg=last_user_msg,
                guard_result=guard_result,
                execution_input=execution_input,
                error=graph_error,
            )
            from contextunity.router.modules.observability import flush as langfuse_flush

            langfuse_flush(langfuse_ctx)

        wall_ms = int((time.monotonic() - t0) * 1000)

        if langfuse_trace_id:
            result["langfuse_trace_id"] = langfuse_trace_id
            result["langfuse_trace_url"] = langfuse_trace_url
        result["wall_ms"] = wall_ms

        return make_response(
            payload=_payload_dict(sanitize_for_struct(result)),
            trace_id=str(unit.trace_id),
            security=unit.security,
        )

    @grpc_stream_error_handler
    async def StreamAgent(
        self, request: ContextUnit, context: ServicerContext[ContextUnit, ContextUnit]
    ) -> AsyncIterator[ContextUnit]:
        """Stream agent/graph execution with per-node progress events.

        Yields ``progress`` and ``brain_event`` responses for each
        graph node, then a final ``result`` event with the full state.

        Raises:
            ConfigurationError: If the payload or graph config is invalid.
            SecurityError: If the token or Shield check fails.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteAgentPayload.model_validate(unit.payload or {})
        except Exception as e:  # graceful-degrade: execution cleanup must not mask primary error
            raise ConfigurationError(f"Invalid agent payload: {e}") from e

        token = validate_dispatcher_access(unit, context)
        tenant_id = _resolve_tenant_id(token)

        resolved = resolve_graph(params.agent_id, tenant_id, self._project_graphs)
        graph_name, graph = resolved.name, resolved.graph

        # Extract user input for Shield
        last_user_msg = extract_last_user_msg(params.input)
        guard_result = None
        metadata: ExecutionMetadata = {}

        if last_user_msg:
            guard_result = await check_user_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise SecurityError(f"Shield blocked input: {guard_result.reason}")

        execution_input, metadata, effective_user_id, callbacks, auto_tracer, langfuse_ctx = (
            prepare_execution(
                params,
                tenant_id,
                token,
                self._project_configs,
            )
        )

        langfuse_trace_id = ""
        langfuse_trace_url = ""
        t0 = time.monotonic()
        graph_error = ""

        # Augment token with PII permissions and shield secrets so SecureTool checks pass
        execution_token = build_execution_token(
            token,
            agent_id=params.agent_id,
            platform=str(metadata.get("platform", "grpc")),
        )
        token_ref = set_current_access_token(execution_token)
        accum_ref = init_provenance_accumulator()
        final_state: GraphResult = {}
        final_accum = []
        try:
            with trace_context(
                session_id=str(metadata.get("session_id", "")),
                platform=str(metadata.get("platform", "grpc")),
                name=f"agent:{graph_name}:stream",
                user_id=effective_user_id,
                trace_id=str(unit.trace_id),
                trace_input=execution_input,
                trace_metadata=execution_metadata_for_trace(metadata),
                tenant_id=tenant_id,
                langfuse_ctx=langfuse_ctx,
            ):
                langfuse_trace_id = get_langfuse_trace_id()
                langfuse_trace_url = get_langfuse_trace_url(langfuse_ctx=langfuse_ctx)

                if langfuse_trace_id:
                    metadata["langfuse_trace_id"] = langfuse_trace_id
                    metadata["langfuse_trace_url"] = langfuse_trace_url

                execution_input["metadata"] = metadata

                run_config = build_run_config(
                    params.graph_run_config,
                    callbacks,
                    default_recursion_limit=resolve_recursion_limit(
                        metadata.get("project_config"), graph_name, graph
                    ),
                )

                # astream_events: lifecycle visibility per node
                raw_nodes = getattr(graph, "nodes", None)
                graph_nodes: set[str] = set()
                if is_object_dict(raw_nodes):
                    graph_nodes = set(raw_nodes.keys())

                from contextunity.router.service.mixins.execution.helpers import (
                    iter_graph_events,
                )

                async for event in iter_graph_events(
                    graph,
                    execution_input,
                    run_config=run_config,
                ):
                    # Feed event into tracer for Brain trace logging
                    # (callbacks are unreliable on astream_events path)
                    await replay_astream_event_to_tracer(auto_tracer, event)

                    kind = str(event.get("event", ""))
                    name = str(event.get("name", ""))

                    if kind == "on_custom_event" and name == "brain_event":
                        data = event.get("data")
                        brain_event_payload: object | None = None
                        # Brain events may arrive either as a dataclass
                        # (``BrainEvent`` from graph nodes) or as a dict when
                        # LangGraph serializes the custom event payload. Both
                        # shapes must propagate to the client SSE stream.
                        if is_dataclass(data) and not isinstance(data, type):
                            brain_event_payload = asdict(data)
                        elif is_object_dict(data):
                            inner = data.get("event")
                            if is_dataclass(inner) and not isinstance(inner, type):
                                brain_event_payload = asdict(inner)
                            elif is_object_dict(inner):
                                brain_event_payload = inner
                        if brain_event_payload is not None:
                            yield make_response(
                                payload=_payload_dict(
                                    sanitize_for_struct(
                                        {
                                            "event_type": "brain_event",
                                            "event": brain_event_payload,
                                        }
                                    )
                                ),
                                trace_id=str(unit.trace_id),
                                security=unit.security,
                            )
                        continue

                    is_graph_node = name in graph_nodes
                    is_graph_root = name == "LangGraph"

                    if not is_graph_node and not is_graph_root:
                        continue

                    if kind == "on_chain_start" and is_graph_node:
                        data = event.get("data")
                        input_val: object = {}
                        # Use the same wide guard as on_chain_end so non-JSON
                        # values like datetime/Decimal don't silently produce
                        # empty progress on start while end accepts them.
                        if is_object_dict(data):
                            input_val = data.get("input", {})
                        # Sanitize the progress event payload before yielding
                        # to the SSE stream. Raw node input can carry
                        # datetime / UUID / Decimal / non-JSON objects that
                        # fail JSON encoding or leak PII.
                        delta_payload = _payload_dict(sanitize_for_struct(input_val))
                        delta_payload["event_type"] = "progress"
                        delta_payload["node"] = name
                        yield make_response(
                            payload=delta_payload,
                            trace_id=str(unit.trace_id),
                            security=unit.security,
                        )
                    elif kind == "on_chain_end" and (is_graph_node or is_graph_root):
                        data = event.get("data")
                        output: object = {}
                        if is_object_dict(data):
                            output = data.get("output", {})
                        state_update = extract_state_update_from_chain_output(output)
                        if state_update:
                            final_state = merge_graph_state_update(final_state, state_update)
                            steps_raw = state_update.get("_steps", [])
                            if is_object_list(steps_raw) and steps_raw and is_graph_node:
                                last_step = steps_raw[-1]
                                if is_object_dict(last_step):
                                    # Sanitize the per-step progress payload to
                                    # keep the SSE wire JSON-clean and PII-aware.
                                    event_payload = _payload_dict(sanitize_for_struct(last_step))
                                    event_payload["event_type"] = "progress"
                                    event_payload["node"] = name
                                    event_payload["step"] = sanitize_for_struct(last_step)

                                    yield make_response(
                                        payload=event_payload,
                                        trace_id=str(unit.trace_id),
                                        security=unit.security,
                                    )
            final_accum = get_accumulated_provenance()
        except Exception as exc:  # graceful-degrade: execution cleanup must not mask primary error
            final_accum = get_accumulated_provenance()
            graph_error = str(exc)
            logger.error_exc("Stream graph '%s' failed", graph_name)
            raise
        finally:
            reset_current_access_token(token_ref)
            reset_provenance_accumulator(accum_ref)

            # PII Session cleanup: destroy ephemeral keys if pii_masking was active
            try:
                await _destroy_pii_session(metadata)
            except (
                Exception
            ) as cleanup_e:  # graceful-degrade: execution cleanup must not mask primary error
                logger.warning("Failed to destroy PII session in cleanup: %s", cleanup_e)

            metadata["_inner_provenance"] = final_accum

            # Always log trace — even on error
            wall_ms = int((time.monotonic() - t0) * 1000)
            await log_execution_trace(
                auto_tracer=auto_tracer,
                result=final_state,
                token=execution_token,
                tenant_id=tenant_id,
                params=params,
                metadata=metadata,
                effective_user_id=effective_user_id,
                graph_name=graph_name,
                wall_ms=wall_ms,
                last_user_msg=last_user_msg,
                guard_result=guard_result,
                execution_input=execution_input,
                stream=True,
                error=graph_error,
            )
            from contextunity.router.modules.observability import flush as langfuse_flush

            langfuse_flush(langfuse_ctx)

        final_state = serialize_messages(final_state)

        if langfuse_trace_id:
            final_state["langfuse_trace_id"] = langfuse_trace_id
            final_state["langfuse_trace_url"] = langfuse_trace_url
        final_state["wall_ms"] = wall_ms
        final_state["_token_usage"] = merge_token_usage(auto_tracer, final_state)

        final_payload = _payload_dict(sanitize_for_struct(final_state))
        final_payload["event_type"] = "result"

        yield make_response(
            payload=final_payload,
            trace_id=str(unit.trace_id),
            security=unit.security,
        )
