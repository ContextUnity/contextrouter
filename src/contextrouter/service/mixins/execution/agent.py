"""ExecuteAgent and StreamAgent mixin."""

from __future__ import annotations

import time

from contextcore import get_context_unit_logger

from contextrouter.cortex.runtime_context import (
    get_accumulated_provenance,
    init_provenance_accumulator,
    reset_current_access_token,
    reset_provenance_accumulator,
    set_current_access_token,
)
from contextrouter.modules.observability import (
    get_langfuse_trace_id,
    get_langfuse_trace_url,
    trace_context,
)
from contextrouter.service.decorators import grpc_error_handler, grpc_stream_error_handler
from contextrouter.service.helpers import make_response, parse_unit
from contextrouter.service.payloads import ExecuteAgentPayload
from contextrouter.service.security import sanitize_for_struct, validate_dispatcher_access
from contextrouter.service.shield_check import check_user_input

from .helpers import (
    _resolve_tenant_id,
    build_execution_token,
    extract_last_user_msg,
    log_execution_trace,
    prepare_execution,
    resolve_graph,
    serialize_messages,
)

logger = get_context_unit_logger(__name__)


class AgentExecutionMixin:
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

        graph_name, graph = resolve_graph(params.agent_id, tenant_id, self._project_graphs)

        # Extract user input for Shield
        last_user_msg = extract_last_user_msg(params.input)
        guard_result = None
        metadata = {}

        if last_user_msg:
            guard_result = await check_user_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise PermissionError(f"Shield blocked: {guard_result.reason}")

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
        result = {}
        graph_error = ""

        # Augment token with PII permissions and shield secrets so SecureTool checks pass
        execution_token = build_execution_token(
            token,
            project_config=metadata.get("project_config"),
            agent_id=params.agent_id,
            platform=metadata.get("platform", "grpc"),
        )
        token_ref = set_current_access_token(execution_token)
        accum_ref = init_provenance_accumulator()
        final_accum = []
        try:
            with trace_context(
                session_id=metadata.get("session_id", ""),
                platform=metadata.get("platform", "grpc"),
                name=f"agent:{graph_name}",
                user_id=effective_user_id,
                trace_id=str(unit.trace_id),
                trace_input=execution_input,
                trace_metadata=metadata,
                tenant_id=tenant_id,
                langfuse_ctx=langfuse_ctx,
            ):
                langfuse_trace_id = get_langfuse_trace_id()
                langfuse_trace_url = get_langfuse_trace_url(langfuse_ctx=langfuse_ctx)

                if langfuse_trace_id:
                    metadata["langfuse_trace_id"] = langfuse_trace_id
                    metadata["langfuse_trace_url"] = langfuse_trace_url

                execution_input["metadata"] = metadata

                config = params.config.copy() if params.config else {}
                config["callbacks"] = callbacks
                result = await graph.ainvoke(execution_input, config=config)

            result = serialize_messages(result) if isinstance(result, dict) else result
            final_accum = get_accumulated_provenance()
        except Exception as exc:
            final_accum = get_accumulated_provenance()
            graph_error = str(exc)
            logger.exception("Graph '%s' failed: %s", graph_name, graph_error)
            raise
        finally:
            reset_current_access_token(token_ref)
            reset_provenance_accumulator(accum_ref)

            # Pass the aggregated inner provenances out to trace logger
            metadata["_inner_provenance"] = final_accum

            # Always log trace — even on error (partial trace is better than no trace)
            wall_ms = int((time.monotonic() - t0) * 1000)
            await log_execution_trace(
                auto_tracer=auto_tracer,
                result=result if isinstance(result, dict) else {},
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
                error=graph_error,
            )
            from contextrouter.modules.observability import flush as langfuse_flush

            langfuse_flush(langfuse_ctx)

        wall_ms = int((time.monotonic() - t0) * 1000)

        if isinstance(result, dict):
            if langfuse_trace_id:
                result["langfuse_trace_id"] = langfuse_trace_id
                result["langfuse_trace_url"] = langfuse_trace_url
            result["wall_ms"] = wall_ms
            result = sanitize_for_struct(result)

        return make_response(
            payload=result if isinstance(result, dict) else {"output": result},
            trace_id=str(unit.trace_id),
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

        graph_name, graph = resolve_graph(params.agent_id, tenant_id, self._project_graphs)

        # Extract user input for Shield
        last_user_msg = extract_last_user_msg(params.input)
        guard_result = None
        metadata = {}

        if last_user_msg:
            guard_result = await check_user_input(
                last_user_msg,
                request_id=str(unit.trace_id),
                tenant=tenant_id,
            )
            if guard_result.blocked:
                raise PermissionError(f"Shield blocked: {guard_result.reason}")

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
            project_config=metadata.get("project_config"),
            agent_id=params.agent_id,
            platform=metadata.get("platform", "grpc"),
        )
        token_ref = set_current_access_token(execution_token)
        accum_ref = init_provenance_accumulator()
        final_state = {}
        final_accum = []
        try:
            with trace_context(
                session_id=metadata.get("session_id", ""),
                platform=metadata.get("platform", "grpc"),
                name=f"agent:{graph_name}:stream",
                user_id=effective_user_id,
                trace_id=str(unit.trace_id),
                trace_input=execution_input,
                trace_metadata=metadata,
                tenant_id=tenant_id,
                langfuse_ctx=langfuse_ctx,
            ):
                langfuse_trace_id = get_langfuse_trace_id()
                langfuse_trace_url = get_langfuse_trace_url(langfuse_ctx=langfuse_ctx)

                if langfuse_trace_id:
                    metadata["langfuse_trace_id"] = langfuse_trace_id
                    metadata["langfuse_trace_url"] = langfuse_trace_url

                execution_input["metadata"] = metadata

                config = params.config.copy() if params.config else {}
                config["callbacks"] = callbacks

                # astream_events: lifecycle visibility per node
                graph_nodes = set(graph.nodes.keys()) if hasattr(graph, "nodes") else set()
                async for event in graph.astream_events(
                    execution_input, config=config, version="v2"
                ):
                    kind = event.get("event", "")
                    name = event.get("name", "")

                    if name not in graph_nodes:
                        continue

                    if kind == "on_chain_start":
                        yield make_response(
                            payload=sanitize_for_struct(
                                {"event_type": "progress", "node": name, "step": {}}
                            ),
                            trace_id=str(unit.trace_id),
                            security=unit.security,
                        )
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            final_state.update(output)
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
                                    security=unit.security,
                                )
            final_accum = get_accumulated_provenance()
        except Exception as exc:
            final_accum = get_accumulated_provenance()
            graph_error = str(exc)
            logger.exception("Stream graph '%s' failed: %s", graph_name, graph_error)
            raise
        finally:
            reset_current_access_token(token_ref)
            reset_provenance_accumulator(accum_ref)

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
            from contextrouter.modules.observability import flush as langfuse_flush

            langfuse_flush(langfuse_ctx)

        final_state = serialize_messages(final_state)

        if isinstance(final_state, dict):
            if langfuse_trace_id:
                final_state["langfuse_trace_id"] = langfuse_trace_id
                final_state["langfuse_trace_url"] = langfuse_trace_url
            final_state["wall_ms"] = wall_ms
            final_state = sanitize_for_struct(final_state)

        yield make_response(
            payload={"event_type": "result", **final_state},
            trace_id=str(unit.trace_id),
            security=unit.security,
        )
