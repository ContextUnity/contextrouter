"""ExecuteNode mixin -- handles single-node execution RPCs for graph debugging and testing."""

from __future__ import annotations

import inspect
import time
from typing import TypeGuard

from contextunity.core import contextunit_pb2, get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, SecurityError
from contextunity.core.permissions import Permissions
from contextunity.core.types import is_json_dict
from grpc.aio import ServicerContext

from contextunity.router.core.context import (
    get_accumulated_provenance,
    init_provenance_accumulator,
    reset_current_access_token,
    reset_provenance_accumulator,
    set_current_access_token,
)
from contextunity.router.cortex.types import GraphNodeExecutor
from contextunity.router.modules.observability import (
    get_langfuse_trace_id,
    get_langfuse_trace_url,
    trace_context,
)
from contextunity.router.service.decorators import grpc_error_handler
from contextunity.router.service.helpers import make_response, parse_unit
from contextunity.router.service.payloads import ExecuteAgentPayload, ExecuteNodePayload
from contextunity.router.service.security import sanitize_for_struct, validate_dispatcher_access

from .helpers import (
    _resolve_tenant_id,
    build_execution_token,
    build_run_config,
    get_registered_project_config,
    log_execution_trace,
    prepare_execution,
    project_id_from_agent_id,
    resolve_execution_project_id,
    resolve_graph,
    resolve_recursion_limit,
    serialize_messages,
)
from .metadata_helpers import execution_metadata_for_trace
from .types import (
    ExecutionMetadata,
    GraphResult,
    ProjectConfigMap,
    ProjectGraphMap,
    RouterCallbackMap,
)

logger = get_contextunit_logger(__name__)

ContextUnit = contextunit_pb2.ContextUnit


def _extract_graph_key(resolved_name: str) -> str:
    """Extract the manifest graph key from a ``project:<id>:<key>`` name.

    Args:
        resolved_name: Fully qualified registry name.

    Returns:
        The graph key portion, or the original name if not namespaced.
    """
    parts = resolved_name.split(":", 2)
    if len(parts) == 3 and parts[0] == "project":
        return parts[2]
    return resolved_name


def _is_graph_node_executor(value: object) -> TypeGuard[GraphNodeExecutor]:
    """Return True when *value* exposes LangGraph node ``ainvoke``."""
    return callable(getattr(value, "ainvoke", None))


def _graph_node_executor(
    graph_nodes: dict[str, GraphNodeExecutor],
    node_name: str,
    *,
    graph_name: str,
) -> GraphNodeExecutor:
    """Resolve a compiled node executor from a LangGraph node map."""
    if node_name not in graph_nodes:
        raise ConfigurationError(
            message=f"Node '{node_name}' does not exist in graph '{graph_name}'",
        )
    executor_obj = graph_nodes[node_name]
    if not _is_graph_node_executor(executor_obj):
        raise ConfigurationError(
            message=f"Node '{node_name}' is not invokable in graph '{graph_name}'",
        )
    return executor_obj


def _node_has_pii_masking(metadata: ExecutionMetadata, node_name: str) -> bool:
    """Return True when the requested node enables PII masking."""
    project_cfg = metadata.get("project_config")
    if not is_json_dict(project_cfg):
        return False
    nodes_raw = project_cfg.get("nodes")
    if not isinstance(nodes_raw, list):
        return False
    for node_cfg in nodes_raw:
        if not is_json_dict(node_cfg):
            continue
        if node_cfg.get("name") == node_name and node_cfg.get("pii_masking") is True:
            return True
    return False


def _coerce_graph_result(value: object) -> GraphResult:
    """Normalize single-node invoke output to ``GraphResult``."""
    if is_json_dict(value):
        return dict(value)
    return {}


class NodeExecutionMixin:
    """Mixin providing the ``ExecuteNode`` gRPC handler for single-node execution."""

    _project_graphs: ProjectGraphMap = {}
    _project_configs: ProjectConfigMap = {}
    _project_router_callbacks: RouterCallbackMap = {}

    @grpc_error_handler
    async def ExecuteNode(
        self,
        request: ContextUnit,
        context: ServicerContext[ContextUnit, ContextUnit],
    ) -> ContextUnit:
        """Execute a single node from a compiled graph (unary RPC).

        Validates access, checks ``router_callbacks`` allowlist,
        invokes the isolated node, and returns the result with trace data.

        Raises:
            ConfigurationError: If the payload or node name is invalid.
            SecurityError: If the token check or callback allowlist fails.
        """
        unit = parse_unit(request)

        try:
            params = ExecuteNodePayload.model_validate(unit.payload or {})
        except Exception as e:  # graceful-degrade: node cleanup must not mask primary error
            raise ConfigurationError(
                message=f"Invalid ExecuteNode payload: {e}",
            ) from e

        token = validate_dispatcher_access(
            unit,
            context,
            permission=Permissions.ROUTER_EXECUTE_NODE,
            rpc_name="ExecuteNode",
        )
        tenant_id = _resolve_tenant_id(token)

        resolved = resolve_graph(params.graph_name, tenant_id, self._project_graphs)
        graph_name, graph = resolved.name, resolved.graph

        graph_key = _extract_graph_key(graph_name)
        project_id = resolve_execution_project_id(
            graph_selector=params.graph_name,
            resolved_graph_name=graph_name,
        )
        callbacks_map = self._project_router_callbacks.get(project_id, {})
        allowed_callbacks = callbacks_map.get(graph_key, [])

        if params.node_name not in allowed_callbacks:
            raise SecurityError(
                message=(
                    f"Node '{params.node_name}' is not exposed for direct execution "
                    f"via router_callbacks in graph '{graph_key}'."
                ),
                node_name=params.node_name,
            )

        node_executor = _graph_node_executor(
            graph.nodes,
            params.node_name,
            graph_name=graph_name,
        )

        dummy_params = ExecuteAgentPayload(
            agent_id=params.graph_name,
            input=params.state,
            config=params.config_overrides,
        )

        execution_input, metadata, effective_user_id, callbacks, auto_tracer, langfuse_ctx = (
            prepare_execution(
                dummy_params,
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

        execution_token = build_execution_token(
            token,
            agent_id=params.graph_name,
            platform=str(metadata.get("platform", "grpc")),
        )
        token_ref = set_current_access_token(execution_token)
        accum_ref = init_provenance_accumulator()
        final_accum: list[str | tuple[str, ...]] = []
        try:
            with trace_context(
                session_id=str(metadata.get("session_id", "")),
                platform=str(metadata.get("platform", "grpc")),
                name=f"node_execute:{graph_name}:{params.node_name}",
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

                project_config = get_registered_project_config(
                    self._project_configs,
                    tenant_id,
                    project_id=project_id_from_agent_id(params.graph_name),
                )
                default_recursion_limit = resolve_recursion_limit(
                    project_config,
                    graph_name,
                    graph,
                )
                run_config = build_run_config(
                    dummy_params.graph_run_config,
                    callbacks,
                    default_recursion_limit=default_recursion_limit,
                )

                invoke_result = node_executor.ainvoke(execution_input, config=run_config)
                result = _coerce_graph_result(
                    await invoke_result if inspect.isawaitable(invoke_result) else invoke_result,
                )

            result = serialize_messages(result)
            final_accum = get_accumulated_provenance()
        except Exception as exc:  # graceful-degrade: node cleanup must not mask primary error
            final_accum = get_accumulated_provenance()
            graph_error = str(exc)
            logger.exception("ExecuteNode '%s' failed: %s", params.node_name, graph_error)
            raise
        finally:
            reset_current_access_token(token_ref)
            reset_provenance_accumulator(accum_ref)

            try:
                if _node_has_pii_masking(metadata, params.node_name):
                    from contextunity.router.cortex.utils.pii import PiiSession

                    session_id = str(metadata.get("session_id", "default"))
                    pii_session = PiiSession(session_id)
                    pii_session.destroy()
            except (
                Exception
            ) as cleanup_e:  # graceful-degrade: node cleanup must not mask primary error
                logger.warning("Failed to destroy PII session in cleanup: %s", cleanup_e)

            metadata["_inner_provenance"] = final_accum

            wall_ms = int((time.monotonic() - t0) * 1000)
            await log_execution_trace(
                auto_tracer=auto_tracer,
                result=result,
                token=execution_token,
                tenant_id=tenant_id,
                params=dummy_params,
                metadata=metadata,
                effective_user_id=effective_user_id,
                graph_name=graph_name,
                wall_ms=wall_ms,
                last_user_msg="",
                guard_result=None,
                execution_input=execution_input,
                error=graph_error,
            )
            from contextunity.router.modules.observability import flush as langfuse_flush

            langfuse_flush(langfuse_ctx)

        wall_ms = int((time.monotonic() - t0) * 1000)

        response_payload = {
            "output": sanitize_for_struct(result),
            "node_name": params.node_name,
            "execution_ms": wall_ms,
        }

        if langfuse_trace_id:
            response_payload["langfuse_trace_id"] = langfuse_trace_id
            response_payload["langfuse_trace_url"] = langfuse_trace_url

        return make_response(
            payload=response_payload,
            trace_id=str(unit.trace_id),
            security=unit.security,
        )
