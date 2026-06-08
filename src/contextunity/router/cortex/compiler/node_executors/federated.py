"""Federated Node Executor for the Graph Compiler.
Creates a LangGraph node function that executes a federated tool
via the BiDi stream_executor_manager. Output is tagged as untrusted.
"""

from __future__ import annotations

import asyncio

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError
from contextunity.core.manifest.helpers import parse_tool_ref
from contextunity.core.types import is_json_dict, is_object_dict
from langchain_core.runnables import RunnableConfig

from contextunity.router.core.exceptions import (
    RouterGraphBuilderError,
    RouterToolTimeout,
)
from contextunity.router.cortex.compiler.node_executors.telemetry import tool_telemetry
from contextunity.router.cortex.compiler.state_routing import read_state_input
from contextunity.router.cortex.compiler.types import NodeMeta
from contextunity.router.cortex.config_resolution import metadata_project_config

from ...compiler.node_config import NodeConfig
from ...compiler.types import CompilerNodeSpec, ProjectManifest
from ...types import GraphState, NodeFunc, StateUpdate

logger = get_contextunit_logger(__name__)


def _federated_skip_update(
    node_name: str,
    state_output_key: str,
    payload: dict[str, object],
) -> StateUpdate:
    """Return a skip payload with the same bookkeeping fields as a successful run."""
    return {
        str(state_output_key): payload,
        "intermediate_results": {node_name: payload},
        "_last_node": node_name,
    }


def _resolve_federated_project_id(state: GraphState) -> str:
    """Resolve BiDi stream routing project id from execution metadata."""
    project_config = metadata_project_config(state)
    project_id_raw = project_config.get("project_id")
    if isinstance(project_id_raw, str) and project_id_raw:
        return project_id_raw

    metadata = state.get("metadata")
    if is_object_dict(metadata):
        metadata_project = metadata.get("project_id")
        if isinstance(metadata_project, str) and metadata_project:
            return metadata_project

    raise RouterGraphBuilderError(
        message="Federated node execution requires project_id in execution metadata"
    )


# Maximum allowed timeout for federated tools (seconds)
_MAX_FEDERATED_TIMEOUT: int = 300

# Connection retry: attempt a few times with delays before reporting failure.
_CONN_RETRY_ATTEMPTS: int = 3
_CONN_RETRY_DELAYS: tuple[float, ...] = (3.0, 5.0, 8.0)

# Error messages that indicate a stream/connection-level failure
# (as opposed to a data-level error like "invalid SQL syntax")
_CONNECTION_ERROR_PATTERNS = (
    "connection is closed",
    "stream disconnected",
    "stream not available",
    "no active stream",
    "connection reset",
    "connection refused",
    "broken pipe",
)


def _is_connection_error(error_msg: str) -> bool:
    """Classify whether an error message indicates a connection-level failure.

    Matches against known connection-failure patterns to distinguish
    retryable transport errors from data-level errors (e.g., invalid SQL).

    Args:
        error_msg: Error text from the federated tool response.

    Returns:
        ``True`` if the error matches a known connection-failure pattern.
    """
    lower = error_msg.lower()
    return any(p in lower for p in _CONNECTION_ERROR_PATTERNS)


def _normalize_federated_args(args: dict[str, object], tool_name: str) -> dict[str, object]:
    """Map planner/tool state into handler kwargs expected by federated tools."""
    if tool_name != "medical_sql" or "sql" in args:
        return args

    nested = args.get("data")
    if is_object_dict(nested) and "sql" in nested:
        merged = {str(key): value for key, value in nested.items()}
        for key, value in args.items():
            if key != "data" and key not in merged:
                merged[key] = value
        return merged

    for alias in ("query", "statement"):
        if alias in args:
            return {**args, "sql": args[alias]}

    return args


def _get_stream_executor_manager():
    """Lazy import to avoid circular dependencies at module level."""
    from contextunity.router.service.stream_executors import (
        get_stream_executor_manager,
    )

    return get_stream_executor_manager()


def make_federated_node(node_spec: CompilerNodeSpec, config: ProjectManifest) -> NodeFunc:
    """Create a LangGraph node for federated tool execution via BiDi streaming.

    The returned closure:
    1. Resolves tool binding from ``federated:<name>`` syntax.
    2. Extracts args from ``state_input_fields`` or ``state_input_key``.
    3. Skips execution if upstream set an ``error`` in state.
    4. Retries connection-level failures up to ``_CONN_RETRY_ATTEMPTS``.
    5. Tags output as ``__untrusted__`` — federated sources are
       outside the platform trust boundary.

    Args:
        node_spec: Compiled node specification with ``tool_binding``.
        config: Project manifest for federated tool map resolution.

    Returns:
        Async ``NodeFunc`` closure for LangGraph registration.

    Raises:
        RouterGraphBuilderError: If ``tool_binding`` is not a valid ``federated:`` ref.
        RouterToolTimeout: If the tool does not respond within deadline.
        PlatformServiceError: If all connection retry attempts fail.
    """
    _ = config
    node_name = node_spec.get("name", "unnamed_federated")
    tool_binding = node_spec.get("tool_binding", "")
    kind, parsed_tool_name = parse_tool_ref(tool_binding)
    if kind != "federated" or not parsed_tool_name:
        raise RouterGraphBuilderError(
            message=(
                f"Federated node '{node_name}' tool_binding must use 'federated:<name>'; "
                f"got '{tool_binding}'."
            ),
            node_name=node_name,
        )

    tool_name = node_spec.get("tool_name", parsed_tool_name)
    tool_kind = node_spec.get("tool_kind", "federated")
    node_meta: NodeMeta = node_spec.get("meta") or {}
    if tool_name.startswith("federated:"):
        tool_name = tool_name[len("federated:") :]
    elif tool_name.startswith("platform:"):
        tool_name = tool_name[len("platform:") :]

    node_config_raw = node_spec.get("config")
    _cfg_dict = dict(node_config_raw) if is_json_dict(node_config_raw) else {}
    nc = NodeConfig.model_validate(_cfg_dict)

    raw_timeout = nc.timeout if nc.timeout is not None else 60
    timeout = min(max(int(raw_timeout), 1), _MAX_FEDERATED_TIMEOUT)
    state_input_key = nc.state_input_key or "messages"
    state_input_fields = nc.state_input_fields
    state_output_key = nc.state_output_key or "final_output"

    async def federated_executor(state: GraphState, config: RunnableConfig) -> StateUpdate:
        """Execute a federated tool call with retry and telemetry.

        Args:
            state: Graph execution state (must include ``__token__``).
            config: LangGraph runnable config with callback handlers.

        Returns:
            State update with untrusted-tagged result and ``_last_node``.

        Raises:
            RouterToolTimeout: If execution exceeds the configured timeout.
            PlatformServiceError: If all connection retry attempts are exhausted.
        """
        _ = config
        logger.debug(
            "Executing Federated Node: %s -> %s (timeout: %ds)",
            node_name,
            tool_binding,
            timeout,
        )

        manager = _get_stream_executor_manager()

        # Token is always injected into state by secure_node — guaranteed non-None.
        ctx_token = state.get("__token__")
        if ctx_token is None:
            raise RouterGraphBuilderError(message="Context token missing in state")

        project_id = _resolve_federated_project_id(state)

        # ── Build tool args from state ──
        # Two modes:
        #   state_input_fields: ["sql", "format"]  → args = {sql: state["sql"], format: state["format"]}
        #   state_input_key: "messages"             → args = state["messages"] (legacy, dict passthrough)
        if state_input_fields:
            final_args: dict[str, object] = {}
            for field in state_input_fields:
                val = read_state_input(state, field)
                if val is not None:
                    final_args[field] = val
            logger.debug(
                "Federated %s: built args from state fields %s", node_name, list(final_args.keys())
            )
        else:
            input_data = read_state_input(state, state_input_key, default={})
            if is_object_dict(input_data):
                final_args = {str(key): value for key, value in input_data.items()}
            else:
                final_args = {"data": input_data}

        final_args = _normalize_federated_args(final_args, tool_name)

        if tool_name == "medical_sql" and "sql" not in final_args:
            missing_sql_error = (
                "Planner output missing required 'sql' field for medical_sql execution"
            )
            logger.warning(
                "Federated %s: skipping — %s (args keys=%s)",
                node_name,
                missing_sql_error,
                list(final_args.keys()),
            )
            return _federated_skip_update(
                node_name,
                state_output_key,
                {
                    "error": missing_sql_error,
                    "__skipped__": True,
                },
            )

        # Skip execution if upstream set an error
        # Check both state-level error (TypedDict) and error inside the input dict
        # (from output_format=json failure: final_output = {"error": "..."})
        upstream_error = read_state_input(state, "error")
        if not upstream_error:
            upstream_error = final_args.get("error")
        if upstream_error:
            logger.info(
                "Federated %s: skipping — upstream error: %s",
                node_name,
                str(upstream_error)[:200],
            )
            return _federated_skip_update(
                node_name,
                state_output_key,
                {"error": upstream_error, "__skipped__": True},
            )

        logger.debug(
            "Federated %s: sending args keys=%s to %s",
            node_name,
            list(final_args.keys()),
            tool_name,
        )

        async def _execute():
            # Retry loop for connection-level errors.
            """Execute with connection-level retry loop.

            Raises:
                PlatformServiceError: If all retry attempts are exhausted.
            """
            last_conn_error = ""
            res: object = None
            for attempt in range(_CONN_RETRY_ATTEMPTS):
                conn_error = ""
                try:
                    res = await asyncio.wait_for(
                        manager.execute(
                            project_id=project_id,
                            tool_name=tool_name,
                            args=final_args,
                            timeout=timeout,
                        ),
                        timeout=timeout,
                    )
                    # Check for connection-level error in result dict
                    if is_object_dict(res) and "error" in res and not res.get("rows"):
                        msg = str(res["error"])
                        if _is_connection_error(msg):
                            conn_error = msg
                except (ConnectionError, PermissionError) as exc:
                    conn_error = str(exc)

                if not conn_error:
                    return res

                last_conn_error = conn_error
                delay = _CONN_RETRY_DELAYS[min(attempt, len(_CONN_RETRY_DELAYS) - 1)]
                logger.warning(
                    "Federated %s: connection error (attempt %d/%d): %s — retrying in %.0fs",
                    node_name,
                    attempt + 1,
                    _CONN_RETRY_ATTEMPTS,
                    conn_error,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise PlatformServiceError(
                    message=(
                        f"Database connection unavailable. "
                        f"Tool '{tool_binding}' failed after "
                        f"{_CONN_RETRY_ATTEMPTS} attempts: {last_conn_error}. "
                        f"Please check the project connection and retry."
                    ),
                    node_name=node_name,
                    tool_binding=tool_binding,
                )

        try:
            result = await tool_telemetry(
                node_name=node_name,
                tool_binding=tool_binding,
                tool_kind=tool_kind,
                node_meta=node_meta,
                state=state,
                execute_fn=_execute,
                tool_args=final_args,
            )
        except asyncio.TimeoutError as e:
            raise RouterToolTimeout(
                message=(
                    f"Node '{node_name}' federated tool '{tool_binding}' "
                    f"did not respond within {timeout}s deadline."
                ),
                node_name=node_name,
                tool_binding=tool_binding,
                timeout_seconds=timeout,
            ) from e

        # Tag output as untrusted — federated source cannot be trusted
        tagged_result: dict[str, object] = {
            "__untrusted__": True,
            "__source__": tool_binding,
        }
        if is_object_dict(result):
            tagged_result.update({str(key): value for key, value in result.items()})
        else:
            tagged_result["data"] = result

        # Write to both output key and intermediate_results (accumulated).
        # Only node_name as key — no project-specific aliases in platform code.
        return {
            str(state_output_key): tagged_result,
            "intermediate_results": {node_name: tagged_result},
            "_last_node": node_name,
        }

    return federated_executor


__all__ = ["make_federated_node"]
