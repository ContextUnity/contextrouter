"""Helper functions for execution mixins — shared logic extracted to avoid duplication."""

from __future__ import annotations

import dataclasses
from typing import Any

from contextcore import ContextToken, get_context_unit_logger

from contextrouter.core.registry import graph_registry
from contextrouter.modules.observability import (
    LangfuseRequestCtx,
    get_langfuse_callbacks,
)
from contextrouter.modules.observability.auto_tracer import BrainAutoTracer
from contextrouter.service.payloads import ExecuteAgentPayload

logger = get_context_unit_logger(__name__)


def _resolve_tenant_id(token) -> str:
    """Derive tenant_id from token. Token is the single point of truth."""
    if token and getattr(token, "allowed_tenants", ()):
        return token.allowed_tenants[0]
    return "default"


def build_execution_token(
    token,
    project_config: dict | None = None,
    agent_id: str | None = None,
    platform: str | None = None,
):
    """Attenuate a project token for graph execution.

    Pure attenuation — NO permission augmentation.
    The root token (minted by SDK's `mint_client_token`) carries
    scopes needed by SecureNode/SecureTool (tool:*, shield:secrets:read,
    zero:*, graph:*). This function only sets the agent_id for
    provenance tracking.
    """
    if token is None:
        return token

    from contextcore.tokens import TokenBuilder

    # Determine the provenance identity for this execution entry
    effective_agent_id = f"agent:{agent_id}" if agent_id else None

    # If no identity to set, return token unchanged
    if not effective_agent_id and not platform:
        return token

    # For tokens without upstream provenance, inject platform origin
    # so the execution chain doesn't look orphaned.
    has_provenance = bool(getattr(token, "provenance", ()))

    if not has_provenance and platform:
        # Bootstrap provenance with platform origin, then attenuate with agent_id
        clean_platform = platform if not platform.startswith("agent:") else platform[6:]
        token = dataclasses.replace(token, provenance=(clean_platform,))

    if effective_agent_id:
        return TokenBuilder().attenuate(
            token,
            agent_id=effective_agent_id,
        )

    return token


def extract_last_user_msg(input_data: dict) -> str:
    """Extract the last user message from execution input."""
    if "messages" in input_data and isinstance(input_data["messages"], list):
        for m in reversed(input_data["messages"]):
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            content = getattr(m, "content", None) or (
                m.get("content") if isinstance(m, dict) else None
            )
            if role == "user" and content:
                return str(content)
    elif "input" in input_data and isinstance(input_data["input"], str):
        return input_data["input"]
    return ""


def resolve_graph(graph_name: str, tenant_id: str, project_graphs: dict) -> tuple[str, Any]:
    """Resolve graph name and build graph instance.

    Returns:
        (resolved_graph_name, compiled_graph)
    """
    logger.info(
        "resolve_graph: name=%s, tenant=%s, project_graphs keys=%s, registry=%s",
        graph_name,
        tenant_id,
        list(project_graphs.keys()),
        graph_registry.has(graph_name),
    )
    if not graph_registry.has(graph_name):
        # Built-in graphs use @register_graph decorators — trigger import
        _ensure_builtin_graphs_loaded()

    if not graph_registry.has(graph_name):
        mapped_name = project_graphs.get(tenant_id)
        if mapped_name and mapped_name == graph_name:
            pass
        elif project_graph := project_graphs.get(graph_name):
            graph_name = project_graph
        else:
            if not graph_registry.has(graph_name):
                default_graph = project_graphs.get(tenant_id)
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
    return graph_name, builder()


_builtin_loaded = False


def _ensure_builtin_graphs_loaded():
    """Import built-in graph modules so @register_graph decorators execute."""
    global _builtin_loaded
    if _builtin_loaded:
        return
    _builtin_loaded = True

    _graph_modules = [
        "contextrouter.cortex.graphs.commerce.graph",
        "contextrouter.cortex.graphs.commerce.matcher.graph",
        "contextrouter.cortex.graphs.tool_executor.graph",
    ]
    import importlib

    for mod_name in _graph_modules:
        try:
            importlib.import_module(mod_name)
            logger.info("Loaded graph module: %s", mod_name)
        except ImportError as e:
            logger.warning("Failed to load graph module %s: %s", mod_name, e)


def _redact_sensitive_keys(data: Any) -> Any:
    """Recursively redact large or sensitive values for observability persistence.

    Domain considerations:
    - ContextBrain (Storage/UI): Redact bulky texts (`_prompt`, `_prompts`, `schema_description`)
      to prevent DB bloat and ensure fast rendering in ContextView traces.
    - ContextShield (Security): Redact cryptographic artifacts (`_signature`) from
      telemetry to minimize exposure of verification hashes outside the execution boundary.
    """
    if isinstance(data, dict):
        return {
            k: "**REDACTED**"
            if isinstance(k, str)
            and (k.endswith(("_signature", "_prompt", "_prompts")) or k == "schema_description")
            else _redact_sensitive_keys(v)
            for k, v in data.items()
        }
    return [_redact_sensitive_keys(i) for i in data] if isinstance(data, list) else data


_LANGFUSE_KEYS = (
    "langfuse_project_id",
    "langfuse_public_key",
    "langfuse_secret_key",
    "langfuse_host",
    "langfuse_trace_id",
    "langfuse_trace_url",
)


def _clean_for_trace(metadata: dict) -> dict:
    """Prepare execution metadata for Brain trace persistence.

    Domain boundary: ContextRouter (Execution) -> ContextBrain (Observability).
    The `project_config` contains heavy manifest definitions needed by ContextRouter
    at runtime (e.g., SecureNode verification, tool resolution).
    Before sending this payload over gRPC to Brain, we redact payload-heavy and
    security-sensitive properties, preserving only the structural telemetry data.
    """
    clean_meta = dict(metadata)
    config = clean_meta.get("project_config")

    if isinstance(config, dict):
        # Apply domain-appropriate redactions to the manifest configuration
        clean_meta["project_config"] = _redact_sensitive_keys(config)

    # 1. Target both the root metadata and the nested config for cleanup/sorting
    targets = [clean_meta]
    if isinstance(clean_meta.get("project_config"), dict):
        targets.append(clean_meta["project_config"])

    is_langfuse_enabled = clean_meta.get("langfuse_enabled", False)

    # Base UI keys: always show whether it's enabled or not
    ui_bottom_keys = ["langfuse_enabled"]

    if is_langfuse_enabled:
        # Include all actual Langfuse telemetry details
        ui_bottom_keys.extend(_LANGFUSE_KEYS)

    # Ensure provenance is strictly at the bottom
    ui_bottom_keys.append("_inner_provenance")

    for target in targets:
        if not is_langfuse_enabled:
            # Strip out all Langfuse details when perfectly disabled
            for k in _LANGFUSE_KEYS:
                target.pop(k, None)

        # 2. Swap keys to the end so they stack neatly in the ContextView UI
        for k in ui_bottom_keys:
            if k in target:
                target[k] = target.pop(k)

    return clean_meta


def prepare_execution(
    params: ExecuteAgentPayload,
    tenant_id: str,
    token: Any,
    project_configs: dict,
) -> tuple[dict, dict, str, Any, Any]:
    """Prepare execution input, metadata, and callbacks.

    Returns:
        (execution_input, metadata, effective_user_id, callbacks, auto_tracer)
    """
    project_config = project_configs.get(tenant_id, {})
    execution_input = params.input.copy()
    metadata = execution_input.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    metadata["tenant_id"] = tenant_id
    metadata["agent_id"] = params.agent_id or ""
    metadata["platform"] = metadata.get("platform") or "grpc"

    # Fully intact config for the graph runtime
    metadata["project_config"] = project_config

    execution_input["metadata"] = metadata

    effective_user_id = getattr(token, "user_id", None) if token else None

    if effective_user_id:
        metadata["user_id"] = effective_user_id

    # Build per-request Langfuse context from project-supplied metadata
    langfuse_ctx = LangfuseRequestCtx.from_metadata(metadata)

    callbacks = get_langfuse_callbacks(
        session_id=metadata.get("session_id", ""),
        user_id=effective_user_id,
        platform=metadata.get("platform", ""),
        langfuse_ctx=langfuse_ctx,
    )

    # Ensure the actual enabled state is stored back in metadata
    # so ContextView can display the 'Langfuse (Disabled)' badge
    metadata["langfuse_enabled"] = langfuse_ctx.enabled

    auto_tracer = BrainAutoTracer()
    if hasattr(callbacks, "callbacks"):
        callbacks.callbacks.append(auto_tracer)
    else:
        callbacks.append(auto_tracer)

    return execution_input, metadata, effective_user_id, callbacks, auto_tracer, langfuse_ctx


def merge_token_usage(auto_tracer: BrainAutoTracer, state: dict) -> dict[str, int | float]:
    """Merge token usage from AutoTracer callbacks and graph state."""
    auto_tokens = auto_tracer.get_token_usage()
    state_tokens = state.get("_token_usage", {}) if isinstance(state, dict) else {}
    return {
        "input_tokens": max(
            auto_tokens.get("input_tokens", 0), state_tokens.get("input_tokens", 0)
        ),
        "output_tokens": max(
            auto_tokens.get("output_tokens", 0), state_tokens.get("output_tokens", 0)
        ),
        "total_cost": max(auto_tokens.get("total_cost", 0.0), state_tokens.get("total_cost", 0.0)),
    }


def extract_answer(result: dict) -> str:
    """Extract final answer content from graph result."""
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        ans = getattr(last_msg, "content", "") or (
            last_msg.get("content") if isinstance(last_msg, dict) else ""
        )
        return str(ans)[:5000]
    return ""


def serialize_messages(state: dict) -> dict:
    """Serialize LangChain message objects in state to dicts."""
    if "messages" not in state:
        return state
    serialized = []
    for m in state["messages"]:
        if hasattr(m, "model_dump"):
            serialized.append(m.model_dump())
        elif hasattr(m, "dict"):
            serialized.append(m.dict())
        else:
            serialized.append(m)
    state["messages"] = serialized
    return state


async def log_execution_trace(
    *,
    auto_tracer: BrainAutoTracer,
    result: dict,
    token: ContextToken | None,
    tenant_id: str,
    params: ExecuteAgentPayload,
    metadata: dict,
    effective_user_id: str | None,
    graph_name: str,
    wall_ms: int,
    last_user_msg: str,
    guard_result: Any,
    execution_input: dict,
    stream: bool = False,
    error: str = "",
) -> None:
    """Log execution trace to Brain via brain_trace_tools.

    Calls the underlying async function directly (not via LangChain ainvoke)
    because this is an infrastructure call — LangChain schema validation on
    TypedDict annotations (ToolCallSummary, TotalTokenUsage) causes
    ``'dict' object is not callable`` errors.
    """
    try:
        from contextrouter.modules.tools.brain_trace_tools import (
            log_execution_trace as _trace_fn,
        )

        token_usage = merge_token_usage(auto_tracer, result)
        answer_content = extract_answer(result)

        base_prov = list(token.provenance) if token and hasattr(token, "provenance") else []
        inner_prov = metadata.get("_inner_provenance", [])

        # Accumulator stores flat strings (node:X, tool:Y:mode, scope:Z).
        # Merge token provenance (delegation chain) with execution trace.
        # Do not use global deduplication ('step not in provenance_flat') here,
        # as it destroys identical scopes used by subsequent nodes. UI handles its own per-node deduplication.
        provenance_flat = list(base_prov)
        for step in inner_prov:
            if isinstance(step, str):
                provenance_flat.append(step)

        steps_list = auto_tracer.get_nested_steps()
        if error:
            metadata["graph_error"] = error
            steps_list.append(
                {"tool": "graph_failure", "status": "error", "timing_ms": wall_ms, "result": error}
            )

        # Pre-compute security flags to avoid inline conditional expression
        sec_flags: dict = {}
        if last_user_msg or error:
            sec_flags["shield_enabled"] = guard_result is not None
            if error:
                sec_flags["error"] = error

        # Call the underlying coroutine directly — bypasses LangChain @tool
        # schema validation which fails on TypedDict annotations.
        await _trace_fn.coroutine(
            tenant_id=tenant_id,
            agent_id=params.agent_id or "",
            session_id=metadata.get("session_id", ""),
            user_id=effective_user_id or "",
            graph_name=graph_name,
            tool_calls=auto_tracer.get_tool_calls_summary(),
            token_usage=token_usage,
            timing_ms=wall_ms,
            steps=steps_list,
            platform=metadata.get("platform", ""),
            model_key=metadata.get("model_key", ""),
            iterations=1,
            message_count=len(execution_input.get("messages", [])),
            user_query=last_user_msg,
            final_answer=answer_content,
            metadata=_clean_for_trace(metadata),
            provenance=provenance_flat,
            security_flags=sec_flags,
        )
    except Exception as tr_err:
        logger.warning("AutoTracer failed to log execution: %s", tr_err, exc_info=True)
