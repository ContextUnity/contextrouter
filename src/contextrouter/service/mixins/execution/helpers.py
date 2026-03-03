"""Helper functions for execution mixins — shared logic extracted to avoid duplication."""

from __future__ import annotations

from typing import Any

from contextcore import get_context_unit_logger

from contextrouter.core.registry import graph_registry
from contextrouter.modules.observability import (
    LangfuseRequestCtx,
    get_langfuse_callbacks,
)
from contextrouter.modules.observability.auto_tracer import BrainAutoTracer
from contextrouter.service.payloads import ExecuteAgentPayload

logger = get_context_unit_logger(__name__)

# Permissions for internal graph tools (PII masking, etc.)
# Router authorizes these when executing graphs — the user's token
# doesn't need to carry them explicitly.
_GRAPH_INTERNAL_PERMISSIONS = (
    "tool:anonymize_text",
    "tool:deanonymize_text",
    "tool:check_pii",
)


def _resolve_tenant_id(token) -> str:
    """Derive tenant_id from token. Token is the single point of truth."""
    if token and getattr(token, "allowed_tenants", ()):
        return token.allowed_tenants[0]
    return "default"


def build_execution_token(token):
    """Create an execution token augmented with graph-internal tool permissions.

    The Router has already validated the user's identity and access.
    Graphs need to call internal infrastructure tools (PII masking, etc.)
    that the user's project token doesn't explicitly authorize.
    We augment the token with these permissions so SecureTool checks pass.

    The original token is not mutated (ContextToken is frozen).
    """
    if token is None:
        return token

    from contextcore import ContextToken

    existing = set(token.permissions)
    needed = [p for p in _GRAPH_INTERNAL_PERMISSIONS if p not in existing]
    if not needed:
        return token  # Already has all permissions

    return ContextToken(
        token_id=token.token_id,
        permissions=token.permissions + tuple(needed),
        allowed_tenants=token.allowed_tenants,
        exp_unix=token.exp_unix,
        revocation_id=token.revocation_id,
        user_id=token.user_id,
        agent_id=token.agent_id,
        user_namespace=token.user_namespace,
    )


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

    if not metadata.get("system_prompt") and project_config.get("planner_prompt"):
        metadata["system_prompt"] = project_config["planner_prompt"]

    metadata["tenant_id"] = tenant_id
    metadata["agent_id"] = params.agent_id or ""
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

    auto_tracer = BrainAutoTracer()
    if hasattr(callbacks, "callbacks"):
        callbacks.callbacks.append(auto_tracer)
    else:
        callbacks.append(auto_tracer)

    return execution_input, metadata, effective_user_id, callbacks, auto_tracer, langfuse_ctx


def merge_token_usage(auto_tracer: BrainAutoTracer, state: dict) -> dict[str, int]:
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
    unit: Any,
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
    """Log execution trace to Brain via the log_execution_trace tool."""
    import contextrouter.modules.tools.brain_trace_tools  # noqa
    from contextrouter.modules.tools import get_tool

    trace_tool = get_tool("log_execution_trace")
    if not trace_tool:
        return

    try:
        token_usage = merge_token_usage(auto_tracer, result)
        answer_content = extract_answer(result)
        suffix = ":stream" if stream else ""

        kwargs = {
            "tenant_id": tenant_id,
            "agent_id": params.agent_id or "",
            "session_id": metadata.get("session_id", ""),
            "user_id": effective_user_id or "",
            "graph_name": graph_name,
            "tool_calls": auto_tracer.get_tool_calls_summary(),
            "token_usage": token_usage,
            "timing_ms": wall_ms,
            "steps": auto_tracer.get_nested_steps(),
            "platform": metadata.get("platform", ""),
            "model_key": metadata.get("model_key", ""),
            "iterations": 1,
            "message_count": len(execution_input.get("messages", [])),
            "user_query": last_user_msg,
            "final_answer": answer_content,
            "metadata": metadata,
            "provenance": (
                list(unit.provenance)
                + [f"router:agent:{graph_name}{suffix}"]
                + auto_tracer.get_provenance()
            ),
            "security_flags": {
                "shield_enabled": guard_result is not None,
                **(
                    {
                        "error": error,
                    }
                    if error
                    else {}
                ),
            }
            if last_user_msg or error
            else {},
        }

        from contextcore import ContextToken

        from contextrouter.cortex.runtime_context import (
            reset_current_access_token as _reset,
        )
        from contextrouter.cortex.runtime_context import (
            set_current_access_token as _set,
        )

        admin_token = ContextToken(
            token_id="internal-trace",  # nosec B106 — service identity, not a password
            permissions=("admin", "tool:log_execution_trace"),
            allowed_tenants=(tenant_id,),
            user_id=effective_user_id or "",
        )
        ref = _set(admin_token)
        try:
            await trace_tool.ainvoke(kwargs)
        finally:
            _reset(ref)
    except Exception as tr_err:
        logger.warning("AutoTracer failed to log execution: %s", tr_err)
