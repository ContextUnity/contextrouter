"""Shared state preparation and response formatting for execution mixin handlers."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.runnables import RunnableConfig


from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, SecurityError
from contextunity.core.types import (
    JsonDict,
    JsonValue,
    is_json_dict,
    is_object_dict,
    is_object_list,
)

from contextunity.router.core.registry import graph_registry
from contextunity.router.cortex.types import (
    RegisteredGraphMap,
    RegisteredProjectConfig,
    RegisteredToolEntry,
    extract_message_content,
    extract_message_role,
    is_runnable_graph,
    merge_graph_state_update,
    serialize_message_object,
)
from contextunity.router.modules.observability import LangfuseRequestCtx, get_langfuse_callbacks
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.modules.tools.schemas import TotalTokenUsage
from contextunity.router.service.mixins.execution.metadata_helpers import (
    copy_execution_metadata,
    execution_metadata_from_payload,
)
from contextunity.router.service.mixins.execution.types import (
    ExecutionContext,
    ExecutionMetadata,
    GraphInput,
    GraphResult,
    GraphRunConfigInput,
    ProjectConfigMap,
    ProjectGraphMap,
    ResolvedGraph,
    RunnableGraph,
    SecurityFlags,
)
from contextunity.router.service.payloads import ExecuteAgentPayload
from contextunity.router.service.shield_check import ShieldCheckResult

logger = get_contextunit_logger(__name__)


def _resolve_tenant_id(token: ContextToken | None) -> str:
    """Derive tenant_id from token. Token is the single point of truth."""
    if token and getattr(token, "allowed_tenants", ()):
        return token.allowed_tenants[0]
    raise SecurityError("Execution requires a token scoped to at least one tenant")


def _intersect_tenant_with_project(
    token: ContextToken | None,
    tenant_id: str,
    project_config: RegisteredProjectConfig,
) -> str:
    """Validate and intersect token tenant with project allowed_tenants.

    The execution ingress tenant_id (from token.allowed_tenants[0]) must be
    covered by the project's manifest allowed_tenants. This prevents
    multi-tenant token scope confusion when projects have distinct tenants.

    Args:
        token: The validated ContextToken (may be None in edge cases).
        tenant_id: The initially resolved tenant_id from token.
        project_config: The resolved project configuration from manifest.

    Returns:
        The validated tenant_id if intersection succeeds.

    Raises:
        SecurityError: If token tenant is not in project allowed_tenants.
    """
    if not token:
        raise SecurityError("Tenant intersection requires valid token")

    token_tenants = token.allowed_tenants
    if not token_tenants:
        raise SecurityError("Token has no allowed_tenants for intersection")

    # Get project allowed_tenants from manifest
    from contextunity.core.manifest.tenants import parse_allowed_tenants_field

    project_allowed_raw = project_config.get("allowed_tenants")
    project_tenants: list[str] | None = parse_allowed_tenants_field(project_allowed_raw)

    if not project_tenants:
        # Project has no explicit allowed_tenants — use project_id as default
        project_id = project_config.get("project_id")
        if isinstance(project_id, str) and project_id:
            project_tenants = [project_id]
        else:
            project_tenants = []

    # Find intersection: token tenant must be in project_tenants
    # Use token's first tenant (already resolved as tenant_id) or check all
    if tenant_id not in project_tenants:
        # Try to find any intersecting tenant
        intersection = [tenant for tenant in token_tenants if tenant in project_tenants]
        if not intersection:
            raise SecurityError(
                (
                    f"Token tenant scope {list(token_tenants)} not covered by "
                    f"project allowed_tenants {project_tenants}. "
                    f"Tenant '{tenant_id}' rejected at execution ingress."
                )
            )
        # Return first intersecting tenant as the effective tenant
        return intersection[0]

    return tenant_id


def resolve_dispatcher_tenant_id(tenant_id: str, token: ContextToken) -> str:
    """Resolve tenant for dispatcher entrypoints.

    Legacy callers pass ``tenant_id="default"``; resolve from the token instead.
    """
    if tenant_id != "default":
        return tenant_id
    return _resolve_tenant_id(token)


def get_registered_project_config(
    configs: ProjectConfigMap,
    tenant_id: str,
    *,
    project_id: str | None = None,
) -> RegisteredProjectConfig:
    """Resolve runtime registration store entry for execution.

    ``_project_configs`` is keyed by ``RegisterManifest`` ``project_id`` (see
    ``RegistrationMixin.RegisterManifest``). ``tenant_id`` always comes from the
    caller token (``_resolve_tenant_id``); it is never omitted.

    Lookup order:

    1. When ``project_id`` is provided (``metadata.project_id``, or parsed from
       ``agent_id`` / ``graph_name`` as ``<project_id>:<graph_key>``), use that key.
    2. Otherwise fall back to ``tenant_id``.

    The fallback is intentional for the default registration shape where
    ``tenant_id`` defaults to ``project_id`` (``get_str(bundle, "tenant_id", project_id)``).
    When a project registers with a distinct tenant, callers must supply
    ``project_id`` explicitly — tenant alone will not alias a different project key.
    """
    if project_id is not None:
        if project_id in configs:
            return configs[project_id]
        return {}
    if tenant_id in configs:
        return configs[tenant_id]
    return {}


def project_id_from_agent_id(agent_id: str | None) -> str | None:
    """Extract project id from ExecuteAgent's graph selector when present."""
    if not agent_id:
        return None
    parts = agent_id.split(":")
    if len(parts) >= 3 and parts[0] == "project":
        return parts[1] or None
    return parts[0] or None


def _project_id_from_registry_graph(resolved_graph_name: str) -> str | None:
    """Extract project id from a compiled registry graph name."""
    parts = resolved_graph_name.split(":", 2)
    if len(parts) == 3 and parts[0] == "project" and parts[1]:
        return parts[1]
    return None


def resolve_execution_project_id(*, graph_selector: str, resolved_graph_name: str) -> str:
    """Resolve the registration project id for execution and callback lookup."""
    project_id = _project_id_from_registry_graph(resolved_graph_name)
    if project_id:
        return project_id
    if graph_selector.startswith("project:"):
        project_id = project_id_from_agent_id(graph_selector)
        if project_id:
            return project_id
    raise SecurityError("Execution requires a project-qualified graph selector")


def build_execution_token(
    token: ContextToken,
    *,
    agent_id: str | None = None,
    platform: str | None = None,
) -> ContextToken:
    """Attenuate a project token for graph execution.

    Pure attenuation — NO permission augmentation.
    The root token (from bootstrap ``get_auth_metadata()`` or a verified caller
    token) carries scopes needed by SecureNode/SecureTool (tool:*,
    privacy:*, graph:*). This function only sets the agent_id for
    provenance tracking.
    """
    from contextunity.core.tokens import TokenBuilder

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

    # If the token lacks a user identity (e.g. Server-to-Server Project Key),
    # bind a per-token service identity so memory tools stay isolated without
    # collapsing all anonymous callers into one shared namespace.
    if getattr(token, "user_id", None) is None:
        token_id = getattr(token, "token_id", None)
        if isinstance(token_id, str) and token_id:
            token = dataclasses.replace(token, user_id=f"service:{token_id}")

    if effective_agent_id:
        return TokenBuilder().attenuate(
            token,
            agent_id=effective_agent_id,
        )

    return token


def extract_last_user_msg(input_data: GraphInput) -> str:
    """Extract the last user message from execution input."""
    messages_raw = input_data.get("messages")
    if is_object_list(messages_raw):
        for message in reversed(messages_raw):
            role = extract_message_role(message)
            if role == "user":
                content = extract_message_content(message)
                if content:
                    return content
    input_value = input_data.get("input")
    if isinstance(input_value, str):
        return input_value
    return ""


_GRAPH_STATE_KEYS = frozenset(
    {
        "final_output",
        "intermediate_results",
        "messages",
        "dynamic",
        "_steps",
        "_last_node",
        "_raw_output",
        "structured_output",
    }
)


def extract_state_update_from_chain_output(output: object) -> dict[str, object]:
    """Normalize ``on_chain_end`` output into a LangGraph state update dict.

    Uses ``is_object_dict`` — not ``is_json_dict`` — because node outputs may
    contain SQL row values (``datetime``, ``Decimal``, …) that are struct-sanitized
    later at the gRPC boundary.
    """
    if not is_object_dict(output):
        return {}
    update_raw = output.get("update")
    if is_object_dict(update_raw):
        return {str(key): value for key, value in update_raw.items()}
    if len(output) == 1:
        sole_key = next(iter(output))
        sole_val = output[sole_key]
        if is_object_dict(sole_val):
            if _GRAPH_STATE_KEYS.intersection(sole_val.keys()):
                return {str(key): value for key, value in sole_val.items()}
    return {str(key): value for key, value in output.items()}


async def invoke_graph(
    graph: RunnableGraph,
    execution_input: GraphInput,
    *,
    run_config: RunnableConfig,
) -> object:
    """Call ``graph.ainvoke`` when the resolved registry object is a compiled graph."""
    result = graph.ainvoke(execution_input, config=run_config)
    if inspect.isawaitable(result):
        return await result
    return result


async def iter_graph_events(
    graph: RunnableGraph,
    execution_input: GraphInput,
    *,
    run_config: RunnableConfig,
) -> AsyncIterator[dict[str, object]]:
    """Stream LangGraph events from a resolved registry graph."""
    async for raw_event in graph.astream_events(execution_input, config=run_config, version="v2"):
        yield raw_event


def resolve_graph(
    graph_name: str, tenant_id: str, project_graphs: ProjectGraphMap
) -> ResolvedGraph:
    """Resolve graph name and build a compiled graph from the registry."""
    logger.info(
        "resolve_graph: name=%s, tenant=%s, project_graphs keys=%s, registry=%s",
        graph_name,
        tenant_id,
        list(project_graphs.keys()),
        graph_registry.has(graph_name),
    )
    if not graph_registry.has(graph_name):
        # ── Phase 8 Multi-Graph resolution ──
        # If graph_name has logic (e.g. "{project_id}:{graph_name}")
        # We need to map the provided graph_name string to parts.
        parts = graph_name.split(":", 1) if graph_name else [graph_name]
        project_id = parts[0]

        graph_entry = project_graphs.get(project_id)
        if not graph_entry:
            raise ConfigurationError(
                message=f"No graph registered for agent '{graph_name}' (Tenant: {tenant_id})",
            )

        if len(parts) > 1:
            # Explicit sub-graph requested (e.g. "my_project:gardener")
            sub_graph = parts[1]
            resolved = graph_entry.get(sub_graph)
            if not resolved:
                raise ConfigurationError(
                    message=f"Graph '{sub_graph}' not registered in multi-graph map for '{project_id}'",
                )
            graph_name = resolved
        else:
            # Fallback to default
            resolved = graph_entry.get("default")
            if not resolved:
                raise ConfigurationError(
                    message=f"No default graph configured in multi-graph map for '{project_id}'",
                )
            graph_name = resolved

        if not graph_registry.has(graph_name):
            raise ConfigurationError(
                message=f"Graph '{graph_name}' resolved but not found in registry",
            )

    from contextunity.router.core.registry import as_runnable_graph_factory

    factory = as_runnable_graph_factory(graph_registry.get(graph_name))
    compiled = factory.build()
    if not is_runnable_graph(compiled):
        raise ConfigurationError(
            message=f"Graph '{graph_name}' builder returned a non-runnable graph"
        )
    return ResolvedGraph(graph_name, compiled)


def _redact_sensitive_keys(data: JsonValue) -> JsonValue:
    """Recursively redact large or sensitive values for observability persistence.

    Domain considerations:
    - contextunity.brain (Storage/UI): Redact bulky texts (`_prompt`, `_prompts`, `schema_description`)
      to prevent DB bloat and ensure fast rendering in contextunity.view traces.
    - contextunity.shield (Security): Redact cryptographic artifacts (`_signature`) from
      telemetry to minimize exposure of verification hashes outside the execution boundary.
    """
    if is_json_dict(data):
        redacted: JsonDict = {}
        for key, value in data.items():
            if key.endswith(("_signature", "_prompt", "_prompts")) or key in (
                "schema_description",
                "model_secret_ref",
                "prompt_signature",
                "prompt_variants_ref",
            ):
                redacted[key] = "**REDACTED**"
            else:
                redacted[key] = _redact_sensitive_keys(value)
        return redacted
    if isinstance(data, list):
        return [_redact_sensitive_keys(item) for item in data]
    return data


_LANGFUSE_KEYS = (
    "langfuse_project_id",
    "langfuse_public_key",
    "langfuse_secret_key",
    "langfuse_host",
    "langfuse_trace_id",
    "langfuse_trace_url",
)


def _redact_project_config(config: RegisteredProjectConfig) -> RegisteredProjectConfig:
    """Redact sensitive manifest fields while preserving registration shape."""
    result: RegisteredProjectConfig = {}
    policy = config.get("policy")
    if is_json_dict(policy):
        cleaned = _redact_sensitive_keys(policy)
        if is_json_dict(cleaned):
            result["policy"] = cleaned
    services = config.get("services")
    if is_json_dict(services):
        cleaned = _redact_sensitive_keys(services)
        if is_json_dict(cleaned):
            result["services"] = cleaned
    graph = config.get("graph")
    if is_json_dict(graph):
        redacted_graph: RegisteredGraphMap = {}
        for graph_key, graph_entry in graph.items():
            if not is_json_dict(graph_entry):
                continue
            cleaned_entry = _redact_sensitive_keys(graph_entry)
            if is_json_dict(cleaned_entry):
                redacted_graph[graph_key] = cleaned_entry
        if redacted_graph:
            result["graph"] = redacted_graph
    tools = config.get("tools")
    if isinstance(tools, list):
        redacted_tools: list[RegisteredToolEntry] = []
        for entry in tools:
            if not is_json_dict(entry):
                continue
            tool_entry: RegisteredToolEntry = {}
            name = entry.get("name")
            if isinstance(name, str):
                tool_entry["name"] = name
            tool_type = entry.get("type")
            if isinstance(tool_type, str):
                tool_entry["type"] = tool_type
            description = entry.get("description")
            if isinstance(description, str):
                tool_entry["description"] = description
            tool_config = entry.get("config")
            if is_json_dict(tool_config):
                cleaned_config = _redact_sensitive_keys(tool_config)
                if is_json_dict(cleaned_config):
                    tool_entry["config"] = cleaned_config
            if tool_entry:
                redacted_tools.append(tool_entry)
        if redacted_tools:
            result["tools"] = redacted_tools
    nodes = config.get("nodes")
    if isinstance(nodes, list):
        result["nodes"] = nodes
    return result


def _clean_for_trace(metadata: ExecutionMetadata) -> ExecutionMetadata:
    """Prepare execution metadata for Brain trace persistence.

    Domain boundary: contextunity.router (Execution) -> contextunity.brain (Observability).
    The `project_config` contains heavy manifest definitions needed by contextunity.router
    at runtime (e.g., SecureNode verification, tool resolution).
    Before sending this payload over gRPC to Brain, we redact payload-heavy and
    security-sensitive properties, preserving only the structural telemetry data.
    """
    clean_meta = copy_execution_metadata(metadata)
    config = clean_meta.get("project_config")

    if config is not None:
        clean_meta["project_config"] = _redact_project_config(config)

    # 1. Target both the root metadata and the nested config for cleanup/sorting
    nested_cfg = clean_meta.get("project_config")

    is_langfuse_enabled = clean_meta.get("langfuse_enabled", False)

    # Base UI keys: always show whether it's enabled or not
    ui_bottom_keys = ["langfuse_enabled"]

    if is_langfuse_enabled:
        # Include all actual Langfuse telemetry details
        ui_bottom_keys.extend(_LANGFUSE_KEYS)

    # Ensure provenance is strictly at the bottom
    ui_bottom_keys.append("_inner_provenance")

    def _apply_trace_layout(target: JsonDict) -> None:
        if not is_langfuse_enabled:
            for k in _LANGFUSE_KEYS:
                _ = target.pop(k, None)
        for k in ui_bottom_keys:
            if k in target:
                target[k] = target.pop(k)

    if not is_langfuse_enabled:
        for k in _LANGFUSE_KEYS:
            _ = clean_meta.pop(k, None)
    for k in ui_bottom_keys:
        if k in clean_meta:
            clean_meta[k] = clean_meta.pop(k)
    if is_json_dict(nested_cfg):
        _apply_trace_layout(nested_cfg)

    return clean_meta


def prepare_execution(
    params: ExecuteAgentPayload,
    tenant_id: str,
    token: ContextToken,
    tenant_configs: ProjectConfigMap,
) -> ExecutionContext:
    """Prepare execution input, metadata, and callbacks."""
    project_config = get_registered_project_config(
        tenant_configs,
        tenant_id,
        project_id=project_id_from_agent_id(params.agent_id),
    )

    # Validate token tenant intersection with project allowed_tenants
    # to prevent multi-tenant token scope confusion when projects have
    # distinct ``allowed_tenants`` in the manifest.
    effective_tenant_id = _intersect_tenant_with_project(token, tenant_id, project_config)
    # Use the intersected tenant (may differ if token has multiple valid tenants)
    tenant_id = effective_tenant_id

    execution_input = dict(params.input)
    metadata_raw = execution_input.get("metadata")
    metadata = (
        execution_metadata_from_payload(metadata_raw)
        if is_json_dict(metadata_raw)
        else ExecutionMetadata()
    )

    metadata["tenant_id"] = tenant_id
    metadata["agent_id"] = params.agent_id or ""
    metadata["platform"] = str(metadata.get("platform") or "grpc")

    # Propagate tool allow/deny lists from graph input into trace metadata
    for _tools_field in ("allowed_tools", "denied_tools"):
        if _tools_field not in metadata:
            _tools_val = execution_input.get(_tools_field)
            if is_object_list(_tools_val):
                metadata[_tools_field] = [item for item in _tools_val if isinstance(item, str)]

    # Fully intact config for the graph runtime
    metadata["project_config"] = project_config

    execution_input["metadata"] = metadata

    effective_user_id = getattr(token, "user_id", None) if token else None

    if effective_user_id:
        metadata["user_id"] = effective_user_id

    # Build per-request Langfuse context from project-supplied metadata
    langfuse_ctx = LangfuseRequestCtx.from_metadata(dict(metadata))

    callbacks = get_langfuse_callbacks(
        session_id=str(metadata.get("session_id", "")),
        user_id=effective_user_id,
        platform=str(metadata.get("platform", "")),
        langfuse_ctx=langfuse_ctx,
    )

    # Ensure the actual enabled state is stored back in metadata
    # so contextunity.view can display the 'Langfuse (Disabled)' badge
    metadata["langfuse_enabled"] = langfuse_ctx.enabled

    auto_tracer = BrainAutoTracer()
    callbacks.append(auto_tracer)

    return ExecutionContext(
        execution_input, metadata, effective_user_id, callbacks, auto_tracer, langfuse_ctx
    )


def _token_usage_floats(raw: object) -> dict[str, float]:
    """Extract numeric token usage fields from graph state."""
    if not is_json_dict(raw):
        return {}
    usage: dict[str, float] = {}
    for key in ("input_tokens", "output_tokens", "total_cost"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            usage[key] = float(value)
    return usage


def merge_token_usage(auto_tracer: BrainAutoTracer, state: GraphResult) -> TotalTokenUsage:
    """Merge token usage from AutoTracer callbacks and graph state."""
    auto_tokens = auto_tracer.get_token_usage()
    state_tokens = _token_usage_floats(state.get("_token_usage"))
    return {
        "input_tokens": int(
            max(auto_tokens.get("input_tokens", 0), state_tokens.get("input_tokens", 0))
        ),
        "output_tokens": int(
            max(auto_tokens.get("output_tokens", 0), state_tokens.get("output_tokens", 0))
        ),
        "total_cost": float(
            max(auto_tokens.get("total_cost", 0.0), state_tokens.get("total_cost", 0.0))
        ),
    }


def extract_answer(result: GraphResult) -> str:
    """Extract final answer content from graph result."""
    messages_raw = result.get("messages")
    if not is_object_list(messages_raw) or not messages_raw:
        return ""
    last_message = messages_raw[-1]
    content = extract_message_content(last_message)
    return content[:5000] if content else ""


def serialize_messages(result: GraphResult) -> GraphResult:
    """Convert LangChain BaseMessage objects in ``result["messages"]`` to plain dicts.

    LangGraph returns BaseMessage instances that can't cross the gRPC protobuf
    boundary. This mutates the result in-place (and returns it for chaining).

    The type signature is ``GraphResult → GraphResult`` because both BaseMessage
    and dict are ``object`` — the transformation is invisible at the type level.
    """
    messages_raw = result.get("messages")
    if not is_object_list(messages_raw):
        return result
    serialized: list[object] = []
    for message in messages_raw:
        serialized.append(serialize_message_object(message))
    result["messages"] = serialized
    return result


def _build_security_flags(
    guard_result: ShieldCheckResult | None,
    provenance: list[str],
    error: str,
) -> SecurityFlags:
    """Build security context flags for trace persistence (contextunity.view badges)."""
    flags: SecurityFlags = {}

    if guard_result is not None:
        flags["shield_enabled"] = guard_result.mode == "shield"
        flags["shield_mode"] = guard_result.mode

    def _provenance_indicates_pii(s: str) -> bool:
        if s.startswith("pii:"):
            return True
        # secure_node records router privacy scope as privacy:pii_applied
        return s == "privacy:pii_applied" or s.startswith("privacy:pii")

    flags["pii_masking_enabled"] = any(_provenance_indicates_pii(s) for s in provenance)

    try:
        from contextunity.core.config import get_core_config as _get_shared_cfg

        cfg = _get_shared_cfg()
        redis_url = cfg.redis.url or ""
        secret_key = cfg.security.redis_secret_key.strip()
        redis_encrypted = bool(secret_key) and secret_key.lower() not in ("false", "0", "no", "")
        flags["redis_enabled"] = bool(cfg.redis.enabled and redis_url)
        flags["redis_tls"] = redis_url.startswith("rediss://")
        flags["redis_encrypted"] = redis_encrypted
    except Exception:  # graceful-degrade: trace error must not crash execution
        pass  # Redis flags simply absent — view dashboard handles missing keys

    if error:
        flags["error"] = error

    return flags


async def log_execution_trace(
    *,
    auto_tracer: BrainAutoTracer,
    result: GraphResult,
    token: ContextToken | None,
    tenant_id: str,
    params: ExecuteAgentPayload,
    metadata: ExecutionMetadata,
    effective_user_id: str | None,
    graph_name: str,
    wall_ms: int,
    last_user_msg: str,
    guard_result: ShieldCheckResult | None,
    execution_input: GraphInput,
    stream: bool = False,
    error: str = "",
) -> None:
    """Log execution trace to Brain via brain_trace_tools.

    Calls the underlying async function directly (not via LangChain ainvoke)
    because this is an infrastructure call — LangChain schema validation on
    TypedDict annotations (ToolCallSummary, TotalTokenUsage) causes
    ``'dict' object is not callable`` errors.

    The ``stream`` flag controls episodic-memory recording: stream runs are
    incremental (one trace per progress tick) so we skip the extra gRPC
    write to Brain's episodic store; unary runs always record the episode.
    """
    try:
        from contextunity.router.modules.tools.brain_trace_tools import (
            log_execution_trace as _trace_fn,
        )

        token_usage = merge_token_usage(auto_tracer, result)
        answer_content = extract_answer(result)

        base_prov: list[str] = (
            list(token.provenance) if token and hasattr(token, "provenance") else []
        )
        raw_prov = metadata.get("_inner_provenance")
        inner_prov: list[str] = []
        if raw_prov:
            inner_prov = [entry for entry in raw_prov if isinstance(entry, str)]

        provenance_flat = base_prov + inner_prov

        steps_list = auto_tracer.get_nested_steps()
        if error:
            metadata["graph_error"] = error
            from contextunity.router.modules.observability.contracts import SpanDict

            failure_step: SpanDict = {
                "step": len(steps_list),
                "iteration": 0,
                "type": "tool_result",
                "tool": "graph_failure",
                "status": "error",
                "result": error,
            }
            steps_list.append(failure_step)

        sec_flags = _build_security_flags(guard_result, inner_prov, error)

        trace_user_id = effective_user_id or "platform"
        trace_metadata = _clean_for_trace(metadata)
        trace_metadata["user_id"] = trace_user_id

        messages_raw = execution_input.get("messages")
        msg_count = len(messages_raw) if is_object_list(messages_raw) else 0

        # _trace_fn is a LangChain @tool (StructuredTool). We call the raw
        # coroutine directly to skip schema validation that chokes on TypedDict.
        from collections.abc import Awaitable

        _coro_obj = getattr(_trace_fn, "coroutine", None)
        if not callable(_coro_obj):
            logger.warning("Trace logging skipped: brain_trace_tools coroutine unavailable")
            return
        trace_coro = _coro_obj
        trace_result = trace_coro(
            tenant_id=tenant_id,
            agent_id=params.agent_id or "",
            session_id=str(metadata.get("session_id", "")),
            user_id=trace_user_id,
            graph_name=graph_name,
            tool_calls=auto_tracer.get_tool_calls_summary(),
            token_usage=token_usage,
            timing_ms=wall_ms,
            steps=steps_list,
            platform=str(metadata.get("platform", "")),
            model_key=str(metadata.get("model_key", "")),
            iterations=1,
            message_count=msg_count,
            user_query=last_user_msg,
            final_answer=answer_content,
            metadata=trace_metadata,
            provenance=provenance_flat,
            security_flags=sec_flags,
            # Stream runs are incremental — skip episodic memory on every
            # progress tick to avoid extra gRPC writes per event. Unary runs
            # always record the episode.
            record_episode=not stream,
        )
        if isinstance(trace_result, Awaitable):
            await trace_result
    except Exception as tr_err:  # graceful-degrade: trace error must not crash execution
        logger.warning("Trace logging failed: %s", tr_err, exc_info=True)


def _entry_max_retries(entry: dict[str, object]) -> int | None:
    """Read ``max_retries`` from a graph entry's top level or nested ``config``."""
    from contextunity.router.cortex.compiler.types import coerce_manifest_int

    top_level = coerce_manifest_int(entry.get("max_retries"))
    if top_level is not None:
        return top_level
    nested = entry.get("config")
    if is_object_dict(nested):
        return coerce_manifest_int(nested.get("max_retries"))
    return None


MAX_MANIFEST_RECURSION_LIMIT = 100


def resolve_recursion_limit(
    project_config: RegisteredProjectConfig | None,
    registry_graph_name: str,
    graph: RunnableGraph,
) -> int | None:
    """Derive a runtime ``recursion_limit`` from the manifest's ``config.max_retries``.

    A cyclic graph declares ``config.max_retries`` as its retry budget. LangGraph
    bounds cycles via ``recursion_limit`` (max super-steps) rather than a dedicated
    retry counter, so the declared budget must be projected onto the runtime limit.
    Without this, cyclic graphs run to LangGraph's default of 25 super-steps instead
    of terminating after ``max_retries + 1`` passes through the graph.

    The limit is ``(max_retries + 1) * node_count`` (plus a small buffer) so each
    node may execute up to ``max_retries + 1`` times. Returns ``None`` when no
    ``max_retries`` is declared, leaving LangGraph's default in force.
    """
    if not is_object_dict(project_config):
        return None
    graph_map = project_config.get("graph")
    if not is_object_dict(graph_map):
        return None
    # Registry names are ``project:<id>:<graph_key>``; entries key on ``<graph_key>``.
    graph_key = registry_graph_name.rsplit(":", 1)[-1]
    entry = graph_map.get(graph_key)
    if not is_object_dict(entry):
        candidates = [value for value in graph_map.values() if is_object_dict(value)]
        if len(candidates) != 1:
            return None
        entry = candidates[0]
    max_retries = _entry_max_retries(entry)
    if max_retries is None:
        return None
    raw_nodes = getattr(graph, "nodes", None)
    node_count = len(raw_nodes) if is_object_dict(raw_nodes) else 0
    node_count = max(node_count, 1)
    limit = (max_retries + 1) * node_count + 2
    return min(limit, MAX_MANIFEST_RECURSION_LIMIT)


def build_run_config(
    user_config: GraphRunConfigInput | None,
    callbacks: list[BaseCallbackHandler],
    *,
    default_recursion_limit: int | None = None,
) -> RunnableConfig:
    """Build a ``RunnableConfig`` from validated execution payload config and callbacks.

    ``default_recursion_limit`` (derived from the manifest's ``max_retries``) is
    the graph's upper bound. Caller-supplied ``recursion_limit`` may lower the
    limit for a request, but cannot raise it above the manifest budget.
    """
    cfg: RunnableConfig = {"callbacks": callbacks}
    if default_recursion_limit is not None:
        cfg["recursion_limit"] = default_recursion_limit
    if not user_config:
        return cfg
    if tags := user_config.get("tags"):
        cfg["tags"] = tags
    if run_name := user_config.get("run_name"):
        cfg["run_name"] = run_name
    if max_concurrency := user_config.get("max_concurrency"):
        cfg["max_concurrency"] = max_concurrency
    if recursion_limit := user_config.get("recursion_limit"):
        if default_recursion_limit is not None:
            cfg["recursion_limit"] = min(recursion_limit, default_recursion_limit)
        else:
            cfg["recursion_limit"] = recursion_limit
    if metadata := user_config.get("metadata"):
        cfg["metadata"] = dict(metadata)
    if configurable := user_config.get("configurable"):
        cfg["configurable"] = dict(configurable)
    return cfg


__all__ = [
    "_intersect_tenant_with_project",
    "_resolve_tenant_id",
    "build_execution_token",
    "build_run_config",
    "extract_answer",
    "extract_last_user_msg",
    "extract_state_update_from_chain_output",
    "get_registered_project_config",
    "invoke_graph",
    "iter_graph_events",
    "log_execution_trace",
    "merge_graph_state_update",
    "merge_token_usage",
    "prepare_execution",
    "resolve_graph",
    "resolve_recursion_limit",
    "serialize_messages",
]
