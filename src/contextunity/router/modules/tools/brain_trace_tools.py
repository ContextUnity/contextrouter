"""Brain Trace Tools for Router graphs.

Provides **universal** execution trace logging backed by contextunity.brain via SDK.
Any graph (dispatcher, sql_analytics, custom) can call ``log_execution_trace``
in its final node to persist a rich trace visible in contextunity.view dashboard.

Key contract:
    * ``tool_calls`` — lightweight summary list (tool name + status).
    * ``steps`` — **detailed** step-by-step timeline with per-step timing,
      request/result data, and token usage.  contextunity.view renders this as the
      "Graph Journey" section and the "Conversation Flow" tab.
    * The tool also records an **episodic memory** entry so the execution
      appears in the Memory Discovery section.

Uses the same BrainClient singleton as brain_memory_tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextunity.core.types import ContextUnitPayload, JsonDict, JsonValue, is_json_dict

if TYPE_CHECKING:
    from contextunity.core.sdk import BrainClient

from contextunity.core import get_contextunit_logger

from contextunity.router.langchain_boundaries import tool
from contextunity.router.modules.tools import register_tool
from contextunity.router.modules.tools.schemas import (
    EpisodeResult,
    ToolCallSummary,
    TotalTokenUsage,
    TraceResult,
)

logger = get_contextunit_logger(__name__)

# Per-tenant BrainClient cache — each tenant gets a properly scoped client
_brain_clients: dict[str, BrainClient] = {}
_brain_client_token_ids: dict[str, str] = {}


def _auth_token_id(token: object | None) -> str:
    if token is None:
        return ""
    token_id = getattr(token, "token_id", None)
    return token_id if isinstance(token_id, str) else ""


def _get_brain_client(tenant_id: str) -> BrainClient:
    """Get or create BrainClient for a specific tenant.

    Uses the verified ``ContextToken`` from the current gRPC auth context.
    Recreates the client when the active token changes (rotation / new user).
    """
    from contextunity.router.modules.tools.auth_context import resolve_tool_context_token

    auth_token = resolve_tool_context_token()
    token_key = _auth_token_id(auth_token)
    cached_key = _brain_client_token_ids.get(tenant_id)
    if tenant_id in _brain_clients and cached_key == token_key:
        return _brain_clients[tenant_id]

    from contextunity.core.sdk import BrainClient

    _brain_clients[tenant_id] = BrainClient(
        tenant_id=tenant_id,
        token=auth_token,
    )
    _brain_client_token_ids[tenant_id] = token_key
    return _brain_clients[tenant_id]


# ============================================================================
# Tool: log_execution_trace
# ============================================================================


def _tool_calls_wire(calls: list[ToolCallSummary]) -> list[dict[str, object]]:
    return [dict(tc) for tc in calls]


def _steps_wire(steps: list[JsonDict] | None) -> list[dict[str, object]]:
    """Serialize Graph Journey steps for Brain/view (UUID/datetime safe)."""
    if not steps:
        return []

    from contextunity.core.types import is_object_dict, is_object_list

    from contextunity.router.service.security import sanitize_for_struct

    def _walk(span: object) -> dict[str, object] | None:
        if not is_object_dict(span):
            return None
        out: dict[str, object] = {}
        for key, value in span.items():
            if key == "children" and is_object_list(value):
                children = [child for item in value if (child := _walk(item)) is not None]
                out[key] = children
            else:
                out[key] = sanitize_for_struct(value)
        return out

    wired: list[dict[str, object]] = []
    for step in steps:
        item = _walk(dict(step))
        if item is not None:
            wired.append(item)
    return wired


def _trace_metadata(
    *,
    metadata: JsonDict | None,
    model_key: str,
    platform: str,
    iterations: int,
    message_count: int,
    steps: list[JsonDict] | None,
) -> JsonDict:
    from contextunity.core.types import is_json_value

    from contextunity.router.service.security import sanitize_for_struct

    wire: JsonDict = {}
    if metadata:
        for key, value in metadata.items():
            # Steps are supplied from the tracer below; avoid caller-provided duplicates.
            if key == "steps":
                continue
            sanitized = sanitize_for_struct(value)
            if is_json_value(sanitized):
                wire[key] = sanitized

    if "model_key" not in wire:
        wire["model_key"] = model_key
    if "platform" not in wire:
        wire["platform"] = platform
    if "iterations" not in wire:
        wire["iterations"] = iterations
    if "message_count" not in wire:
        wire["message_count"] = message_count

    step_wire = _steps_wire(steps)
    if step_wire:
        steps_wire: list[JsonValue] = []
        for step in step_wire:
            if is_json_dict(step):
                steps_wire.append(step)
        if steps_wire:
            wire["steps"] = steps_wire

    return wire


def _token_usage_json(usage: TotalTokenUsage) -> JsonDict:
    raw = dict(usage)
    return raw if is_json_dict(raw) else {}


def _episode_metadata(
    *,
    agent_id: str,
    trace_id: str,
    user_query: str,
    final_answer: str,
    tool_calls: list[ToolCallSummary],
    token_usage: TotalTokenUsage,
    timing_ms: int,
    platform: str,
    iterations: int,
) -> JsonDict:
    tc = tool_calls or []
    total_tokens = token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0)
    wire: ContextUnitPayload = {
        "event_type": "agent_execution",
        "agent_id": agent_id,
        "trace_id": trace_id,
        "status": "completed",
        "user_query": user_query[:2000],
        "final_answer": final_answer[:2000],
        "tool_calls": _tool_calls_wire(tc),
        "token_usage": _token_usage_json(token_usage),
        "duration_ms": timing_ms,
        "tokens_used": total_tokens,
        "platform": platform,
        "iterations": iterations,
    }
    return wire if is_json_dict(wire) else {}


@tool
async def log_execution_trace(
    tenant_id: str,
    agent_id: str,
    session_id: str,
    user_id: str | None,
    graph_name: str,
    tool_calls: list[ToolCallSummary],
    token_usage: TotalTokenUsage,
    timing_ms: int,
    steps: list[JsonDict] | None = None,
    platform: str = "",
    model_key: str = "",
    iterations: int = 0,
    message_count: int = 0,
    user_query: str = "",
    final_answer: str = "",
    metadata: JsonDict | None = None,
    provenance: list[str] | None = None,
    security_flags: JsonDict | None = None,
    record_episode: bool = True,
) -> TraceResult:
    """Log a full execution trace to contextunity.brain for observability.

    Call this at the end of any graph execution to persist the trace.
    The trace will be visible in contextunity.view dashboard with full
    Graph Journey, Conversation Flow, per-tool timing, and Memory Discovery.

    Args:
        tenant_id: Tenant identifier for isolation.
        agent_id: Agent/graph identifier (e.g. "contextmed:chat").
        session_id: Session identifier for grouping traces.
        user_id: User identifier.
        graph_name: Name of the graph that was executed.
        tool_calls: List of tool call summaries (tool name, status).
        token_usage: Total token usage dict (input_tokens, output_tokens).
        timing_ms: Total execution time in milliseconds.
        steps: Detailed step-by-step timeline with per-step timing,
            request/result data.  Each step should have at minimum:
            ``{"tool": "...", "status": "ok", "timing_ms": 123}``.
            Optional fields: ``request``, ``result``, ``tokens``.
        platform: Platform identifier (e.g. "contextmed", "grpc").
        model_key: Model used for LLM calls (e.g. "openai/gpt-5-mini").
        iterations: Number of retry/iteration loops.
        message_count: Total messages exchanged.
        user_query: Original user question/task.
        final_answer: Agent's final response summary.
        metadata: Optional additional metadata.
        provenance: Optional provenance chain.
        security_flags: Optional security context.
        record_episode: If True, also record an episodic memory entry.

    Returns:
        Dict with trace_id and success status.
    """
    try:
        # ── Build rich metadata ──
        # Merge caller-provided metadata with dashboard-required fields
        full_metadata = _trace_metadata(
            metadata=metadata,
            model_key=model_key,
            platform=platform,
            iterations=iterations,
            message_count=message_count,
            steps=steps,
        )

        brain = _get_brain_client(tenant_id)

        trace_id = await brain.log_trace(
            tenant_id=tenant_id,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            graph_name=graph_name,
            tool_calls=_tool_calls_wire(tool_calls) if tool_calls else None,
            token_usage=_token_usage_json(token_usage),
            timing_ms=timing_ms,
            security_flags=security_flags or {},
            metadata=full_metadata,
            provenance=provenance or [f"agent:{agent_id}"],
        )

        logger.info(
            "Logged trace %s for %s/%s (%dms, %d steps, %d tool_calls)",
            trace_id,
            tenant_id,
            graph_name,
            timing_ms,
            len(steps or []),
            len(tool_calls),
        )

        # ── Episodic memory ──
        if record_episode:
            try:
                tc = tool_calls or []
                task_summary = (user_query[:500] + "...") if len(user_query) > 500 else user_query
                outcome_summary = (
                    (final_answer[:500] + "...") if len(final_answer) > 500 else final_answer
                )

                episode_content = (
                    f"User Task: {task_summary or 'No human query found'}\n\n"
                    f"Outcome: {outcome_summary or 'Agent completed with tool calls only'}\n\n"
                    f"Stats: {len(tc)} tool calls, {timing_ms}ms"
                )

                _ = await brain.add_episode(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    content=episode_content,
                    session_id=session_id,
                    metadata=_episode_metadata(
                        agent_id=agent_id,
                        trace_id=trace_id,
                        user_query=user_query,
                        final_answer=final_answer,
                        tool_calls=tc,
                        token_usage=token_usage,
                        timing_ms=timing_ms,
                        platform=platform,
                        iterations=iterations,
                    ),
                )
                logger.info("Recorded episode for trace %s", trace_id)
            except Exception as ep_err:
                logger.warning("Episode recording failed: %s", ep_err)

        return {
            "success": True,
            "trace_id": trace_id,
            "tenant_id": tenant_id,
            "graph_name": graph_name,
        }
    except Exception as e:
        detail_attr = getattr(e, "details", None)
        detail = str(detail_attr() if callable(detail_attr) else (detail_attr or e))
        code_attr = getattr(e, "code", None)
        code = str(code_attr() if callable(code_attr) else (code_attr or type(e).__name__))
        logger.error("Brain trace failed [%s]: %s", code, detail)
        return {"success": False, "error": detail}


# ============================================================================
# Tool: record_execution_episode
# ============================================================================


@tool
async def record_execution_episode(
    tenant_id: str,
    user_id: str | None,
    session_id: str,
    agent_id: str,
    user_query: str,
    final_answer: str,
    trace_id: str = "",
    tool_calls: list[ToolCallSummary] | None = None,
    token_usage: TotalTokenUsage | None = None,
    timing_ms: int = 0,
    platform: str = "",
) -> EpisodeResult:
    """Record an execution episode in Brain's episodic memory.

    Creates a rich episodic memory entry summarizing the agent's execution.
    This feeds into contextunity.view dashboard and can be recalled later.

    Note: ``log_execution_trace`` already records an episode automatically
    when ``record_episode=True`` (the default).  Use this tool only when
    you need to record an episode WITHOUT a full trace.

    Args:
        tenant_id: Tenant identifier.
        user_id: User identifier.
        session_id: Session identifier.
        agent_id: Agent/graph identifier.
        user_query: Original user question/task.
        final_answer: Agent's final response summary.
        trace_id: Optional trace_id from log_execution_trace.
        tool_calls: Optional tool call summaries.
        token_usage: Optional total token usage.
        timing_ms: Total execution time in ms.
        platform: Platform identifier (e.g. "contextmed", "grpc").

    Returns:
        Dict with episode_id and success status.
    """
    try:
        brain = _get_brain_client(tenant_id)
        task_summary = (user_query[:500] + "...") if len(user_query) > 500 else user_query
        outcome_summary = (final_answer[:500] + "...") if len(final_answer) > 500 else final_answer

        episode_content = (
            f"User Task: {task_summary or 'No human query found'}\n\n"
            f"Outcome: {outcome_summary or 'Agent completed with tool calls only'}\n\n"
            f"Stats: {len(tool_calls or [])} tool calls, {timing_ms}ms"
        )

        usage: TotalTokenUsage = token_usage or TotalTokenUsage()
        episode_id = await brain.add_episode(
            tenant_id=tenant_id,
            user_id=user_id,
            content=episode_content,
            session_id=session_id,
            metadata=_episode_metadata(
                agent_id=agent_id,
                trace_id=trace_id,
                user_query=user_query,
                final_answer=final_answer,
                tool_calls=tool_calls or [],
                token_usage=usage,
                timing_ms=timing_ms,
                platform=platform,
                iterations=0,
            ),
        )

        logger.info("Recorded episode %s for session %s", episode_id, session_id)
        return {
            "success": True,
            "episode_id": episode_id,
        }
    except Exception as e:
        detail_attr = getattr(e, "details", None)
        detail = str(detail_attr() if callable(detail_attr) else (detail_attr or e))
        code_attr = getattr(e, "code", None)
        code = str(code_attr() if callable(code_attr) else (code_attr or type(e).__name__))
        logger.error("Brain episode failed [%s]: %s", code, detail)
        return {"success": False, "error": detail}


# ============================================================================
# Auto-register tools
# ============================================================================

_TRACE_TOOLS = [
    log_execution_trace,
    record_execution_episode,
]

for _t in _TRACE_TOOLS:
    register_tool(_t)

logger.info("Registered %d Brain trace tools", len(_TRACE_TOOLS))

__all__ = [
    "log_execution_trace",
    "record_execution_episode",
]
