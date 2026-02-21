"""Brain Trace Tools for Router graphs.

Provides **universal** execution trace logging backed by ContextBrain via SDK.
Any graph (dispatcher, sql_analytics, custom) can call ``log_execution_trace``
in its final node to persist a rich trace visible in ContextView dashboard.

Key contract:
    * ``tool_calls`` — lightweight summary list (tool name + status).
    * ``steps`` — **detailed** step-by-step timeline with per-step timing,
      request/result data, and token usage.  ContextView renders this as the
      "Graph Journey" section and the "Conversation Flow" tab.
    * The tool also records an **episodic memory** entry so the execution
      appears in the Memory Discovery section.

Uses the same BrainClient singleton as brain_memory_tools.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from contextrouter.modules.tools import register_tool

logger = logging.getLogger(__name__)

# Reuse the same BrainClient singleton
_brain_client = None


def _get_brain_client():
    """Get or create BrainClient singleton."""
    global _brain_client
    if _brain_client is None:
        from contextcore.permissions import Permissions
        from contextcore.sdk import BrainClient
        from contextcore.tokens import ContextToken

        from contextrouter.core import get_core_config

        brain_host = get_core_config().brain.grpc_endpoint
        token = ContextToken(
            token_id="router-trace-service",
            permissions=(
                Permissions.TRACE_WRITE,
                Permissions.TRACE_READ,
                Permissions.MEMORY_WRITE,
                Permissions.MEMORY_READ,
            ),
        )
        _brain_client = BrainClient(host=brain_host, mode="grpc", token=token)
    return _brain_client


# ============================================================================
# Tool: log_execution_trace
# ============================================================================


@tool
async def log_execution_trace(
    tenant_id: str,
    agent_id: str,
    session_id: str,
    user_id: str,
    graph_name: str,
    tool_calls: list[dict[str, Any]],
    token_usage: dict[str, int],
    timing_ms: int,
    # ── Rich trace fields (used by ContextView dashboard) ──
    steps: list[dict[str, Any]] | None = None,
    platform: str = "",
    model_key: str = "",
    iterations: int = 0,
    message_count: int = 0,
    user_query: str = "",
    final_answer: str = "",
    # ── Standard fields ──
    metadata: dict[str, Any] | None = None,
    provenance: list[str] | None = None,
    security_flags: dict[str, Any] | None = None,
    record_episode: bool = True,
) -> dict[str, Any]:
    """Log a full execution trace to ContextBrain for observability.

    Call this at the end of any graph execution to persist the trace.
    The trace will be visible in ContextView dashboard with full
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
        brain = _get_brain_client()

        # ── Build rich metadata ──
        # Merge caller-provided metadata with dashboard-required fields
        full_metadata = dict(metadata or {})
        full_metadata.setdefault("model_key", model_key)
        full_metadata.setdefault("platform", platform)
        full_metadata.setdefault("iterations", iterations)
        full_metadata.setdefault("message_count", message_count)

        # Steps are the key field for ContextView's Graph Journey
        if steps:
            full_metadata["steps"] = steps

        trace_id = await brain.log_trace(
            tenant_id=tenant_id,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            graph_name=graph_name,
            tool_calls=tool_calls,
            token_usage=token_usage,
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

                total_tokens = token_usage.get("input_tokens", 0) + token_usage.get(
                    "output_tokens", 0
                )

                await brain.add_episode(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    content=episode_content,
                    session_id=session_id,
                    metadata={
                        "event_type": "agent_execution",
                        "agent_id": agent_id,
                        "trace_id": trace_id,
                        "status": "completed",
                        "user_query": user_query[:2000],
                        "final_answer": final_answer[:2000],
                        "tool_calls": tc,
                        "token_usage": token_usage,
                        "duration_ms": timing_ms,
                        "tokens_used": total_tokens,
                        "platform": platform,
                        "iterations": iterations,
                    },
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
        detail = e.details() if hasattr(e, "details") else str(e)
        code = e.code() if hasattr(e, "code") else ""
        logger.error("Brain trace failed [%s]: %s", code or type(e).__name__, detail)
        return {"success": False, "error": detail}


# ============================================================================
# Tool: record_execution_episode
# ============================================================================


@tool
async def record_execution_episode(
    tenant_id: str,
    user_id: str,
    session_id: str,
    agent_id: str,
    user_query: str,
    final_answer: str,
    trace_id: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    token_usage: dict[str, int] | None = None,
    timing_ms: int = 0,
    platform: str = "",
) -> dict[str, Any]:
    """Record an execution episode in Brain's episodic memory.

    Creates a rich episodic memory entry summarizing the agent's execution.
    This feeds into ContextView dashboard and can be recalled later.

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
        brain = _get_brain_client()

        task_summary = (user_query[:500] + "...") if len(user_query) > 500 else user_query
        outcome_summary = (final_answer[:500] + "...") if len(final_answer) > 500 else final_answer
        tc = tool_calls or []

        episode_content = (
            f"User Task: {task_summary or 'No human query found'}\n\n"
            f"Outcome: {outcome_summary or 'Agent completed with tool calls only'}\n\n"
            f"Stats: {len(tc)} tool calls, {timing_ms}ms"
        )

        episode_id = await brain.add_episode(
            tenant_id=tenant_id,
            user_id=user_id,
            content=episode_content,
            session_id=session_id,
            metadata={
                "event_type": "agent_execution",
                "agent_id": agent_id,
                "trace_id": trace_id,
                "status": "completed",
                "user_query": user_query[:2000],
                "final_answer": final_answer[:2000],
                "tool_calls": tc,
                "token_usage": token_usage or {},
                "duration_ms": timing_ms,
                "platform": platform,
            },
        )

        logger.info("Recorded episode %s for session %s", episode_id, session_id)
        return {
            "success": True,
            "episode_id": episode_id,
        }
    except Exception as e:
        detail = e.details() if hasattr(e, "details") else str(e)
        code = e.code() if hasattr(e, "code") else ""
        logger.error("Brain episode failed [%s]: %s", code or type(e).__name__, detail)
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
