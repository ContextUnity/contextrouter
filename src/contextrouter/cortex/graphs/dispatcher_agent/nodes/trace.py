"""Trace reflection node — logs execution timeline to Brain.

Uses the universal ``log_execution_trace`` tool so the dashboard rendering
is consistent across all graphs (dispatcher, sql_analytics, custom).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from contextrouter.cortex.graphs.dispatcher_agent.state import DispatcherState

logger = logging.getLogger(__name__)


async def reflect_dispatcher(state: DispatcherState) -> dict[str, Any]:
    """Log a detailed execution trace at the end of the dispatcher flow.

    Captures a step-by-step timeline of every LLM call, tool invocation,
    and tool result so the dashboard can show the full execution story.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from contextrouter.modules.tools import get_tool

    trace_tool = get_tool("log_execution_trace")
    if not trace_tool:
        logger.warning("log_execution_trace tool not found, skipping trace")
        return {}

    pipeline_start = state.get("_start_ts") or time.monotonic()
    tenant_id = state.get("tenant_id", "default")
    session_id = state.get("session_id", "")
    user_id = state.get("metadata", {}).get("user_id", "anonymous")
    platform = state.get("platform", "")

    meta = state.get("metadata") or {}
    model_key = meta.get("model_key", "")
    current_iteration = state.get("iteration", 0)

    # ── Reconstruct timeline from messages ──
    steps: list[dict[str, Any]] = []
    tool_calls_summary: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    step_idx = 0
    token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    all_messages = state.get("messages", [])
    user_query = ""
    final_answer = ""

    # Trim to this turn only — find last HumanMessage
    messages = all_messages
    if len(all_messages) > 2:
        last_human_idx = -1
        for idx_m in range(len(all_messages) - 1, -1, -1):
            if isinstance(all_messages[idx_m], HumanMessage):
                last_human_idx = idx_m
                break
        if last_human_idx > 0:
            messages = all_messages[last_human_idx:]
            logger.debug(
                "Reflect: trimmed %d historical msgs, logging %d from this turn",
                last_human_idx,
                len(messages),
            )

    for msg in messages:
        if isinstance(msg, SystemMessage):
            steps.append(
                {
                    "step": step_idx,
                    "iteration": 0,
                    "type": "system",
                    "content": str(msg.content)[:1000],
                }
            )
            step_idx += 1

        elif isinstance(msg, HumanMessage):
            content = str(msg.content)
            if not user_query:
                user_query = content
            steps.append(
                {
                    "step": step_idx,
                    "iteration": 0,
                    "type": "user",
                    "content": content[:3000],
                }
            )
            step_idx += 1

        elif isinstance(msg, AIMessage):
            resp_meta = getattr(msg, "response_metadata", None) or {}
            tu = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
            msg_tokens: dict[str, int] = {}
            if tu:
                msg_tokens["input"] = tu.get("prompt_tokens") or tu.get("input_tokens", 0) or 0
                msg_tokens["output"] = (
                    tu.get("completion_tokens") or tu.get("output_tokens", 0) or 0
                )
                token_usage["input_tokens"] += msg_tokens["input"]
                token_usage["output_tokens"] += msg_tokens["output"]

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_name = tc.get("name", "")
                    tc_id = tc.get("id", "")
                    tc_args = {str(k): str(v)[:3000] for k, v in (tc.get("args") or {}).items()}

                    tool_call_step = {
                        "step": step_idx,
                        "iteration": current_iteration,
                        "type": "tool_call",
                        "tool": tc_name,
                        "tool_call_id": tc_id,
                        "args": tc_args,
                        "tokens": msg_tokens,
                    }
                    steps.append(tool_call_step)
                    pending_tool_calls[tc_id] = tool_call_step

                    tool_calls_summary.append({"tool": tc_name, "args": tc_args})
                    step_idx += 1
            else:
                content = str(msg.content)
                final_answer = content
                steps.append(
                    {
                        "step": step_idx,
                        "iteration": current_iteration,
                        "type": "assistant",
                        "content": content[:5000],
                        "tokens": msg_tokens,
                    }
                )
                step_idx += 1

        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "")
            content = str(msg.content) if msg.content is not None else ""
            is_error = bool(getattr(msg, "additional_kwargs", {}).get("error"))
            tool_name = getattr(msg, "name", "") or pending_tool_calls.get(tc_id, {}).get(
                "tool", ""
            )

            steps.append(
                {
                    "step": step_idx,
                    "iteration": current_iteration,
                    "type": "tool_result",
                    "tool": tool_name,
                    "tool_call_id": tc_id,
                    "status": "error" if is_error else "ok",
                    "result": content[:10000],
                }
            )

            tool_calls_summary.append({"tool": tool_name, "status": "error" if is_error else "ok"})
            step_idx += 1

    timing_ms = int((time.monotonic() - pipeline_start) * 1000)

    # ── Security context ──
    security_flags = list(state.get("security_flags", []))

    access_token = state.get("access_token")
    token_info: dict[str, Any] = {}
    if access_token is not None:
        token_info = {
            "token_id": getattr(access_token, "token_id", ""),
            "user_id": getattr(access_token, "user_id", ""),
            "agent_id": getattr(access_token, "agent_id", ""),
            "user_namespace": getattr(access_token, "user_namespace", "default"),
            "permissions": list(getattr(access_token, "permissions", ())),
            "allowed_tenants": list(getattr(access_token, "allowed_tenants", ())),
        }
        if token_info.get("user_id"):
            user_id = token_info["user_id"]

    agent_id = (
        token_info.get("agent_id")
        or (f"router:{tenant_id}" if tenant_id != "default" else None)
        or "router:dispatcher"
    )

    graph_name = meta.get("graph_name", "dispatcher")
    provenance = [f"agent:{agent_id}"]
    for tc in tool_calls_summary:
        provenance.append(f"tool:{tc['tool']}")
    provenance.append(f"router:{graph_name}:execute")

    # ── Log via universal tool ──
    try:
        await trace_tool.ainvoke(
            {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "graph_name": graph_name,
                "tool_calls": tool_calls_summary,
                "token_usage": token_usage,
                "timing_ms": timing_ms,
                # ── Rich fields ──
                "steps": steps,
                "platform": platform,
                "model_key": model_key,
                "iterations": current_iteration,
                "message_count": len(messages),
                "user_query": user_query[:2000],
                "final_answer": final_answer[:2000],
                "metadata": {
                    "total_messages": len(all_messages),
                    "allowed_tools": list(state.get("allowed_tools", [])),
                    "denied_tools": list(state.get("denied_tools", [])),
                    # Langfuse observability — injected by execution mixin
                    "langfuse_trace_id": meta.get("langfuse_trace_id", ""),
                    "langfuse_trace_url": meta.get("langfuse_trace_url", ""),
                },
                "provenance": provenance,
                "security_flags": {"events": security_flags, **token_info},
                "record_episode": True,
            }
        )
        logger.info(
            "Logged dispatcher trace for session %s (%d steps)",
            session_id,
            len(steps),
        )
    except Exception as e:
        detail = e.details() if hasattr(e, "details") else str(e)
        code = e.code() if hasattr(e, "code") else ""
        logger.error("Brain trace failed [%s]: %s", code or type(e).__name__, detail)

    return {}


__all__ = ["reflect_dispatcher"]
