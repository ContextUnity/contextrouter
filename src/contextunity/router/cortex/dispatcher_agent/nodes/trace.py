"""Trace reflection node — logs execution timeline to Brain.

Uses the universal ``log_execution_trace`` tool so the dashboard rendering
is consistent across all graphs (dispatcher, sql_analytics, custom).
"""

from __future__ import annotations

import time

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonDict, is_object_dict, is_object_list
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from contextunity.router.cortex.types import (
    GraphState,
    StateUpdate,
    TokenInfoDict,
    extract_message_content,
)
from contextunity.router.modules.tools.schemas import ToolCallSummary, TraceStep, TraceTokens

logger = get_contextunit_logger(__name__)


def _response_metadata(message: AIMessage) -> dict[str, object]:
    """Return response metadata as a plain object dictionary."""
    metadata_raw = getattr(message, "response_metadata", None)
    return dict(metadata_raw) if is_object_dict(metadata_raw) else {}


def _token_usage_from_metadata(metadata: dict[str, object]) -> TraceTokens | None:
    """Extract prompt/completion token counts from common provider metadata."""
    usage_raw = metadata.get("token_usage") or metadata.get("usage")
    if not is_object_dict(usage_raw):
        return None
    prompt_tokens = usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0
    completion_tokens = usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0
    return {"input": _to_int(prompt_tokens), "output": _to_int(completion_tokens)}


def _to_int(value: object) -> int:
    """Convert common numeric wire values to ``int``."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _tool_calls_from_ai_message(message: AIMessage) -> list[dict[str, object]]:
    """Return AI tool calls as normalized object dictionaries."""
    raw_tool_calls = getattr(message, "tool_calls", None)
    if not is_object_list(raw_tool_calls):
        return []
    tool_calls: list[dict[str, object]] = []
    for item in raw_tool_calls:
        if is_object_dict(item):
            tool_calls.append(dict(item))
    return tool_calls


def _string_dict(value: object) -> dict[str, str]:
    """Convert a plain object mapping to truncated string values."""
    if not is_object_dict(value):
        return {}
    return {str(key): str(item)[:3000] for key, item in value.items()}


def _json_args(value: object) -> JsonDict:
    """Convert tool-call args to JSON-safe values for trace summaries."""
    if not is_object_dict(value):
        return {}
    result: JsonDict = {}
    for key, item in value.items():
        result[str(key)] = str(item)[:3000]
    return result


def _tool_name_from_call(tool_call: dict[str, object]) -> str:
    """Read the tool name from a tool call payload."""
    name_raw = tool_call.get("name")
    if isinstance(name_raw, str) and name_raw:
        return name_raw
    legacy_name = tool_call.get("tool")
    if isinstance(legacy_name, str) and legacy_name:
        return legacy_name
    return ""


def _tool_call_id_from_call(tool_call: dict[str, object]) -> str:
    """Read the call id from a tool call payload."""
    raw_id = tool_call.get("id")
    return str(raw_id) if raw_id is not None else ""


def _tool_message_has_error(message: ToolMessage) -> bool:
    """Return whether a tool message is marked as an error."""
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if not is_object_dict(additional_kwargs):
        return False
    return bool(additional_kwargs.get("error"))


async def reflect_dispatcher(state: GraphState, config: RunnableConfig) -> StateUpdate:
    """Log a detailed execution trace at the end of the dispatcher flow.

    Captures a step-by-step timeline of every LLM call, tool invocation,
    and tool result so the dashboard can show the full execution story.
    """
    _ = config
    from contextunity.router.modules.tools import get_tool

    trace_tool = get_tool("log_execution_trace")
    if trace_tool is None:
        logger.warning("log_execution_trace tool not found, skipping trace")
        return {}

    pipeline_start = state["_start_ts"] or time.monotonic()
    tenant_id = state["tenant_id"]
    session_id = state["session_id"]
    platform = state["platform"]
    meta = state["metadata"]
    user_id = meta.get("user_id", "anonymous")
    model_key = meta.get("model_key", "")
    current_iteration = state["iteration"]

    # ── Reconstruct timeline from messages ──
    steps: list[TraceStep] = []
    tool_calls_summary: list[ToolCallSummary] = []
    pending_tool_calls: dict[str, TraceStep] = {}
    step_idx = 0
    token_usage: dict[str, int | float] = {"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0}

    all_messages = list(state["messages"])
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
            system_step: TraceStep = {
                "step": step_idx,
                "iteration": 0,
                "type": "system",
                "content": extract_message_content(msg)[:1000],
            }
            steps.append(system_step)
            step_idx += 1

        elif isinstance(msg, HumanMessage):
            content = extract_message_content(msg)
            if not user_query:
                user_query = content
            user_step: TraceStep = {
                "step": step_idx,
                "iteration": 0,
                "type": "user",
                "content": content[:3000],
            }
            steps.append(user_step)
            step_idx += 1

        elif isinstance(msg, AIMessage):
            resp_meta = _response_metadata(msg)
            msg_tokens = _token_usage_from_metadata(resp_meta)
            if msg_tokens is not None:
                token_usage["input_tokens"] += msg_tokens["input"]
                token_usage["output_tokens"] += msg_tokens["output"]

            tool_calls = _tool_calls_from_ai_message(msg)
            if tool_calls:
                for tc in tool_calls:
                    tc_name = _tool_name_from_call(tc)
                    tc_id = _tool_call_id_from_call(tc)
                    tc_args = _string_dict(tc.get("args"))
                    summary_args = _json_args(tc.get("args"))

                    tool_call_step: TraceStep = {
                        "step": step_idx,
                        "iteration": current_iteration,
                        "type": "tool_call",
                        "tool": tc_name,
                        "tool_call_id": str(tc_id),
                        "args": tc_args,
                    }
                    if msg_tokens is not None:
                        tool_call_step["tokens"] = msg_tokens
                    steps.append(tool_call_step)
                    pending_tool_calls[str(tc_id)] = tool_call_step

                    summary: ToolCallSummary = {"tool": tc_name}
                    if summary_args:
                        summary["args"] = summary_args
                    tool_calls_summary.append(summary)
                    step_idx += 1
            else:
                content = extract_message_content(msg)
                final_answer = content
                assistant_step: TraceStep = {
                    "step": step_idx,
                    "iteration": current_iteration,
                    "type": "assistant",
                    "content": content[:5000],
                }
                if msg_tokens is not None:
                    assistant_step["tokens"] = msg_tokens
                steps.append(assistant_step)
                step_idx += 1

        elif isinstance(msg, ToolMessage):
            tc_id = getattr(msg, "tool_call_id", "")
            content = extract_message_content(msg)
            is_error = _tool_message_has_error(msg)
            tool_name = getattr(msg, "name", "")
            if not isinstance(tool_name, str) or not tool_name:
                pending_step = pending_tool_calls.get(tc_id)
                tool_name = pending_step.get("tool", "") if pending_step is not None else ""

            tool_result_step: TraceStep = {
                "step": step_idx,
                "iteration": current_iteration,
                "type": "tool_result",
                "tool": tool_name,
                "tool_call_id": str(tc_id),
                "status": "error" if is_error else "ok",
                "result": content[:10000],
            }
            steps.append(tool_result_step)

            tool_calls_summary.append({"tool": tool_name, "status": "error" if is_error else "ok"})
            step_idx += 1

    timing_ms = int((time.monotonic() - pipeline_start) * 1000)

    # ── Security context ──
    security_flags = list(state.get("security_flags", []))

    state_map: dict[str, object] = dict(state)
    access_token = state_map.get("__token__") or state_map.get("access_token")
    token_info: TokenInfoDict = {}
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

    # ── Log via universal tool ──
    try:
        from contextunity.router.langchain_boundaries import invoke_tool_arun

        _ = await invoke_tool_arun(
            trace_tool,
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
                "security_flags": {"events": security_flags, **token_info},
                "record_episode": True,
            },
        )
        logger.info(
            "Logged dispatcher trace for session %s (%d steps)",
            session_id,
            len(steps),
        )
    except Exception as e:
        detail_attr = getattr(e, "details", None)
        detail = detail_attr() if callable(detail_attr) else str(detail_attr or e)
        code_attr = getattr(e, "code", None)
        code = code_attr() if callable(code_attr) else str(code_attr or type(e).__name__)
        logger.error("Brain trace failed [%s]: %s", code, detail)

    return {}


__all__ = ["reflect_dispatcher"]
