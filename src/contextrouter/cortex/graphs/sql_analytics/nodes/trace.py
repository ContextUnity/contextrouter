"""Trace node — logs rich execution trace to Brain.

Uses the universal ``log_execution_trace`` tool which handles:
  • Detailed step timeline (Graph Journey in ContextView)
  • Per-tool timing, request/result data
  • Token usage and cost estimation
  • Episodic memory recording
  • Provenance chain
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage

from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.tools import get_tool

logger = logging.getLogger(__name__)


def make_reflect_node():
    """Create the reflect (trace logging) node closure."""

    async def reflect_node(state: SqlAnalyticsState):
        """Log execution trace via universal log_execution_trace tool."""
        trace_tool = get_tool("log_execution_trace")
        if not trace_tool:
            logger.warning("log_execution_trace tool not found, skipping trace")
            return {}

        pipeline_start = state.get("_start_ts") or time.monotonic()
        timing_ms = int((time.monotonic() - pipeline_start) * 1000)
        metadata = state.get("metadata") or {}

        # _steps is accumulated by every node via operator.add reducer
        raw_steps = state.get("_steps") or []
        has_pii = bool(metadata.get("session_id"))

        # ── Build rich steps for ContextView ──
        # Convert flat _steps (from helpers.step()) into the format that
        # ContextView expects: tool_call + tool_result pairs with timing.
        #
        # Timing semantics:
        #   tool_call.timing_ms  = gap/overhead since previous step ended
        #   tool_result.timing_ms = tool execution time (from StepTimer)
        steps: list[dict] = []
        tool_calls_summary: list[dict] = []
        prev_end_ts: float = 0.0  # tracks when previous step finished

        for s in raw_steps:
            tool_name = s.get("tool", "unknown")
            status = s.get("status", "ok")
            step_timing = s.get("timing_ms", 0)
            step_ts = s.get("ts", 0)  # unix timestamp when step completed
            request_data = s.get("request")
            result_data = s.get("result")

            # Compute step start time and gap from previous step
            step_start_ts = (step_ts - step_timing / 1000.0) if step_ts else 0
            gap_ms = 0
            if prev_end_ts and step_start_ts:
                gap_ms = max(0, int((step_start_ts - prev_end_ts) * 1000))
            # Update prev_end_ts for next iteration
            if step_ts:
                prev_end_ts = step_ts

            # Build tool_call step (with args = request data)
            # timing_ms = gap/overhead since previous step ended
            tool_call_step: dict = {
                "step": len(steps),
                "iteration": 1,
                "type": "tool_call",
                "tool": tool_name,
                "tool_call_id": f"{tool_name}_{len(steps)}",
                "status": status,
                "timing_ms": gap_ms,
            }
            if request_data:
                if isinstance(request_data, str):
                    try:
                        import json

                        tool_call_step["args"] = json.loads(request_data)
                    except (ValueError, TypeError):
                        tool_call_step["args"] = {"input": request_data[:3000]}
                elif isinstance(request_data, dict):
                    tool_call_step["args"] = {k: str(v)[:3000] for k, v in request_data.items()}
                else:
                    tool_call_step["args"] = {"input": str(request_data)[:3000]}
            steps.append(tool_call_step)

            # Build tool_result step (with result data)
            # timing_ms = actual tool execution time (from StepTimer)
            if result_data:
                tool_result_step: dict = {
                    "step": len(steps),
                    "iteration": 1,
                    "type": "tool_result",
                    "tool": tool_name,
                    "tool_call_id": tool_call_step["tool_call_id"],
                    "status": status,
                    "timing_ms": step_timing,
                    "result": str(result_data)[:10000] if result_data else "",
                }
                steps.append(tool_result_step)

            # Summary for top-level tool_calls
            tool_calls_summary.append({"tool": tool_name, "status": status})

        # ── Extract user query for episodic memory ──
        user_query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_query = str(msg.content)
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # ── Build provenance chain ──
        provenance = [
            f"agent:{metadata.get('agent_id', 'sql_analytics')}",
            "router:sql_analytics:execute",
        ]
        if has_pii:
            provenance.append("zero:pii_masking")
        for tc in tool_calls_summary:
            provenance.append(f"tool:{tc['tool']}")

        # ── Token usage ──
        token_usage_raw = state.get("_token_usage") or {}
        token_usage = {
            "input_tokens": token_usage_raw.get("input_tokens", 0),
            "output_tokens": token_usage_raw.get("output_tokens", 0),
        }

        # ── Security context from runtime access token ──
        from contextrouter.cortex.runtime_context import get_current_access_token

        access_token = get_current_access_token()
        token_info: dict = {}
        if access_token is not None:
            token_info = {
                "token_id": getattr(access_token, "token_id", ""),
                "user_id": getattr(access_token, "user_id", ""),
                "agent_id": getattr(access_token, "agent_id", ""),
                "user_namespace": getattr(access_token, "user_namespace", "default"),
                "permissions": list(getattr(access_token, "permissions", ())),
                "allowed_tenants": list(getattr(access_token, "allowed_tenants", ())),
            }

        try:
            await trace_tool.ainvoke(
                {
                    "tenant_id": metadata.get("tenant_id", "default"),
                    "agent_id": metadata.get("agent_id", "sql_analytics"),
                    "session_id": metadata.get("session_id", ""),
                    "user_id": metadata.get("user_id", "anonymous"),
                    "graph_name": metadata.get("graph_name", "sql_analytics"),
                    "tool_calls": tool_calls_summary,
                    "token_usage": token_usage,
                    "timing_ms": timing_ms,
                    # ── Rich fields for ContextView ──
                    "steps": steps,
                    "platform": metadata.get("platform", ""),
                    "model_key": metadata.get("model_key", ""),
                    "iterations": state.get("retry_count", 0) + 1,
                    "message_count": len(state.get("messages", [])),
                    "user_query": user_query[:2000],
                    "final_answer": "",  # sql_analytics returns components, not text
                    # ── Extended metadata ──
                    "metadata": {
                        "sql": state.get("sql", "")[:500],
                        "format": state.get("format", ""),
                        "purpose": state.get("purpose", ""),
                        "error": state.get("error", ""),
                        "retry_count": state.get("retry_count", 0),
                        "pii_masking": has_pii,
                        # Langfuse observability — injected by execution mixin
                        "langfuse_trace_id": metadata.get("langfuse_trace_id", ""),
                        "langfuse_trace_url": metadata.get("langfuse_trace_url", ""),
                    },
                    "provenance": provenance,
                    "security_flags": token_info,
                    "record_episode": True,
                }
            )
        except Exception as e:
            detail = e.details() if hasattr(e, "details") else str(e)
            code = e.code() if hasattr(e, "code") else ""
            logger.warning("Brain trace failed [%s]: %s", code or type(e).__name__, detail)
        # ── Destroy PII session to wipe keys from RAM ──
        if has_pii and metadata.get("session_id"):
            try:
                destroy_tool = get_tool("destroy_privacy_session")
                if destroy_tool:
                    await destroy_tool.ainvoke({"session_id": metadata["session_id"]})
                    logger.debug("Destroyed PII session '%s'", metadata["session_id"])
            except Exception as e:
                logger.debug("PII session cleanup: %s", e)

        return {}

    return reflect_node


__all__ = ["make_reflect_node"]
