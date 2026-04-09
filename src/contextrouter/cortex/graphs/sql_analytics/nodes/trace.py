"""Trace node — logs rich execution trace to Brain.

Uses the universal ``log_execution_trace`` tool which handles:
  • Detailed step timeline (Graph Journey in ContextView)
  • Per-tool timing, request/result data
  • Token usage and cost estimation
  • Episodic memory recording
  • Provenance chain
"""

from __future__ import annotations

import time

from contextcore import get_context_unit_logger
from langchain_core.messages import HumanMessage

from contextrouter.cortex.graphs.sql_analytics.state import SqlAnalyticsState
from contextrouter.modules.tools import get_tool

logger = get_context_unit_logger(__name__)


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
        has_pii = bool(metadata.get("session_id"))

        # Steps and tool_calls are now captured by BrainAutoTracer via
        # LangChain callbacks — no manual conversion needed here.
        steps: list[dict] = []
        tool_calls_summary: list[dict] = []

        # ── Extract user query for episodic memory ──
        user_query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_query = str(msg.content)
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # ── Token usage ──
        token_usage_raw = state.get("_token_usage") or {}
        token_usage = {
            "input_tokens": token_usage_raw.get("input_tokens", 0),
            "output_tokens": token_usage_raw.get("output_tokens", 0),
            "total_cost": token_usage_raw.get("total_cost", 0.0),
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
                    "security_flags": token_info,
                    "record_episode": True,
                }
            )
        except Exception as e:
            detail_attr = getattr(e, "details", None)
            detail = detail_attr() if callable(detail_attr) else str(detail_attr or e)
            code_attr = getattr(e, "code", None)
            code = code_attr() if callable(code_attr) else str(code_attr or type(e).__name__)
            logger.warning("Brain trace failed [%s]: %s", code, detail)
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
