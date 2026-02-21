"""Reflection node — post-interaction memory + trace persistence.

Runs at the end of the graph flow to:
1. Record the conversation episode in episodic memory
2. Extract and persist user facts (entity memory)
3. Log an execution trace for observability
"""

from __future__ import annotations

import logging
import time
from typing import Any

from contextcore.sdk import BrainClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from contextrouter.core import get_core_config
from contextrouter.core.memory import MemoryManager
from contextrouter.cortex import AgentState

logger = logging.getLogger(__name__)


def _extract_tool_calls(state: AgentState) -> list[dict[str, Any]]:
    """Extract tool call records from message history."""
    calls = []
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(
                    {
                        "tool": tc.get("name", ""),
                        "args": {k: str(v)[:200] for k, v in tc.get("args", {}).items()},
                    }
                )
        elif isinstance(msg, ToolMessage):
            calls.append(
                {
                    "tool": msg.name or "",
                    "status": "ok" if not msg.additional_kwargs.get("error") else "error",
                }
            )
    return calls


def _estimate_token_usage(state: AgentState) -> dict[str, Any]:
    """Estimate token usage from message metadata (best effort)."""
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and hasattr(msg, "response_metadata"):
            meta = msg.response_metadata or {}
            token_info = meta.get("token_usage") or meta.get("usage") or {}
            if token_info:
                usage["input_tokens"] += token_info.get(
                    "prompt_tokens", token_info.get("input_tokens", 0)
                )
                usage["output_tokens"] += token_info.get(
                    "completion_tokens", token_info.get("output_tokens", 0)
                )
    return usage


async def reflect_interaction(state: AgentState) -> dict:
    """Analyze finished interaction: persist episode, extract facts, log trace.

    This is the END node of the graph — it produces no new messages.
    """
    start_ts = time.monotonic()
    config = get_core_config()
    manager = MemoryManager(config)

    user_id = state.get("metadata", {}).get("user_id", "anonymous")
    tenant_id = state.get("metadata", {}).get("tenant_id", "default")
    session_id = state.get("session_id", "")
    platform = state.get("platform", "")

    # ── 1. Record Episode ──────────────────────────────────────────
    last_user_msg = ""
    last_ai_msg = ""

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and not last_user_msg:
            last_user_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, AIMessage) and not last_ai_msg:
            last_ai_msg = msg.content if isinstance(msg.content, str) else str(msg.content)
        if last_user_msg and last_ai_msg:
            break

    if last_user_msg:
        summary = f"User asked: {last_user_msg[:200]}... AI responded: {last_ai_msg[:200]}..."
        try:
            await manager.record_episode(
                user_id=user_id,
                content=summary,
                session_id=session_id,
                metadata={
                    "full_query": last_user_msg[:500],
                    "platform": platform,
                },
                tenant_id=tenant_id,
            )
        except Exception as e:
            logger.error("Failed to record episode: %s", e)

    # ── 2. Extract Facts (heuristic) ───────────────────────────────
    if last_user_msg:
        lower = last_user_msg.lower()
        # Language detection heuristic
        if any(c in lower for c in ["люблю", "подобається", "хочу"]):
            for color in ["синій", "червоний", "чорний", "білий", "зелений"]:
                if color in lower:
                    try:
                        await manager.upsert_user_fact(
                            user_id=user_id,
                            key="preferred_color",
                            value=color,
                            confidence=0.8,
                            tenant_id=tenant_id,
                        )
                        logger.info("Memory: Extracted preference for %s", color)
                    except Exception as e:
                        logger.error("Failed to upsert fact: %s", e)

    # ── 3. Log Execution Trace ─────────────────────────────────────
    timing_ms = int((time.monotonic() - start_ts) * 1000)
    tool_calls = _extract_tool_calls(state)
    token_usage = _estimate_token_usage(state)

    # Determine graph_name from platform/source
    graph_name = "rag_retrieval"  # Default; dispatcher will pass its own

    # Build provenance chain
    provenance = [f"agent:router:{graph_name}"]
    for tc in tool_calls:
        tool_name = tc.get("tool", "")
        if tool_name:
            provenance.append(f"tool:{tool_name}")
    provenance.append(f"router:{graph_name}:reflect")

    try:
        brain_host = getattr(config.providers, "brain_host", "localhost:50051")

        from contextrouter.core.brain_token import get_brain_service_token

        brain = BrainClient(host=brain_host, token=get_brain_service_token())
        await brain.log_trace(
            tenant_id=tenant_id,
            agent_id=f"router:{graph_name}",
            session_id=session_id,
            user_id=user_id,
            graph_name=graph_name,
            tool_calls=tool_calls,
            token_usage=token_usage,
            timing_ms=timing_ms,
            metadata={
                "platform": platform,
                "user_query": last_user_msg[:200] if last_user_msg else "",
                "message_count": len(state.get("messages", [])),
                "security_flags": list(state.get("security_flags", [])),
            },
            provenance=provenance,
        )
        logger.info("Logged trace for session %s", session_id)
    except Exception as e:
        logger.error("Failed to log trace: %s", e)

    return {}
