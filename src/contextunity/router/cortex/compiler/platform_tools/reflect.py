"""Reflection node — post-interaction memory + trace persistence.

Runs at the end of the graph flow to:
1. Record the conversation episode in episodic memory
2. Extract and persist user facts (entity memory)
3. Log an execution trace for observability
"""

from __future__ import annotations

import time
from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk import BrainClient
from contextunity.core.types import (
    ContextUnitPayload,
    JsonDict,
    JsonValue,
    is_json_dict,
    is_object_dict,
    is_object_list,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ConfigDict

from contextunity.router.core import get_core_config
from contextunity.router.core.memory import MemoryManager
from contextunity.router.cortex.compiler.platform_tools.helpers.base import (
    resolve_tenant_from_state,
)
from contextunity.router.cortex.types import GraphState, StateUpdate, extract_message_content


class ReflectConfig(BaseModel, frozen=True):
    """Platform tool config for registry validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str | None = None


logger = get_contextunit_logger(__name__)


def _trace_metadata(state: GraphState, *, platform: str, last_user_msg: str) -> JsonDict:
    return {
        "platform": platform,
        "user_query": last_user_msg[:200] if last_user_msg else "",
        "message_count": len(state.get("messages", [])),
        "security_flags": _security_flags_from_state(state),
    }


def _security_flags_from_state(state: GraphState) -> list[JsonValue]:
    flags_raw = state.get("dynamic", {}).get("security_flags", [])
    if not is_object_list(flags_raw):
        return []
    return [flag for flag in flags_raw if is_json_dict(flag)]


def _extract_tool_calls(state: GraphState) -> list[ContextUnitPayload]:
    """Extract tool call records from message history."""
    calls: list[ContextUnitPayload] = []
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if not is_object_dict(tc):
                    continue
                args_raw = tc.get("args", {})
                args_dict = (
                    {str(key): str(value)[:200] for key, value in args_raw.items()}
                    if is_object_dict(args_raw)
                    else {}
                )
                calls.append(
                    {
                        "tool": str(tc.get("name", "")),
                        "args": args_dict,
                    }
                )
        elif isinstance(msg, ToolMessage):
            additional_kwargs_raw: object = getattr(msg, "additional_kwargs", {})
            additional_kwargs = (
                additional_kwargs_raw if is_object_dict(additional_kwargs_raw) else {}
            )
            calls.append(
                {
                    "tool": msg.name or "",
                    "status": "ok" if not additional_kwargs.get("error") else "error",
                }
            )
    return calls


def _estimate_token_usage(state: GraphState) -> JsonDict:
    """Estimate token usage from message metadata (best effort)."""
    usage: JsonDict = {"input_tokens": 0, "output_tokens": 0}
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            meta_raw: object = getattr(msg, "response_metadata", None)
            if not is_object_dict(meta_raw):
                continue
            token_info_raw: object = meta_raw.get("token_usage") or meta_raw.get("usage") or {}
            if not is_object_dict(token_info_raw):
                continue
            prompt_tokens = token_info_raw.get(
                "prompt_tokens", token_info_raw.get("input_tokens", 0)
            )
            completion_tokens = token_info_raw.get(
                "completion_tokens", token_info_raw.get("output_tokens", 0)
            )
            current_input = usage.get("input_tokens", 0)
            current_output = usage.get("output_tokens", 0)
            usage["input_tokens"] = (current_input if isinstance(current_input, int) else 0) + (
                prompt_tokens if isinstance(prompt_tokens, int) else 0
            )
            usage["output_tokens"] = (current_output if isinstance(current_output, int) else 0) + (
                completion_tokens if isinstance(completion_tokens, int) else 0
            )
    return usage


async def reflect_interaction(state: GraphState) -> StateUpdate:
    """Analyze finished interaction: persist episode, extract facts, log trace.

    This is the END node of the graph — it produces no new messages.
    """
    start_ts = time.monotonic()
    config = get_core_config()
    manager = MemoryManager(config)

    metadata_raw: object = state.get("metadata")
    metadata = metadata_raw if is_object_dict(metadata_raw) else {}
    user_id_raw = metadata.get("user_id", "anonymous")
    user_id = user_id_raw if isinstance(user_id_raw, str) else "anonymous"
    tenant_id = resolve_tenant_from_state(state, binding="router_reflect")
    session_id = state.get("session_id", "")
    platform = state.get("platform", "")

    # ── 1. Record Episode ──────────────────────────────────────────
    last_user_msg = ""
    last_ai_msg = ""

    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and not last_user_msg:
            last_user_msg = extract_message_content(msg)
        if isinstance(msg, AIMessage) and not last_ai_msg:
            last_ai_msg = extract_message_content(msg)
        if last_user_msg and last_ai_msg:
            break

    if last_user_msg:
        summary = f"User asked: {last_user_msg[:200]}... AI responded: {last_ai_msg[:200]}..."
        try:
            _ = await manager.record_episode(
                user_id=user_id,
                content=summary,
                session_id=session_id,
                metadata={
                    "full_query": last_user_msg[:500],
                    "platform": platform,
                },
                tenant_id=tenant_id,
            )
        except Exception as e:  # graceful-degrade: tool failure returns empty result
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
                    except Exception as e:  # graceful-degrade: tool failure returns empty result
                        logger.error("Failed to upsert fact: %s", e)

    # ── 3. Log Execution Trace ─────────────────────────────────────
    timing_ms = int((time.monotonic() - start_ts) * 1000)
    tool_calls = _extract_tool_calls(state)
    token_usage = _estimate_token_usage(state)

    # Determine graph_name from platform/source
    graph_name = "rag_retrieval"  # Default; dispatcher will pass its own

    try:
        from contextunity.core.config import get_core_config as _get_shared_config

        brain_host = _get_shared_config().brain_url

        from contextunity.router.core.brain_token import get_brain_service_token

        brain = BrainClient(
            host=brain_host,
            token=get_brain_service_token(allowed_tenants=(tenant_id,)),
        )
        _ = await brain.log_trace(
            tenant_id=tenant_id,
            agent_id=f"router:{graph_name}",
            session_id=session_id,
            user_id=user_id,
            graph_name=graph_name,
            tool_calls=tool_calls,
            token_usage=token_usage,
            timing_ms=timing_ms,
            metadata=_trace_metadata(state, platform=platform, last_user_msg=last_user_msg),
        )
        logger.info("Logged trace for session %s", session_id)
    except Exception as e:  # graceful-degrade: tool failure returns empty result
        logger.error("Failed to log trace: %s", e)

    return {}


__all__ = ["reflect_interaction"]
