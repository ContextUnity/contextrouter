"""Runner entrypoint for the RAG agent.

This module provides the main entry points for invoking the LangGraph agent,
with proper Langfuse tracing integration.

Main Functions:
    stream_agent: Primary streaming entrypoint used by apps/api.
    invoke_agent: Async invocation for non-streaming use cases.
    invoke_agent_sync: Sync wrapper for non-async contexts.

Usage:
    from contextrouter.cortex import stream_agent

    async for event in stream_agent(
        messages=lc_messages,
        session_id="...",
        platform="api",
    ):
        # Process LangGraph events
        pass
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from contextrouter.core import TokenBuilder, UserCtx, get_core_config
from contextrouter.cortex import compile_graph, get_last_user_query
from contextrouter.cortex.callbacks.tool_handler import ToolEventCallbackHandler
from contextrouter.modules.observability import get_langfuse_callbacks, trace_context
from contextrouter.modules.retrieval.rag import (
    get_effective_data_store_id,
    get_rag_retrieval_settings,
    use_runtime_settings,
)
from contextrouter.modules.retrieval.rag.types import RuntimeRagSettings

logger = logging.getLogger(__name__)


def _trace_input_from_messages(messages: Sequence[BaseMessage]) -> dict[str, object]:
    last_user_message = get_last_user_query(messages)

    return {
        "last_user_message": last_user_message,
        "message_count": len(messages),
    }


def _trace_metadata() -> dict[str, object]:
    """Build Langfuse trace metadata.

    Store ONLY the effective configuration to keep traces readable and stable.
    """
    core_cfg = get_core_config()
    rag_cfg = get_rag_retrieval_settings()

    # Align trace metadata with the actual datastore selection logic used by retrieval.
    effective_data_store_id = get_effective_data_store_id()

    return {
        "effective_config": {
            "default_llm": core_cfg.models.default_llm,
            "temperature": core_cfg.llm.temperature,
            "max_output_tokens": core_cfg.llm.max_output_tokens,
            "location": core_cfg.vertex.location,
            "data_store_id": effective_data_store_id,
            "general_retrieval_enabled": rag_cfg.general_retrieval_enabled,
            "general_retrieval_initial_count": rag_cfg.general_retrieval_initial_count,
            "general_retrieval_final_count": rag_cfg.general_retrieval_final_count,
            "max_books": rag_cfg.max_books,
            "max_videos": rag_cfg.max_videos,
            "max_qa": rag_cfg.max_qa,
            "citations_books": rag_cfg.citations_books,
            "citations_videos": rag_cfg.citations_videos,
            "citations_qa": rag_cfg.citations_qa,
            "env_locked": sorted(rag_cfg.env_locked),
        }
    }


def _trace_tags(
    *,
    platform: str,
    runtime_settings: RuntimeRagSettings | None,
    enable_suggestions: bool | None = None,
    enable_web_search: bool | None = None,
) -> list[str]:
    """Curated, low-cardinality tags for Langfuse filtering."""
    core_cfg = get_core_config()
    rag_cfg = get_rag_retrieval_settings()

    ds = None
    if isinstance(runtime_settings, dict):
        v = runtime_settings.get("rag_dataset")
        if isinstance(v, str) and v.strip().lower() in {"blue", "green"}:
            ds = v.strip().lower()

    retrieval_mode = (
        "general" if getattr(rag_cfg, "general_retrieval_enabled", False) else "per_source"
    )

    tags = [
        platform,
        f"model:{core_cfg.models.default_llm}",
        f"dataset:{ds or 'default'}",
        f"retrieval:{retrieval_mode}",
    ]
    if enable_web_search is not None:
        tags.append(f"web:{'on' if enable_web_search else 'off'}")
    if enable_suggestions is not None:
        tags.append(f"suggest:{'on' if enable_suggestions else 'off'}")
    return tags


async def invoke_agent(
    messages: Sequence[BaseMessage],
    session_id: str,
    platform: str,
    user_ctx: UserCtx | None = None,
    *,
    style_prompt: str = "",
    no_results_prompt: str = "",
    rag_system_prompt_override: str = "",
    rag_filter: str = "",
    runtime_settings: RuntimeRagSettings | None = None,
    enable_suggestions: bool = True,
) -> dict[str, object]:
    """Invoke the RAG agent (non-streaming)."""
    effective_user_id = user_ctx.get("user_id") if user_ctx else None

    graph = compile_graph()
    callbacks = get_langfuse_callbacks(
        session_id=session_id,
        user_id=effective_user_id,
        platform=platform,
    )
    # Add tool event callback handler for observability
    tool_handler = ToolEventCallbackHandler()
    callbacks.append(tool_handler)

    input_state = {
        "messages": list(messages),
        "session_id": session_id,
        "platform": platform,
        "citations_output": "raw",
        "citations_allowed_types": [],
        "style_prompt": style_prompt,
        "no_results_prompt": no_results_prompt,
        "rag_system_prompt_override": rag_system_prompt_override,
        "rag_filter": rag_filter,
        "enable_suggestions": enable_suggestions,
    }

    # Optional Biscuit token security (config-driven).
    core_cfg = get_core_config()
    if core_cfg.security.enabled:
        builder = TokenBuilder(enabled=True, private_key_path=core_cfg.security.private_key_path)
        token = builder.mint_root(
            user_ctx=user_ctx or {},
            permissions=(
                core_cfg.security.policies.read_permission,
                core_cfg.security.policies.write_permission,
            ),
            ttl_s=300.0,
        )
        input_state["access_token"] = token
        logger.debug(
            "Security enabled: minted access_token token_id=%s perms=%s ttl_s=%s",
            getattr(token, "token_id", None),
            list(getattr(token, "permissions", ()) or ()),
            300.0,
        )
    else:
        logger.debug("Security disabled: providers will run without token verification")

    # IMPORTANT: `use_runtime_settings()` must be entered BEFORE we compute effective_config/tags.
    # Otherwise `get_config()` will reflect only env defaults and not the per-user UI settings.
    with use_runtime_settings(runtime_settings):
        tags = _trace_tags(platform=platform, runtime_settings=runtime_settings)
        meta = _trace_metadata()
        with trace_context(
            session_id=session_id,
            platform=platform,
            name="rag_invoke",
            user_id=effective_user_id,
            trace_input=_trace_input_from_messages(messages),
            trace_metadata=meta,
            trace_tags=tags,
        ):
            result = await graph.ainvoke(input_state, config={"callbacks": callbacks})

    return result


async def stream_agent(
    messages: Sequence[BaseMessage],
    session_id: str,
    platform: str,
    user_ctx: UserCtx | None = None,
    *,
    citations_allowed_types: list[str] | None = None,
    style_prompt: str = "",
    no_results_prompt: str = "",
    rag_system_prompt_override: str = "",
    search_suggestions_prompt_override: str = "",
    rag_filter: str = "",
    enable_suggestions: bool = True,
    suggestions_model: str = "",
    enable_web_search: bool = True,
    web_allowed_domains: list[str] | None = None,
    max_web_results: int = 10,
    runtime_settings: RuntimeRagSettings | None = None,
) -> AsyncIterator[dict[str, object]]:
    """Stream the RAG agent response.

    Yields event dicts from LangGraph astream_events (v2 format).
    """
    effective_user_id = user_ctx.get("user_id") if user_ctx else None

    graph = compile_graph()
    callbacks = get_langfuse_callbacks(
        session_id=session_id,
        user_id=effective_user_id,
        platform=platform,
    )
    # Add tool event callback handler for observability
    tool_handler = ToolEventCallbackHandler()
    callbacks.append(tool_handler)

    core_cfg = get_core_config()
    effective_suggestions_model = (
        (suggestions_model or "").strip()
        or (core_cfg.models.rag.suggestions.model or "").strip()
        or "vertex/gemini-2.5-flash-lite"
    )

    input_state = {
        "messages": list(messages),
        "session_id": session_id,
        "platform": platform,
        "enable_suggestions": bool(enable_suggestions),
        "suggestions_model": effective_suggestions_model,
        "enable_web_search": bool(enable_web_search),
        "web_allowed_domains": list(web_allowed_domains or []),
        "max_web_results": int(max_web_results),
        "citations_allowed_types": list(citations_allowed_types or []),
        "style_prompt": style_prompt,
        "no_results_prompt": no_results_prompt,
        "rag_system_prompt_override": rag_system_prompt_override,
        "search_suggestions_prompt_override": search_suggestions_prompt_override,
        "rag_filter": rag_filter,
    }
    if core_cfg.security.enabled:
        builder = TokenBuilder(enabled=True, private_key_path=core_cfg.security.private_key_path)
        token = builder.mint_root(
            user_ctx=user_ctx or {},
            permissions=(
                core_cfg.security.policies.read_permission,
                core_cfg.security.policies.write_permission,
            ),
            ttl_s=300.0,
        )
        input_state["access_token"] = token
        logger.debug(
            "Security enabled: minted access_token token_id=%s perms=%s ttl_s=%s",
            getattr(token, "token_id", None),
            list(getattr(token, "permissions", ()) or ()),
            300.0,
        )
    else:
        logger.debug("Security disabled: providers will run without token verification")

    # IMPORTANT: `use_runtime_settings()` must be entered BEFORE we compute effective_config/tags.
    with use_runtime_settings(runtime_settings):
        tags = _trace_tags(
            platform=platform,
            runtime_settings=runtime_settings,
            enable_suggestions=enable_suggestions,
            enable_web_search=enable_web_search,
        )
        meta = _trace_metadata()
        with trace_context(
            session_id=session_id,
            platform=platform,
            name="rag_stream",
            user_id=effective_user_id,
            trace_input=_trace_input_from_messages(messages),
            trace_metadata=meta,
            trace_tags=tags,
        ):
            async for event in graph.astream_events(
                input_state, config={"callbacks": callbacks}, version="v2"
            ):
                yield event


def invoke_agent_sync(
    messages: Sequence[BaseMessage],
    session_id: str,
    platform: str,
) -> dict[str, object]:
    """Synchronous version of invoke_agent for non-async contexts."""
    graph = compile_graph()
    callbacks = get_langfuse_callbacks(
        session_id=session_id,
        user_id=None,
        platform=platform,
    )
    # Add tool event callback handler for observability
    tool_handler = ToolEventCallbackHandler()
    callbacks.append(tool_handler)

    input_state = {
        "messages": list(messages),
        "session_id": session_id,
        "platform": platform,
    }

    with trace_context(
        session_id=session_id,
        platform=platform,
        name="rag_invoke_sync",
        user_id=None,
        trace_input={"messages": [m.content for m in messages if hasattr(m, "content")]},
    ):
        result = graph.invoke(input_state, config={"callbacks": callbacks})

    return result


def create_input_from_query(
    query: str,
    session_id: str,
    platform: str,
    history: Sequence[BaseMessage] | None = None,
) -> dict[str, object]:
    """Helper to create input state from a simple query string."""
    messages: list[BaseMessage] = list(history) if history else []
    messages.append(HumanMessage(content=query))

    return {
        "messages": messages,
        "session_id": session_id,
        "platform": platform,
    }


__all__ = ["stream_agent", "invoke_agent", "invoke_agent_sync", "create_input_from_query"]
