"""Adapter functions mapping platform state to specific domain logic signatures."""

from __future__ import annotations

from contextunity.core.types import is_object_dict

from contextunity.router.cortex.compiler.state_routing import read_state_input

from ..no_results import no_results_response
from ..sql_visualizer import make_visualizer_node
from . import sql_executor, sql_planner, sql_verifier
from .bridges import EMPTY_RUNNABLE_CONFIG
from .contracts import (
    PlatformResult,
    PlatformState,
)


async def no_results_adapter(state: PlatformState) -> PlatformResult:
    """Adapt no_results_response (AIMessage) → dict for platform tool contract."""
    from ....utils.messages import format_conversation_history

    result_msg = await no_results_response(
        user_query=str(read_state_input(state, "user_query", default="")),
        conversation_history=format_conversation_history(state.get("messages", [])),
        state=state,
        prompt_override=str(read_state_input(state, "no_results_prompt", default="") or ""),
    )
    return {
        "messages": [result_msg],
        "citations": [],
        "generation_complete": True,
        "should_retrieve": False,
    }


async def sql_plan_adapter(state: PlatformState) -> PlatformResult:
    """Platform wrapper — delegates to plan_sql domain function."""
    result = await sql_planner.plan_sql(state, EMPTY_RUNNABLE_CONFIG)
    return dict(result) if result else {}


async def sql_execute_adapter(state: PlatformState) -> PlatformResult:
    """Platform wrapper for sql execute factory."""
    node = sql_executor.make_execute_node(sql_tool=None, tool_names=[])
    result = await node(state)
    return dict(result) if result else {}


async def sql_verify_adapter(state: PlatformState) -> PlatformResult:
    """Platform wrapper — delegates to verify_sql domain function."""
    result = await sql_verifier.verify_sql(state, EMPTY_RUNNABLE_CONFIG)
    return dict(result) if result else {}


async def sql_visualize_adapter(state: PlatformState) -> PlatformResult:
    """Platform wrapper for sql visualize factory."""
    raw_sub_prompts = read_state_input(state, "visualizer_sub_prompts")
    sub_prompts: dict[str, str] | None = (
        {key: str(value) for key, value in raw_sub_prompts.items()}
        if is_object_dict(raw_sub_prompts)
        else None
    )

    node = make_visualizer_node(
        visualizer_prompt=str(read_state_input(state, "visualizer_prompt", default="")),
        visualizer_sub_prompts=sub_prompts,
        default_model_key=str(read_state_input(state, "model_key", default="") or ""),
    )
    result = await node(state, EMPTY_RUNNABLE_CONFIG)
    return dict(result) if result else {}


__all__ = [
    "no_results_adapter",
    "sql_execute_adapter",
    "sql_plan_adapter",
    "sql_verify_adapter",
    "sql_visualize_adapter",
]
