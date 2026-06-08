"""Dispatcher agent graph state types.

Contains only the dispatcher-specific graph state TypedDict.
Shared types (ExecutionMetadata, SecurityFlag, etc.) live in ``cortex/types.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeGuard

from contextunity.core import ContextToken
from contextunity.core.types import is_object_dict
from langchain_core.messages import BaseMessage

from ..types import (
    ExecutionMetadata,
    GraphState,
    SecurityFlag,
    StateUpdate,
)


class DispatcherState(GraphState):
    """Full dispatcher state — all fields required.

    Extends ``GraphState`` with dispatcher-specific fields.
    ``state["field"]`` access is safe without ``.get()``.

    Always constructed with all keys populated in ``DispatcherService.invoke``
    and ``DispatcherService.stream``.

    Flow: agent -> security -> [execute|blocked|end]
                             -> execute -> tools -> agent
                             -> blocked -> agent (with error)
                             -> end -> reflect -> END
    """

    security_flags: list[SecurityFlag]
    hitl_approved: bool
    error_detected: bool
    healing_triggered: bool


def make_dispatcher_state(
    *,
    messages: Sequence[BaseMessage],
    access_token: ContextToken,
    tenant_id: str,
    session_id: str,
    platform: str,
    trace_id: str,
    max_iterations: int,
    allowed_tools: list[str],
    denied_tools: list[str],
    metadata: ExecutionMetadata | None = None,
) -> DispatcherState:
    """Create a fully initialized ``DispatcherState`` with zero-value defaults.

    Encapsulates the construction contract so callers never need to remember
    which fields require explicit initialization versus which can be zeroed.

    Args:
        messages: Initial conversation messages (typically a single ``HumanMessage``).
        access_token: The authenticated ``ContextToken`` for permission checks.
        tenant_id: Tenant isolation boundary (e.g. ``"acme-corp"``).
        session_id: Unique session identifier for checkpoint scoping.
        platform: Platform origin label (e.g. ``"telegram"``, ``"web"``).
        trace_id: Distributed tracing correlation ID.
        max_iterations: Hard limit on agent reasoning loops.
        allowed_tools: Whitelist of tool names the agent may invoke
            (``["*"]`` permits all).
        denied_tools: Blacklist of tool names the agent must not invoke.
        metadata: Optional execution metadata (user ID, graph name, etc.).

    Returns:
        A ``DispatcherState`` with all required keys populated, including
        zeroed counters (``iteration=0``), empty collections, and
        ``False`` flags for error/healing state.
    """
    return DispatcherState(
        messages=list(messages),
        access_token=access_token,
        tenant_id=tenant_id,
        session_id=session_id,
        platform=platform,
        trace_id=trace_id,
        metadata=metadata or ExecutionMetadata(),
        max_iterations=max_iterations,
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
        # — zero-value defaults —
        iteration=0,
        _start_ts=0.0,
        final_output={},
        structured_output="",
        intermediate_results={},
        _last_node="",
        _raw_output="",
        dynamic={},
        security_flags=[],
        hitl_approved=False,
        error_detected=False,
        healing_triggered=False,
    )


def _validate_dispatcher_field_types(value: StateUpdate) -> None:
    """Reject wrong runtime types for dispatcher fields that are present."""
    from contextunity.core.exceptions import ConfigurationError

    hitl = value.get("hitl_approved")
    if hitl is not None and not isinstance(hitl, bool):
        raise ConfigurationError(
            message=(
                f"Dispatcher state field 'hitl_approved' must be bool, got {type(hitl).__name__}"
            )
        )
    flags = value.get("security_flags")
    if flags is not None and not isinstance(flags, list):
        raise ConfigurationError(
            message=(
                f"Dispatcher state field 'security_flags' must be list, got {type(flags).__name__}"
            )
        )
    messages = value.get("messages")
    if messages is not None and not isinstance(messages, list):
        raise ConfigurationError(
            message=(
                f"Dispatcher state field 'messages' must be list, got {type(messages).__name__}"
            )
        )


def _dispatcher_state_keys_present(value: StateUpdate) -> bool:
    """Return whether ``value`` carries the full dispatcher state key contract."""
    return "messages" in value and "security_flags" in value and "hitl_approved" in value


def validated_dispatcher_stream_event(event: object) -> StateUpdate:
    """Validate a LangGraph stream chunk and return it as a plain ``StateUpdate``."""
    from contextunity.core.exceptions import ConfigurationError

    if not is_object_dict(event):
        raise ConfigurationError(
            message=(
                "Dispatcher stream returned invalid event "
                f"({type(event).__name__}); refusing to forward graph output."
            )
        )

    update: StateUpdate = dict(event)
    _validate_dispatcher_field_types(update)
    if _dispatcher_state_keys_present(update):
        return update

    for node_update in update.values():
        if is_object_dict(node_update):
            partial: StateUpdate = dict(node_update)
            _validate_dispatcher_field_types(partial)
    return update


def is_dispatcher_state(value: object) -> TypeGuard[DispatcherState]:
    """Return whether ``value`` is a complete, well-typed dispatcher graph state."""
    if not is_object_dict(value):
        return False
    state: StateUpdate = dict(value)
    if not _dispatcher_state_keys_present(state):
        return False
    return (
        isinstance(state.get("hitl_approved"), bool)
        and isinstance(state.get("security_flags"), list)
        and isinstance(state.get("messages"), list)
    )


__all__ = [
    "DispatcherState",
    "is_dispatcher_state",
    "make_dispatcher_state",
    "validated_dispatcher_stream_event",
]
