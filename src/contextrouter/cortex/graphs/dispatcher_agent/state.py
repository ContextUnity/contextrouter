"""State definition for dispatcher agent graph."""

from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DispatcherState(TypedDict):
    """State for dispatcher agent graph.

    Flow: agent → security → [execute|blocked|end]
                              → execute → tools → agent
                              → blocked → agent (with error)
                              → end → reflect → END
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tenant_id: str  # Tenant identifier for multi-tenant isolation
    session_id: str
    platform: str
    metadata: dict[str, Any]
    iteration: int
    max_iterations: int
    allowed_tools: list[str]  # Tool access control: allowed tool names
    denied_tools: list[str]  # Tool access control: denied tool names
    access_token: Any | None  # ContextToken forwarded from gRPC boundary

    # Tracing integration
    trace_id: str | None  # Distributed trace ID for observability
    _start_ts: float  # Pipeline start time (monotonic) for duration measurement

    # Security events (populated by security_guard_node)
    security_flags: list[dict[str, Any]]  # Blocked tool events, HITL events
    hitl_approved: bool  # Set to True after human approves CONFIRM-risk tools

    # Self-healing integration
    error_detected: bool  # Flag to trigger self-healing
    healing_triggered: bool  # Flag to track if healing was triggered


__all__ = ["DispatcherState"]
