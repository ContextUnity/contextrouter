"""Execution mixin package -- agent, node, and dispatcher execution RPC handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentExecutionMixin
    from .dispatcher import DispatcherExecutionMixin
    from .helpers import _resolve_tenant_id
    from .node import NodeExecutionMixin

    class ExecutionMixin(AgentExecutionMixin, DispatcherExecutionMixin, NodeExecutionMixin):
        """Mixin providing ExecuteAgent, ExecuteDispatcher, StreamDispatcher, ExecuteNode handlers."""


def __getattr__(name: str) -> object:
    """Lazily expose execution handlers without importing payload consumers."""
    if name == "_resolve_tenant_id":
        from .helpers import _resolve_tenant_id

        return _resolve_tenant_id
    if name == "ExecutionMixin":
        from .agent import AgentExecutionMixin
        from .dispatcher import DispatcherExecutionMixin
        from .node import NodeExecutionMixin

        class ExecutionMixin(AgentExecutionMixin, DispatcherExecutionMixin, NodeExecutionMixin):
            """Combined Router execution RPC handlers."""

        globals()["ExecutionMixin"] = ExecutionMixin
        return ExecutionMixin
    raise AttributeError(name)


__all__ = ["ExecutionMixin", "_resolve_tenant_id"]
