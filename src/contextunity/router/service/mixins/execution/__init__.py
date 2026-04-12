"""Execution mixin package."""

from .agent import AgentExecutionMixin
from .dispatcher import DispatcherExecutionMixin
from .helpers import _resolve_tenant_id


class ExecutionMixin(AgentExecutionMixin, DispatcherExecutionMixin):
    """Mixin providing ExecuteAgent, ExecuteDispatcher, StreamDispatcher handlers."""


__all__ = ["ExecutionMixin", "_resolve_tenant_id"]
