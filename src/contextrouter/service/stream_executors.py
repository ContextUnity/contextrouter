"""Stream Executor Manager — manages bidi streams for project-side tool execution.

Projects connect via ToolExecutorStream and register as remote executors
for their tools. The manager dispatches execution requests to the
appropriate project stream and awaits results.

Usage:
    manager = get_stream_executor_manager()
    if manager.is_available("acme", "execute_analytics_sql"):
        result = await manager.execute("acme", "execute_analytics_sql", {"sql": "SELECT ..."})
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A pending tool execution request awaiting result from project."""

    request_id: str
    future: asyncio.Future
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class StreamExecutor:
    """Represents an active bidi stream connection from a project."""

    project_id: str
    tool_names: list[str]
    send_queue: asyncio.Queue  # Router → Project messages
    pending: dict[str, PendingRequest] = field(default_factory=dict)


class StreamExecutorManager:
    """Manages bidi streams from projects for tool execution callback.

    Thread-safe singleton that tracks active project streams and
    dispatches execution requests to them.
    """

    def __init__(self) -> None:
        # project_id → StreamExecutor
        self._executors: dict[str, StreamExecutor] = {}

    def register(
        self,
        project_id: str,
        tool_names: list[str],
        send_queue: asyncio.Queue,
    ) -> StreamExecutor:
        """Register an active stream executor for a project.

        Called when project sends 'ready' message on ToolExecutorStream.
        """
        executor = StreamExecutor(
            project_id=project_id,
            tool_names=tool_names,
            send_queue=send_queue,
        )
        self._executors[project_id] = executor
        logger.info(
            "Stream executor registered: project=%s tools=%s",
            project_id,
            tool_names,
        )
        return executor

    def unregister(self, project_id: str) -> None:
        """Remove stream executor when project disconnects."""
        executor = self._executors.pop(project_id, None)
        if executor:
            # Cancel any pending requests
            for req in executor.pending.values():
                if not req.future.done():
                    req.future.set_exception(
                        ConnectionError(f"Stream disconnected for project '{project_id}'")
                    )
            logger.info("Stream executor unregistered: project=%s", project_id)

    def is_available(self, project_id: str, tool_name: str) -> bool:
        """Check if a project has an active stream for this tool."""
        executor = self._executors.get(project_id)
        if not executor:
            return False
        return tool_name in executor.tool_names

    async def execute(
        self,
        project_id: str,
        tool_name: str,
        args: dict,
        timeout: float = 30.0,
    ) -> dict:
        """Send execution request to project via bidi stream and await result.

        Args:
            project_id: Target project ID
            tool_name: Tool to execute
            args: Tool arguments (e.g. {"sql": "SELECT ..."})
            timeout: Max seconds to wait for result

        Returns:
            Result dict from project

        Raises:
            ConnectionError: If stream not available
            TimeoutError: If project doesn't respond in time
        """
        executor = self._executors.get(project_id)
        if not executor or tool_name not in executor.tool_names:
            raise ConnectionError(f"No active stream for project '{project_id}' tool '{tool_name}'")

        request_id = str(uuid.uuid4())[:8]
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        pending = PendingRequest(request_id=request_id, future=future)
        executor.pending[request_id] = pending

        # Send execute request to project — include caller context for
        # audit trail and project-side tenant filtering (VULN-7 fix).
        # When security is enabled, caller_tenant is MANDATORY.
        caller_tenant = ""
        caller_user = ""
        try:
            from contextrouter.cortex.runtime_context import get_current_access_token

            token = get_current_access_token()
            if token:
                caller_tenant = (getattr(token, "allowed_tenants", None) or ("",))[0]
                caller_user = getattr(token, "user_id", "") or ""
        except Exception:
            pass  # Best-effort extraction

        # Fail-closed: if security is enabled and we can't resolve the
        # caller, reject rather than forwarding with empty context.
        if not caller_tenant:
            try:
                from contextcore.exceptions import SecurityError

                from contextrouter.core import get_core_config

                config = get_core_config()
                if config.security.enabled:
                    raise SecurityError(
                        f"Cannot forward execution to project '{project_id}': "
                        f"no caller_tenant resolved. Security is enabled — "
                        f"caller context is mandatory for stream execution."
                    )
            except ImportError:
                pass

        message = {
            "action": "execute",
            "tool": tool_name,
            "request_id": request_id,
            "args": args,
            "caller_tenant": caller_tenant,
            "caller_user": caller_user,
        }
        await executor.send_queue.put(message)

        logger.info(
            "Sent execute request: project=%s tool=%s request_id=%s",
            project_id,
            tool_name,
            request_id,
        )

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Tool execution timed out ({timeout}s): "
                f"project={project_id} tool={tool_name} request_id={request_id}"
            )
        finally:
            executor.pending.pop(request_id, None)

    def resolve_result(self, project_id: str, request_id: str, result: dict) -> None:
        """Resolve a pending request with the result from project.

        Called when project sends 'result' or 'error' message.
        """
        executor = self._executors.get(project_id)
        if not executor:
            logger.warning(
                "Result for unknown project: project=%s request_id=%s",
                project_id,
                request_id,
            )
            return

        pending = executor.pending.get(request_id)
        if not pending:
            logger.warning(
                "Result for unknown request: project=%s request_id=%s",
                project_id,
                request_id,
            )
            return

        if not pending.future.done():
            pending.future.set_result(result)

    def get_executor(self, project_id: str) -> StreamExecutor | None:
        """Get executor for a project (if connected)."""
        return self._executors.get(project_id)


# Singleton instance
_manager: StreamExecutorManager | None = None


def get_stream_executor_manager() -> StreamExecutorManager:
    """Get or create the global StreamExecutorManager singleton."""
    global _manager
    if _manager is None:
        _manager = StreamExecutorManager()
    return _manager


__all__ = ["StreamExecutorManager", "get_stream_executor_manager"]
