"""Stream mixin — ToolExecutorStream bidi stream handler."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import threading
from collections.abc import AsyncIterator
from typing import Protocol

import grpc
from contextunity.core import ContextUnit, contextunit_pb2, get_contextunit_logger
from contextunity.core.authz.context import VerifiedAuthContext
from contextunity.core.discovery import get_project_stream_secret
from contextunity.core.types import is_object_list
from grpc.aio import ServicerContext

from contextunity.router.service.mixins.execution.types import ProjectToolMap
from contextunity.router.service.stream_executors import (
    StreamExecutorManager,
    get_stream_executor_manager,
)

logger = get_contextunit_logger(__name__)


class StreamHost(Protocol):
    def get_cached_stream_secret(self, project_id: str) -> str | None: ...
    def put_cached_stream_secret(self, project_id: str, secret: str) -> None: ...

    # Per-project tool list — used to validate ``ready.tools`` is a subset
    # of the tools registered for each project.
    _project_tools: dict[str, list[str]]

    async def stream_reader(
        self,
        request_iterator: AsyncIterator[contextunit_pb2.ContextUnit],
        manager: StreamExecutorManager,
        send_queue: asyncio.Queue[dict[str, object] | None],
        context: ServicerContext[contextunit_pb2.ContextUnit, contextunit_pb2.ContextUnit],
        auth_ctx: VerifiedAuthContext,
    ) -> None: ...


class StreamMixin:
    """Mixin providing ToolExecutorStream bidi RPC handler."""

    _stream_secrets: dict[str, str] = {}
    _stream_secrets_lock: threading.Lock = threading.Lock()
    _project_tools: ProjectToolMap = {}

    def get_cached_stream_secret(self, project_id: str) -> str | None:
        """Return in-memory stream auth secret for *project_id*, if cached."""
        with self._stream_secrets_lock:
            return self._stream_secrets.get(project_id)

    def put_cached_stream_secret(self, project_id: str, secret: str) -> None:
        """Cache stream auth secret for reconnecting project executors."""
        with self._stream_secrets_lock:
            self._stream_secrets[project_id] = secret

    async def ToolExecutorStream(
        self: StreamHost,
        request_iterator: AsyncIterator[contextunit_pb2.ContextUnit],
        context: ServicerContext[contextunit_pb2.ContextUnit, contextunit_pb2.ContextUnit],
    ) -> AsyncIterator[contextunit_pb2.ContextUnit]:
        """Handle bidirectional stream for project-side tool execution.

        This is an async generator: yields ContextUnit messages to the project
        while concurrently reading from request_iterator.

        Flow:
        1. Project sends 'ready' with project_id and tool list
        2. Router yields 'execute' requests when tools are invoked
        3. Project sends 'result' or 'error' responses
        """
        manager = get_stream_executor_manager()
        project_id: str | None = None
        send_queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        stream_done = manager.track_stream(send_queue)

        # Start background task to read incoming messages from project
        from contextunity.core.authz.context import get_auth_context

        auth_ctx = get_auth_context()
        if auth_ctx is None:
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                "ToolExecutorStream requires a verified auth token",
            )

        reader_task = asyncio.create_task(
            self.stream_reader(request_iterator, manager, send_queue, context, auth_ctx)
        )

        try:
            # Yield outgoing messages to project
            while not context.cancelled():
                try:
                    message = await asyncio.wait_for(send_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    unit = ContextUnit(
                        payload={"action": "keepalive"},
                        provenance=["router:stream_keepalive"],
                    )
                    yield unit.to_protobuf(contextunit_pb2)
                    continue

                # Check if reader signalled shutdown
                if message is None:
                    break

                # Extract project_id from ready message for logging
                if message.get("action") == "_registered":
                    pid = message.get("project_id")
                    project_id = pid if isinstance(pid, str) else None
                    continue

                unit = ContextUnit(
                    payload=message,
                    provenance=["router:stream_executor"],
                )
                yield unit.to_protobuf(contextunit_pb2)

        except asyncio.CancelledError:
            logger.info(
                "ToolExecutorStream sender stopped (server shutdown/cancel): project_id=%s",
                project_id or "unknown",
            )
        except Exception as e:  # graceful-degrade: stream cleanup must not crash
            logger.warning(
                "ToolExecutorStream sender ended: project=%s error=%s",
                project_id or "unknown",
                e,
            )
        finally:
            _ = reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
            if project_id:
                _ = manager.unregister(project_id, send_queue=send_queue)
                logger.info(
                    "ToolExecutorStream: project '%s' disconnected",
                    project_id,
                )
            manager.untrack_stream(send_queue, stream_done)

    async def stream_reader(
        self: StreamMixin,
        request_iterator: AsyncIterator[contextunit_pb2.ContextUnit],
        manager: StreamExecutorManager,
        send_queue: asyncio.Queue[dict[str, object] | None],
        context: ServicerContext[contextunit_pb2.ContextUnit, contextunit_pb2.ContextUnit],
        auth_ctx: VerifiedAuthContext,
    ) -> None:
        """Read incoming messages from project stream.

        Runs as background task. Processes ready, result, error, heartbeat.
        """
        project_id: str | None = None

        try:
            async for msg in request_iterator:
                from contextunity.core.sdk.payload import wire_payload_from_field

                payload = wire_payload_from_field(msg.payload)
                action_obj = payload.get("action", "")
                action = str(action_obj) if action_obj is not None else ""

                if action == "ready":
                    project_id_raw = payload.get("project_id", "")
                    project_id = str(project_id_raw) if project_id_raw else ""
                    tool_names_raw = payload.get("tools", [])
                    tool_names = (
                        [str(name) for name in tool_names_raw if isinstance(name, str)]
                        if is_object_list(tool_names_raw)
                        else []
                    )
                    stream_secret_raw = payload.get("stream_secret", "")
                    stream_secret = str(stream_secret_raw) if stream_secret_raw else ""

                    if not project_id or not tool_names:
                        logger.warning(
                            "ToolExecutorStream: invalid ready — project_id=%s tools=%s",
                            project_id,
                            tool_names,
                        )
                        continue

                    token = auth_ctx.token
                    token_project = auth_ctx.project_id or ""
                    if token_project and token_project != project_id:
                        await context.abort(
                            grpc.StatusCode.PERMISSION_DENIED,
                            "ToolExecutorStream token project does not match ready.project_id",
                        )
                    elif not token.can_access_tenant(project_id):
                        await context.abort(
                            grpc.StatusCode.PERMISSION_DENIED,
                            "ToolExecutorStream token is not scoped to ready.project_id",
                        )
                    elif not (
                        token.has_permission("stream:executor")
                        or token.has_permission(f"stream:executor:{project_id}")
                    ):
                        await context.abort(
                            grpc.StatusCode.PERMISSION_DENIED,
                            "ToolExecutorStream requires stream executor permission",
                        )
                    else:
                        # Secret is generated on every RegisterManifest and cached
                        # both in memory and Redis so restarts or multi-router
                        # deployments can accept the project reconnect.

                        stored = self.get_cached_stream_secret(project_id)
                        if not stored:
                            stored = get_project_stream_secret(project_id)
                            if stored:
                                self.put_cached_stream_secret(project_id, stored)

                        if not stored:
                            logger.warning(
                                "ToolExecutorStream: no stored secret for project '%s' — register or re-register tools first.",
                                project_id,
                            )
                            await send_queue.put(
                                {
                                    "action": "error",
                                    "error": (
                                        f"No stream secret found for project '{project_id}'. "
                                        "Register tools first to obtain a stream_secret."
                                    ),
                                }
                            )
                            continue

                        if not secrets.compare_digest(stored.encode(), stream_secret.encode()):
                            logger.warning(
                                "ToolExecutorStream: authentication FAILED for project '%s' — stream_secret mismatch.",
                                project_id,
                            )
                            await send_queue.put(
                                {
                                    "action": "error",
                                    "error": (
                                        f"Stream authentication failed for project '{project_id}'. "
                                        "Invalid stream_secret. Re-register tools to obtain a new secret."
                                    ),
                                }
                            )
                            continue

                        # Validate that the requested tool_names are a subset
                        # of the tools registered for this project. Without this
                        # guard, a stale or malicious executor could subscribe to
                        # arbitrary tool names, even tools registered for other
                        # projects, breaking the per-project isolation contract.
                        registered_tools = set(self._project_tools.get(project_id, []))
                        requested_tools = set(tool_names)
                        extra_tools = requested_tools - registered_tools
                        if extra_tools:
                            logger.warning(
                                (
                                    "ToolExecutorStream: project '%s' ready rejected — "
                                    "tools %s are not registered for this project."
                                ),
                                project_id,
                                sorted(extra_tools),
                            )
                            await send_queue.put(
                                {
                                    "action": "error",
                                    "error": (
                                        f"Tool names {sorted(extra_tools)} are not registered "
                                        f"for project '{project_id}'. Re-register to expose them."
                                    ),
                                }
                            )
                            continue

                        logger.info(
                            "ToolExecutorStream: project '%s' authenticated",
                            project_id,
                        )

                        _ = manager.register(project_id, tool_names, send_queue)
                        logger.info(
                            "ToolExecutorStream: project '%s' authenticated and ready, tools=%s",
                            project_id,
                            tool_names,
                        )

                        # Signal the sender about the project
                        await send_queue.put(
                            {
                                "action": "_registered",
                                "project_id": project_id,
                            }
                        )

                elif action in ("result", "error"):
                    request_id_raw = payload.get("request_id", "")
                    request_id = str(request_id_raw) if request_id_raw else ""
                    if not project_id or not request_id:
                        continue

                    if action == "error":
                        error_msg_raw = payload.get("error", "Unknown error")
                        error_msg = str(error_msg_raw) if error_msg_raw else "Unknown error"
                        manager.resolve_result(
                            project_id,
                            request_id,
                            {"error": error_msg},
                        )
                        logger.warning(
                            "ToolExecutorStream: error from '%s' req=%s: %s",
                            project_id,
                            request_id,
                            error_msg,
                        )
                    else:
                        manager.resolve_result(
                            project_id,
                            request_id,
                            {str(key): value for key, value in payload.items()},
                        )
                        logger.info(
                            "ToolExecutorStream: result from '%s' req=%s rows=%s",
                            project_id,
                            request_id,
                            payload.get("row_count", "?"),
                        )

                elif action == "heartbeat":
                    pass

                else:
                    logger.warning(
                        "ToolExecutorStream: unknown action '%s'",
                        action,
                    )

        except asyncio.CancelledError:
            logger.info(
                "ToolExecutorStream reader stopped (server shutdown/cancel): project_id=%s",
                project_id or "unknown",
            )
            return
        except Exception as e:  # graceful-degrade: stream cleanup must not crash
            logger.warning(
                "ToolExecutorStream reader ended: project=%s error=%s",
                project_id or "unknown",
                e,
            )
        finally:
            # Signal sender to stop
            with contextlib.suppress(asyncio.QueueFull):
                send_queue.put_nowait(None)


__all__ = ["StreamMixin"]
