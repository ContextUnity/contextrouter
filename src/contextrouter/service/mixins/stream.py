"""Stream mixin — ToolExecutorStream bidi stream handler."""

from __future__ import annotations

import asyncio
import secrets

from contextcore import ContextUnit, context_unit_pb2, get_context_unit_logger
from google.protobuf.json_format import MessageToDict

from contextrouter.service.stream_executors import get_stream_executor_manager

logger = get_context_unit_logger(__name__)


class StreamMixin:
    """Mixin providing ToolExecutorStream bidi RPC handler."""

    async def ToolExecutorStream(self, request_iterator, context):
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
        send_queue: asyncio.Queue = asyncio.Queue()

        # Start background task to read incoming messages from project
        reader_task = asyncio.create_task(
            self._stream_reader(request_iterator, manager, send_queue)
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
                    yield unit.to_protobuf(context_unit_pb2)
                    continue

                # Check if reader signalled shutdown
                if message is None:
                    break

                # Extract project_id from ready message for logging
                if message.get("action") == "_registered":
                    project_id = message.get("project_id")
                    continue

                unit = ContextUnit(
                    payload=message,
                    provenance=["router:stream_executor"],
                )
                yield unit.to_protobuf(context_unit_pb2)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(
                "ToolExecutorStream sender ended: project=%s error=%s",
                project_id or "unknown",
                e,
            )
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
            if project_id:
                manager.unregister(project_id)
                logger.info(
                    "ToolExecutorStream: project '%s' disconnected",
                    project_id,
                )

    async def _stream_reader(
        self,
        request_iterator,
        manager,
        send_queue: asyncio.Queue,
    ):
        """Read incoming messages from project stream.

        Runs as background task. Processes ready, result, error, heartbeat.
        """
        project_id: str | None = None

        try:
            async for msg in request_iterator:
                payload = MessageToDict(msg.payload)
                action = payload.get("action", "")

                if action == "ready":
                    project_id = payload.get("project_id", "")
                    tool_names = payload.get("tools", [])
                    stream_secret = payload.get("stream_secret", "")

                    if not project_id or not tool_names:
                        logger.warning(
                            "ToolExecutorStream: invalid ready — project_id=%s tools=%s",
                            project_id,
                            tool_names,
                        )
                        continue

                    # ── Verify stream secret (one-time use) ──────────
                    # Secret was generated in RegisterTools and stored
                    # in self._stream_secrets.  After successful verify,
                    # the secret is DELETED (consumed) → per-registration.
                    # Reconnect requires re-registration → fresh secret.

                    with self._stream_secrets_lock:
                        stored = self._stream_secrets.get(project_id)

                    if not stored:
                        logger.warning(
                            "ToolExecutorStream: no stored secret for "
                            "project '%s' — register or re-register tools first.",
                            project_id,
                        )
                        await send_queue.put(
                            {
                                "action": "error",
                                "error": (
                                    f"No stream secret found for project "
                                    f"'{project_id}'. Register tools first "
                                    f"to obtain a stream_secret."
                                ),
                            }
                        )
                        continue

                    if not secrets.compare_digest(stored.encode(), stream_secret.encode()):
                        logger.warning(
                            "ToolExecutorStream: authentication FAILED for "
                            "project '%s' — stream_secret mismatch.",
                            project_id,
                        )
                        await send_queue.put(
                            {
                                "action": "error",
                                "error": (
                                    f"Stream authentication failed for project "
                                    f"'{project_id}'. Invalid stream_secret. "
                                    f"Re-register tools to obtain a new secret."
                                ),
                            }
                        )
                        continue

                    # One-time use: consume the secret (thread-safe)
                    with self._stream_secrets_lock:
                        self._stream_secrets.pop(project_id, None)
                    logger.info(
                        "ToolExecutorStream: project '%s' authenticated "
                        "(secret consumed, one-time use)",
                        project_id,
                    )

                    manager.register(project_id, tool_names, send_queue)
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
                    request_id = payload.get("request_id", "")
                    if not project_id or not request_id:
                        continue

                    if action == "error":
                        error_msg = payload.get("error", "Unknown error")
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
                        manager.resolve_result(project_id, request_id, payload)
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

        except Exception as e:
            logger.warning(
                "ToolExecutorStream reader ended: project=%s error=%s",
                project_id or "unknown",
                e,
            )
        finally:
            # Signal sender to stop
            await send_queue.put(None)


__all__ = ["StreamMixin"]
