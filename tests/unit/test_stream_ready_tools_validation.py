"""Tests for (ready.tools subset validation) in StreamMixin.

when a project executor sends a ``ready`` action over the
ToolExecutorStream, the ``tool_names`` it requests must be a subset of
the tools registered for that project via ``RegisterManifest``. Otherwise
a stale or malicious executor could subscribe to arbitrary tool names,
breaking the per-project isolation contract.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
from contextunity.core import contextunit_pb2

from contextunity.router.service.mixins.stream import StreamMixin


def _make_ready_message(
    project_id: str = "proj_x",
    tool_names: list[str] | None = None,
    stream_secret: str = "secret-abc",
) -> contextunit_pb2.ContextUnit:
    msg = contextunit_pb2.ContextUnit()
    payload = {
        "action": "ready",
        "project_id": project_id,
        "tools": tool_names or [],
        "stream_secret": stream_secret,
    }
    msg.payload.update(payload)
    return msg


class _FakeStreamHost(StreamMixin):
    """Minimal host with the attributes StreamMixin needs for validation."""

    def __init__(self, project_tools: dict[str, list[str]], stream_secret: str) -> None:
        self._project_tools = project_tools
        self.put_cached_stream_secret("proj_x", stream_secret)


async def _collect_send_queue(q: asyncio.Queue) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.mark.asyncio
async def test_ready_with_unregistered_tool_is_rejected() -> None:
    """tools not in _project_tools[project_id] must be rejected."""
    project_tools = {"proj_x": ["registered_tool"]}
    host = _FakeStreamHost(project_tools, "secret-abc")

    # auth_ctx is bypassed by passing a mock that satisfies the early checks.
    auth_ctx = MagicMock()
    auth_ctx.project_id = "proj_x"
    # Bypass token check by providing a token with admin:all
    auth_ctx.token.has_permission.return_value = True
    auth_ctx.token.can_access_tenant.return_value = True

    # Send ready with one unknown tool
    msg = _make_ready_message(tool_names=["unknown_tool"])
    sent: asyncio.Queue = asyncio.Queue()

    # Patch manager.register so the test fails if validation lets it through
    manager = MagicMock()

    async def request_iter() -> AsyncIterator[contextunit_pb2.ContextUnit]:
        yield msg
        # After yielding, the stream reader will validate and either
        # call manager.register or send an error to sent queue.
        await asyncio.sleep(0)
        # Close
        return
        yield  # pragma: no cover  # for type checker

    ctx = MagicMock()

    # Run one pass of stream_reader
    await host.stream_reader(request_iter(), manager, sent, ctx, auth_ctx)
    errors = [m for m in list(sent._queue) if isinstance(m, dict) and m.get("action") == "error"]
    assert any("unknown_tool" in str(e.get("error", "")) for e in errors), (
        f"Expected validation error mentioning 'unknown_tool', got: {list(sent._queue)}"
    )
    manager.register.assert_not_called()


@pytest.mark.asyncio
async def test_ready_with_subset_of_registered_tools_passes() -> None:
    """tools fully within _project_tools[project_id] must be accepted."""
    project_tools = {"proj_x": ["alpha", "beta", "gamma"]}
    host = _FakeStreamHost(project_tools, "secret-abc")

    auth_ctx = MagicMock()
    auth_ctx.project_id = "proj_x"
    auth_ctx.token.has_permission.return_value = True
    auth_ctx.token.can_access_tenant.return_value = True

    msg = _make_ready_message(tool_names=["alpha", "beta"])
    sent: asyncio.Queue = asyncio.Queue()
    manager = MagicMock()

    async def request_iter() -> AsyncIterator[contextunit_pb2.ContextUnit]:
        yield msg
        await asyncio.sleep(0)
        return
        yield  # pragma: no cover

    ctx = MagicMock()
    await host.stream_reader(request_iter(), manager, sent, ctx, auth_ctx)

    # Manager.register was called with the requested tools
    manager.register.assert_called_once()
    call_args = manager.register.call_args
    assert call_args.args[0] == "proj_x"
    assert set(call_args.args[1]) == {"alpha", "beta"}
