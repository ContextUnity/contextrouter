"""Test that ``log_execution_trace`` forwards ``record_episode=not stream``.

Stream runs are incremental (one trace per progress tick) so they should
skip the extra gRPC write to Brain's episodic store. Unary runs always
record the episode. The previous behavior ignored the ``stream`` flag
entirely, doing the extra work on every tick.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextunity.core import ContextToken

from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.service.mixins.execution.helpers import log_execution_trace
from contextunity.router.service.mixins.execution.types import ExecutionMetadata
from contextunity.router.service.payloads import ExecuteAgentPayload


def _parent_token() -> ContextToken:
    return ContextToken(
        token_id="t1",
        permissions=("tool:read",),
        allowed_tenants=("tenant-1",),
    )


def _params() -> ExecuteAgentPayload:
    return ExecuteAgentPayload(
        agent_id="test_agent",
        input={"messages": []},
    )


@pytest.mark.asyncio
@patch("contextunity.router.modules.tools.brain_trace_tools._get_brain_client")
async def test_unary_run_records_episode(mock_get_client: MagicMock) -> None:
    """Default (non-stream) call must pass record_episode=True."""
    mock_client = AsyncMock()
    mock_client.log_trace.return_value = "trace-1"
    mock_get_client.return_value = mock_client

    await log_execution_trace(
        auto_tracer=BrainAutoTracer(),
        result={"final_output": "ok"},
        token=_parent_token(),
        tenant_id="tenant-1",
        params=_params(),
        metadata=ExecutionMetadata(),
        effective_user_id="user-1",
        graph_name="test",
        wall_ms=100,
        last_user_msg="hi",
        guard_result=None,
        execution_input={"messages": []},
        stream=False,
    )

    mock_client.log_trace.assert_called_once()
    # add_episode is invoked when record_episode=True (the default for unary).
    mock_client.add_episode.assert_called_once()


@pytest.mark.asyncio
@patch("contextunity.router.modules.tools.brain_trace_tools._get_brain_client")
async def test_stream_run_skips_episode_recording(mock_get_client: MagicMock) -> None:
    """stream=True must pass record_episode=False, skipping add_episode."""
    mock_client = AsyncMock()
    mock_client.log_trace.return_value = "trace-2"
    mock_get_client.return_value = mock_client

    await log_execution_trace(
        auto_tracer=BrainAutoTracer(),
        result={"final_output": "ok"},
        token=_parent_token(),
        tenant_id="tenant-1",
        params=_params(),
        metadata=ExecutionMetadata(),
        effective_user_id="user-1",
        graph_name="test",
        wall_ms=100,
        last_user_msg="hi",
        guard_result=None,
        execution_input={"messages": []},
        stream=True,
    )

    mock_client.log_trace.assert_called_once()
    # Stream runs must NOT record an episode per tick.
    mock_client.add_episode.assert_not_called()
