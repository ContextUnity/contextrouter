"""Unit tests for brain_trace_tools and log_execution_trace fallback logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextunity.router.modules.tools.brain_trace_tools import (
    _get_auth_token_string,
    log_execution_trace,
)
from contextunity.router.service.mixins.execution.helpers import BrainAutoTracer


@pytest.fixture
def dummy_auto_tracer():
    class DummyAutoTracer(BrainAutoTracer):
        def get_token_usage(self):
            return {"input_tokens": 10, "output_tokens": 20, "total_cost": 0.05}

        def get_nested_steps(self):
            return [{"tool": "search", "timing_ms": 100}]

        def get_tool_calls_summary(self):
            return [{"name": "search", "args": {}}]

    return DummyAutoTracer()


class TestBrainTraceToolsTokenExtraction:
    def test_get_auth_token_string_success(self):
        """Should resolve the token string correctly from the runtime context."""
        with patch("contextunity.core.authz.context.get_auth_context") as mock_ctx:
            mock_ctx.return_value = MagicMock(token_string="mock_token_123")
            token = _get_auth_token_string()
            assert token == "mock_token_123"

    def test_get_auth_token_string_none(self):
        """Should return None if get_auth_context returns None."""
        with patch("contextunity.core.authz.context.get_auth_context") as mock_ctx:
            mock_ctx.return_value = None
            token = _get_auth_token_string()
            assert token is None

    def test_get_auth_token_string_exception(self):
        """Exception during resolution should gracefully return None."""
        with patch("contextunity.core.authz.context.get_auth_context") as mock_ctx:
            mock_ctx.side_effect = RuntimeError("Context failure")
            token = _get_auth_token_string()
            assert token is None


@pytest.mark.asyncio
class TestLogExecutionTrace:
    @patch("contextunity.router.modules.tools.brain_trace_tools._get_brain_client")
    async def test_trace_success(self, mock_get_client, dummy_auto_tracer):
        """Ensure log_execution_trace succeeds and returns trace_id."""
        mock_client = AsyncMock()

        async def mock_log(*args, **kwargs):
            return "tr-123"

        async def mock_episode(*args, **kwargs):
            return type("EpisodeResponse", (), {"episode_id": "ep-123"})()

        mock_client.log_trace = mock_log
        mock_client.add_episode = mock_episode
        mock_get_client.return_value = mock_client

        result = await log_execution_trace.coroutine(
            tenant_id="default",
            agent_id="test_agent",
            session_id="session-123",
            user_id="user1",
            graph_name="test_graph",
            tool_calls=[],
            token_usage={},
            timing_ms=100,
            steps=[],
            platform="grpc",
            model_key="gpt-4o",
            iterations=1,
            message_count=5,
            user_query="hi",
            final_answer="hello",
            metadata={},
            security_flags=[],
            record_episode=True,
        )

        assert result["success"] is True
        assert result["trace_id"] == "tr-123"
        assert result["tenant_id"] == "default"

    @patch("contextunity.router.modules.tools.brain_trace_tools._get_brain_client")
    async def test_trace_graceful_failure(self, mock_get_client):
        """Trace fails gracefully (e.g., PERMISSION_DENIED) returning success: False."""
        from unittest.mock import AsyncMock

        from contextunity.core.exceptions import ContextUnityError

        mock_client = AsyncMock()

        async def mock_log_fail(*args, **kwargs):
            raise ContextUnityError(code="DENIED", details="Fake failure")

        mock_client.log_trace = mock_log_fail
        mock_get_client.return_value = mock_client

        result = await log_execution_trace.coroutine(
            tenant_id="default",
            agent_id="test_agent",
            session_id="",
            user_id="",
            graph_name="",
            tool_calls=[],
            token_usage={},
            timing_ms=0,
            steps=[],
            platform="",
            model_key="",
            iterations=1,
            message_count=0,
            user_query="",
            final_answer="",
            metadata={},
            security_flags=[],
            record_episode=False,
        )

        assert result["success"] is False
        assert "Fake failure" in result["error"]
