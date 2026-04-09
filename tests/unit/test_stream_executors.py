"""Unit tests for StreamExecutorManager and bidi tool execution."""

import asyncio
from unittest.mock import MagicMock

import pytest

from contextrouter.service.stream_executors import StreamExecutorManager


@pytest.fixture
def manager():
    return StreamExecutorManager()


@pytest.fixture
def dummy_queue():
    return asyncio.Queue()


@pytest.mark.asyncio
class TestStreamExecutorRegistration:
    async def test_register_executor(self, manager, dummy_queue):
        """Test successful registration of an executor."""
        executor = manager.register("test_project", ["tool_A", "tool_B"], dummy_queue)

        assert executor.project_id == "test_project"
        assert "tool_A" in executor.tool_names

        # Verify it's in the manager
        assert manager.is_available("test_project", "tool_A") is True
        assert manager.is_available("test_project", "tool_C") is False
        assert manager.is_available("other_project", "tool_A") is False

    async def test_unregister_executor(self, manager, dummy_queue):
        """Test unregistering an executor cancels pending futures."""
        manager.register("test_project", ["tool_A"], dummy_queue)
        assert manager.is_available("test_project", "tool_A") is True

        # Add a dummy pending future
        executor = manager._executors["test_project"]
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        executor.pending["req-123"] = type("PendingRequest", (), {"future": future})()

        # Unregister should cancel futures
        manager.unregister("test_project")

        assert manager.is_available("test_project", "tool_A") is False
        assert future.done() is True
        with pytest.raises(ConnectionError, match="disconnected"):
            future.result()

    async def test_drain_all(self, manager, dummy_queue):
        """Test drain_all sends shutdown sentinels to all queues."""
        manager.register("proj1", ["tool"], dummy_queue)
        queue2 = asyncio.Queue()
        manager.register("proj2", ["tool"], queue2)

        await manager.drain_all()

        # Both queues should have received None (shutdown sentinel)
        assert dummy_queue.get_nowait() is None
        assert queue2.get_nowait() is None


@pytest.mark.asyncio
class TestStreamExecutorExecution:
    @pytest.fixture(autouse=True)
    def mock_auth_token(self):
        from unittest.mock import patch

        with patch("contextrouter.cortex.runtime_context.get_current_access_token") as mock_get:
            mock_token = MagicMock()
            mock_token.allowed_tenants = ["test_tenant"]
            mock_token.user_id = "test_user"
            mock_token.session_id = "test_session"
            mock_get.return_value = mock_token
            yield mock_get

    async def test_execute_success(self, manager, dummy_queue):
        """Successfully dispatches and receives result."""
        manager.register("test_project", ["tool_A"], dummy_queue)

        # Mock the async execution to simulate project sending a result back
        async def mock_project_response():
            # Wait for message to be put in queue
            msg = await dummy_queue.get()
            request_id = msg.get("request_id")
            # Resolve the future directly
            manager.resolve_result("test_project", request_id, {"status": "ok", "data": "value"})

        # Run execute and mock_project_response concurrently
        result, _ = await asyncio.gather(
            manager.execute("test_project", "tool_A", {"param": 1}), mock_project_response()
        )

        assert result == {"status": "ok", "data": "value"}

    async def test_execute_timeout(self, manager, dummy_queue):
        """Raises TimeoutError if project doesn't respond in time."""
        manager.register("test_project", ["tool_A"], dummy_queue)

        with pytest.raises(TimeoutError, match="Tool execution timed out"):
            # Set a very short timeout
            await manager.execute("test_project", "tool_A", {"param": 1}, timeout=0.1)

    async def test_execute_not_available(self, manager):
        """Raises ConnectionError if project or tool is not active."""
        with pytest.raises(ConnectionError, match="No active stream"):
            await manager.execute("unknown_project", "tool_A", {})

    async def test_resolve_result_unknown(self, manager):
        """Resolving a result for an unknown request gracefully ignores it."""
        # This shouldn't raise any error
        manager.resolve_result("test_project", "unknown_req", {})
