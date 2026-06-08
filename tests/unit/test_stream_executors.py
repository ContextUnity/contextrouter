"""Unit tests for StreamExecutorManager and bidi tool execution.

Tests use real objects (no MagicMock for auth):
  - Real ContextToken via set_current_access_token (no mock.patch)
  - Real asyncio.Queue (no FakeQueue needed — Queue IS the real impl)
  - Real StreamExecutorManager (not mocked)
"""

import asyncio
import threading

import pytest
from contextunity.core import ContextUnit, contextunit_pb2
from contextunity.core.tokens import ContextToken

from contextunity.router.core.context import (
    reset_current_access_token,
    set_current_access_token,
)
from contextunity.router.service.mixins.stream import StreamMixin
from contextunity.router.service.stream_executors import StreamExecutorManager


@pytest.fixture
def manager():
    return StreamExecutorManager()


@pytest.fixture
def dummy_queue():
    return asyncio.Queue()


@pytest.fixture
def _set_caller_token():
    """Set a real ContextToken in contextvars — no MagicMock."""
    token = ContextToken(
        token_id="stream-test",
        user_id="test_user",
        allowed_tenants=("test_tenant",),
        permissions=("stream:executor",),
    )
    ref = set_current_access_token(token)
    yield token
    reset_current_access_token(ref)


@pytest.mark.asyncio
class TestStreamExecutorRegistration:
    async def test_register_executor(self, manager, dummy_queue):
        """Register adds executor and makes it available."""
        executor = manager.register("test_project", ["tool_A", "tool_B"], dummy_queue)

        assert executor.project_id == "test_project"
        assert "tool_A" in executor.tool_names
        assert manager.is_available("test_project", "tool_A") is True
        assert manager.is_available("test_project", "tool_C") is False
        assert manager.is_available("other_project", "tool_A") is False

    async def test_unregister_executor(self, manager, dummy_queue):
        """Unregister removes executor AND cancels pending futures."""
        manager.register("test_project", ["tool_A"], dummy_queue)
        assert manager.is_available("test_project", "tool_A") is True

        executor = manager._executors["test_project"]
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        executor.pending["req-123"] = type("PendingRequest", (), {"future": future})()

        manager.unregister("test_project")

        # Behavioral: tool no longer available
        assert manager.is_available("test_project", "tool_A") is False
        # Behavioral: pending future resolved with error
        assert future.done() is True
        with pytest.raises(ConnectionError):
            future.result()

    async def test_unregister_idempotent(self, manager):
        """Unregistering a non-existent project does not raise."""
        manager.unregister("nonexistent")  # no error
        assert manager.is_available("nonexistent", "any") is False

    async def test_register_replaces_previous(self, manager, dummy_queue):
        """Re-registering same project replaces the previous executor."""
        manager.register("proj", ["tool_A"], dummy_queue)
        q2 = asyncio.Queue()
        manager.register("proj", ["tool_B"], q2)

        assert manager.is_available("proj", "tool_B") is True
        assert manager.is_available("proj", "tool_A") is False
        assert dummy_queue.get_nowait() is None

    async def test_old_stream_cleanup_does_not_remove_replacement(self, manager):
        old_queue = asyncio.Queue()
        new_queue = asyncio.Queue()
        manager.register("proj", ["old"], old_queue)
        replacement = manager.register("proj", ["new"], new_queue)

        assert manager.unregister("proj", send_queue=old_queue) is False
        assert manager.get_executor("proj") is replacement
        assert manager.is_available("proj", "new") is True

    async def test_drain_all(self, manager, dummy_queue):
        """Drain sends shutdown sentinels to all queues."""
        manager.register("proj1", ["tool"], dummy_queue)
        queue2 = asyncio.Queue()
        manager.register("proj2", ["tool"], queue2)

        await manager.drain_all()

        assert dummy_queue.get_nowait() is None
        assert queue2.get_nowait() is None

    async def test_drain_all_tracks_unregistered_streams(self, manager):
        """Shutdown drains raw streams even before they register an executor."""
        queue = asyncio.Queue()
        done = manager.track_stream(queue)

        async def close_stream():
            assert await queue.get() is None
            manager.untrack_stream(queue, done)

        closer = asyncio.create_task(close_stream())
        await manager.drain_all()
        await closer
        assert done.is_set()

    async def test_track_untrack_lifecycle(self, manager):
        """track_stream → untrack_stream marks done and removes stream."""
        q = asyncio.Queue()
        done = manager.track_stream(q)

        assert not done.is_set()
        assert id(q) in manager._streams

        manager.untrack_stream(q, done)

        assert done.is_set()
        assert id(q) not in manager._streams


@pytest.mark.asyncio
class TestStreamExecutorExecution:
    async def test_execute_success(self, manager, dummy_queue, _set_caller_token):
        """Full execute lifecycle: dispatch → resolve → verify result."""
        manager.register("test_project", ["tool_A"], dummy_queue)

        async def mock_project_response():
            msg = await dummy_queue.get()
            # Behavioral: message has correct structure
            assert msg["action"] == "execute"
            assert msg["tool"] == "tool_A"
            assert msg["args"] == {"param": 1}
            assert msg["caller_tenant"] == "test_tenant"
            assert msg["user_id"] == "test_user"
            assert "request_id" in msg
            manager.resolve_result(
                "test_project", msg["request_id"], {"status": "ok", "data": "value"}
            )

        result, _ = await asyncio.gather(
            manager.execute("test_project", "tool_A", {"param": 1}), mock_project_response()
        )

        assert result == {"status": "ok", "data": "value"}

    async def test_execute_cleans_pending_after_success(
        self, manager, dummy_queue, _set_caller_token
    ):
        """After successful execution, request_id is removed from pending."""
        manager.register("test_project", ["tool_A"], dummy_queue)

        async def respond():
            msg = await dummy_queue.get()
            manager.resolve_result("test_project", msg["request_id"], {"ok": True})

        await asyncio.gather(manager.execute("test_project", "tool_A", {}), respond())

        executor = manager.get_executor("test_project")
        assert len(executor.pending) == 0

    async def test_execute_rejects_missing_tenant(self, manager, dummy_queue):
        """Rejects execution if caller has no tenant (SecurityError, fail-closed)."""
        manager.register("test_project", ["tool_A"], dummy_queue)

        # Set token with empty tenants
        token = ContextToken(
            token_id="no-tenant", user_id="user", allowed_tenants=(), permissions=()
        )
        ref = set_current_access_token(token)
        try:
            from contextunity.core.exceptions import SecurityError

            with pytest.raises(SecurityError):
                await manager.execute("test_project", "tool_A", {})
        finally:
            reset_current_access_token(ref)

    async def test_execute_rejects_no_token(self, manager, dummy_queue):
        """Rejects execution if no token is set at all (fail-closed)."""
        manager.register("test_project", ["tool_A"], dummy_queue)
        # No token set — fail-closed
        from contextunity.core.exceptions import SecurityError

        with pytest.raises(SecurityError):
            await manager.execute("test_project", "tool_A", {})

    async def test_execute_timeout(self, manager, dummy_queue, _set_caller_token):
        """Raises TimeoutError if project doesn't respond in time."""
        manager.register("test_project", ["tool_A"], dummy_queue)

        with pytest.raises(TimeoutError):
            await manager.execute("test_project", "tool_A", {"param": 1}, timeout=0.05)

    async def test_execute_not_available(self, manager):
        """Raises ConnectionError if project or tool is not active."""
        with pytest.raises(ConnectionError):
            await manager.execute("unknown_project", "tool_A", {})

    async def test_resolve_result_unknown_project(self, manager):
        """Resolving for unknown project is a no-op (graceful)."""
        manager.resolve_result("nonexistent", "req-1", {"data": "x"})  # no error
        assert "nonexistent" not in manager._executors

    async def test_resolve_result_unknown_request(self, manager, dummy_queue):
        """Resolving for unknown request_id in known project is a no-op."""
        manager.register("proj", ["tool"], dummy_queue)
        manager.resolve_result("proj", "unknown-req", {"data": "x"})  # no error
        assert not manager._executors["proj"].pending  # no leftover state

    async def test_resolve_result_delivers_correct_payload(
        self, manager, dummy_queue, _set_caller_token
    ):
        """resolve_result delivers the exact payload to the waiting future."""
        manager.register("proj", ["tool"], dummy_queue)

        async def respond():
            msg = await dummy_queue.get()
            manager.resolve_result(
                "proj", msg["request_id"], {"key": "specific_value", "count": 42}
            )

        result, _ = await asyncio.gather(manager.execute("proj", "tool", {}), respond())

        assert result["key"] == "specific_value"
        assert result["count"] == 42

    async def test_resolve_result_rejects_invalid_payload(
        self, manager, dummy_queue, _set_caller_token
    ):
        """Invalid stream payloads fail the waiting future instead of resolving."""
        manager.register("proj", ["tool"], dummy_queue)

        async def respond_bad():
            msg = await dummy_queue.get()
            manager.resolve_result("proj", msg["request_id"], {"bad": object()})

        with pytest.raises(ValueError, match="Invalid stream result"):
            await asyncio.gather(manager.execute("proj", "tool", {}), respond_bad())


class _AsyncIterator:
    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


@pytest.mark.asyncio
class TestToolExecutorStreamHandshake:
    def _build_mixin(self):
        mixin = StreamMixin.__new__(StreamMixin)
        mixin._stream_secrets = {}
        mixin._stream_secrets_lock = threading.Lock()
        # Mirror DispatcherService's per-project tool map. Tests that
        # exercise the ready.tools path must populate this; defaulting
        # to the requested tool set keeps the legacy handshake tests
        # passing without weakening the validation.
        mixin._project_tools = {"proj-1": ["tool_A"]}
        return mixin

    def _auth_ctx(self, project_id="proj-1"):
        from contextunity.core.authz.context import VerifiedAuthContext
        from contextunity.core.tokens import ContextToken

        token = ContextToken(
            token_id="stream-token",
            permissions=("stream:executor", f"stream:executor:{project_id}"),
            allowed_tenants=(project_id,),
        )
        return VerifiedAuthContext.from_token(token, "token", project_id=project_id)

    def _context(self):
        """Real-enough gRPC context with abort that raises RuntimeError."""
        from unittest.mock import MagicMock

        context = MagicMock()

        async def _abort(code, details):
            raise RuntimeError(details)

        context.abort.side_effect = _abort
        return context

    async def test_stream_reader_registers_project_on_valid_secret(self):
        manager = StreamExecutorManager()
        mixin = self._build_mixin()
        mixin._stream_secrets["proj-1"] = "secret-1"
        send_queue = asyncio.Queue()
        request_iterator = _AsyncIterator(
            [
                ContextUnit(
                    payload={
                        "action": "ready",
                        "project_id": "proj-1",
                        "tools": ["tool_A"],
                        "stream_secret": "secret-1",
                    }
                ).to_protobuf(contextunit_pb2)
            ]
        )

        await mixin.stream_reader(
            request_iterator,
            manager,
            send_queue,
            self._context(),
            self._auth_ctx(),
        )

        executor = manager.get_executor("proj-1")
        assert executor is not None
        assert executor.tool_names == ["tool_A"]
        assert mixin._stream_secrets["proj-1"] == "secret-1"
        assert await send_queue.get() == {"action": "_registered", "project_id": "proj-1"}
        assert await send_queue.get() is None

    async def test_stream_reader_registers_project_with_redis_secret_fallback(self):
        from unittest.mock import patch

        manager = StreamExecutorManager()
        mixin = self._build_mixin()
        send_queue = asyncio.Queue()
        request_iterator = _AsyncIterator(
            [
                ContextUnit(
                    payload={
                        "action": "ready",
                        "project_id": "proj-1",
                        "tools": ["tool_A"],
                        "stream_secret": "secret-1",
                    }
                ).to_protobuf(contextunit_pb2)
            ]
        )

        with patch(
            "contextunity.router.service.mixins.stream.get_project_stream_secret",
            return_value="secret-1",
        ):
            await mixin.stream_reader(
                request_iterator,
                manager,
                send_queue,
                self._context(),
                self._auth_ctx(),
            )

        executor = manager.get_executor("proj-1")
        assert executor is not None
        assert mixin._stream_secrets["proj-1"] == "secret-1"
        assert await send_queue.get() == {"action": "_registered", "project_id": "proj-1"}
        assert await send_queue.get() is None

    async def test_stream_reader_rejects_invalid_secret(self):
        manager = StreamExecutorManager()
        mixin = self._build_mixin()
        mixin._stream_secrets["proj-1"] = "secret-1"
        send_queue = asyncio.Queue()
        request_iterator = _AsyncIterator(
            [
                ContextUnit(
                    payload={
                        "action": "ready",
                        "project_id": "proj-1",
                        "tools": ["tool_A"],
                        "stream_secret": "wrong-secret",
                    }
                ).to_protobuf(contextunit_pb2)
            ]
        )

        await mixin.stream_reader(
            request_iterator,
            manager,
            send_queue,
            self._context(),
            self._auth_ctx(),
        )

        assert manager.get_executor("proj-1") is None
        message = await send_queue.get()
        assert message["action"] == "error"
        assert await send_queue.get() is None


# ── validate_stream_result ────────────────────────────────────────────────


class TestValidateStreamResult:
    def test_valid_flat_dict(self):
        from contextunity.router.service.stream_result import validate_stream_result

        result = validate_stream_result({"answer": "hello", "count": 1})
        assert result == {"answer": "hello", "count": 1}

    def test_valid_nested_dict(self):
        from contextunity.router.service.stream_result import validate_stream_result

        result = validate_stream_result({"data": {"nested": {"value": True}}})
        assert result["data"]["nested"]["value"] is True

    def test_valid_list_value(self):
        from contextunity.router.service.stream_result import validate_stream_result

        result = validate_stream_result({"items": ["a", "b", 3]})
        assert result["items"] == ["a", "b", 3]

    def test_rejects_non_dict(self):
        from contextunity.router.service.stream_result import validate_stream_result

        with pytest.raises(ValueError, match="must be a JSON object"):
            validate_stream_result(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_rejects_too_many_top_level_keys(self):
        from contextunity.router.service.stream_result import validate_stream_result

        oversized = {str(i): i for i in range(257)}
        with pytest.raises(ValueError, match="exceeds maximum key count"):
            validate_stream_result(oversized)

    def test_rejects_nested_dict_too_large(self):
        from contextunity.router.service.stream_result import validate_stream_result

        big_nested = {str(i): i for i in range(257)}
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_stream_result({"data": big_nested})

    def test_rejects_list_too_large(self):
        from contextunity.router.service.stream_result import validate_stream_result

        big_list = list(range(257))
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_stream_result({"items": big_list})

    def test_rejects_excess_nesting_depth(self):
        from contextunity.router.service.stream_result import validate_stream_result

        deep: dict[str, object] = {"v": None}
        for _ in range(10):
            deep = {"nested": deep}
        with pytest.raises(ValueError, match="maximum nesting depth"):
            validate_stream_result(deep)

    def test_rejects_unsupported_value_type(self):
        from contextunity.router.service.stream_result import _validate_stream_value

        with pytest.raises(ValueError, match="Unsupported stream result value type"):
            _validate_stream_value(object(), depth=1)

    def test_returns_copy(self):
        from contextunity.router.service.stream_result import validate_stream_result

        original = {"key": "value"}
        result = validate_stream_result(original)
        result["key"] = "mutated"
        assert original["key"] == "value"
