"""Background service for always-active dispatcher agent."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError, RedisNotAvailable
from contextunity.core.parsing import json_dumps
from contextunity.core.types import is_object_list
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig

from contextunity.router.core import get_core_config
from contextunity.router.cortex.dispatcher_agent.types import (
    DispatcherState,
    is_dispatcher_state,
    make_dispatcher_state,
    validated_dispatcher_stream_event,
)
from contextunity.router.cortex.types import ExecutionMetadata, StateUpdate, extract_message_content
from contextunity.router.modules.observability import (
    LangfuseRequestCtx,
    get_langfuse_callbacks,
    trace_context,
)

logger = get_contextunit_logger(__name__)


# Global dispatcher instance
_dispatcher_instance: "DispatcherService | None" = None
_dispatcher_lock = threading.Lock()


class _DispatcherStateSnapshot(Protocol):
    """Typed subset of LangGraph state snapshots used by the dispatcher service."""

    values: dict[str, object]


@runtime_checkable
class _AGetStateMethod(Protocol):
    async def __call__(self, config: RunnableConfig, /) -> _DispatcherStateSnapshot: ...


@runtime_checkable
class _AInvokeMethod(Protocol):
    async def __call__(
        self, input: DispatcherState, /, *, config: RunnableConfig | None = None
    ) -> object: ...


@runtime_checkable
class _AStreamMethod(Protocol):
    def __call__(
        self,
        input: DispatcherState,
        /,
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[StateUpdate]: ...


class _DispatcherGraphAdapter:
    """Narrow adapter over the compiled LangGraph surface used here."""

    _graph_obj: object
    checkpointer: object | None

    def __init__(self, graph_obj: object) -> None:
        self._graph_obj = graph_obj
        self.checkpointer = getattr(graph_obj, "checkpointer", None)

    async def aget_state(self, config: RunnableConfig, /) -> _DispatcherStateSnapshot:
        method = getattr(self._graph_obj, "aget_state", None)
        if not isinstance(method, _AGetStateMethod):
            raise TypeError("Dispatcher graph missing aget_state")
        return await method(config)

    async def ainvoke(
        self,
        input: DispatcherState,
        /,
        *,
        config: RunnableConfig | None = None,
    ) -> object:
        method = getattr(self._graph_obj, "ainvoke", None)
        if not isinstance(method, _AInvokeMethod):
            raise TypeError("Dispatcher graph missing ainvoke")
        return await method(input, config=config)

    def astream(
        self,
        input: DispatcherState,
        /,
        *,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[StateUpdate]:
        method = getattr(self._graph_obj, "astream", None)
        if not isinstance(method, _AStreamMethod):
            raise TypeError("Dispatcher graph missing astream")
        return method(input, config=config)


class DispatcherService:
    """Always-active dispatcher agent service.

    This service maintains a persistent graph instance and provides
    access to the dispatcher agent via API or Python import.
    """

    def __init__(self) -> None:
        """Initialize the dispatcher service.

        The underlying LangGraph graph is compiled lazily on first use.
        """
        self._graph: _DispatcherGraphAdapter | None = None
        self._initialized: bool = False
        self._lock: threading.Lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazily compile the dispatcher graph with an appropriate checkpointer.

        Uses Redis-backed checkpointing if available, falling back to in-memory.
        Thread-safe via double-checked locking.
        """
        if self._initialized and self._graph is not None:
            return

        with self._lock:
            if self._initialized and self._graph is not None:
                return

            logger.info("Initializing dispatcher agent graph...")

            from contextunity.router.cortex.dispatcher_agent import compile_dispatcher_graph

            # Enable checkpointing for session persistence
            checkpointer = None
            config = get_core_config()

            if config.redis.enabled and config.redis.url:
                try:
                    from contextunity.router.cortex.services.redis_saver import (
                        RedisCheckpointSaver,
                    )

                    checkpointer = RedisCheckpointSaver()
                    logger.info("Checkpointing enabled with Redis")
                except ImportError:
                    from langgraph.checkpoint.memory import MemorySaver

                    checkpointer = MemorySaver()
                    logger.warning(
                        "%s: falling back to in-memory checkpointing",
                        RedisNotAvailable("redis_saver dependency is not installed"),
                    )
                except Exception as e:  # graceful-degrade: Redis checkpointer → MemorySaver
                    from langgraph.checkpoint.memory import MemorySaver

                    checkpointer = MemorySaver()
                    logger.warning(
                        (
                            "DispatcherService: failed to init Redis checkpointer: %s. "
                            "Falling back to in-memory checkpointing."
                        ),
                        e,
                        exc_info=True,
                    )
            else:
                from langgraph.checkpoint.memory import MemorySaver

                checkpointer = MemorySaver()

            compiled_graph_obj: object = compile_dispatcher_graph(checkpointer=checkpointer)
            self._graph = _DispatcherGraphAdapter(compiled_graph_obj)
            self._initialized = True
            logger.info("Dispatcher agent graph initialized")

    @property
    def graph(self) -> _DispatcherGraphAdapter:
        """Access the compiled LangGraph state graph, initializing if needed.

        Raises:
            ServiceStartupError: If graph compilation fails.
        """
        self._ensure_initialized()
        if self._graph is None:
            from contextunity.core.exceptions import ServiceStartupError

            raise ServiceStartupError("Dispatcher graph failed to initialize")
        return self._graph

    async def _deduplicate_messages(
        self,
        messages: Sequence[BaseMessage],
        config: RunnableConfig,
    ) -> Sequence[BaseMessage]:
        """Remove messages that already exist in the checkpoint.

        Strategy:
        1. Load existing messages from checkpoint.
        2. Extract content strings from both existing and incoming.
        3. Find the longest suffix of incoming that overlaps with existing.
        4. Return only the non-overlapping tail.

        If no checkpointer or no overlap → return original messages unchanged.
        """
        if self._graph is None or self._graph.checkpointer is None or not messages:
            return messages

        try:
            state_now = await self._graph.aget_state(config)
            existing_raw = state_now.values.get("messages", [])
            if not is_object_list(existing_raw) or not existing_raw:
                return messages
            existing_msgs = existing_raw
        except Exception as e:  # graceful-degrade: dedup falls back to full message list
            logger.warning("Dedup: failed to read state: %s", e)
            return messages

        existing_contents = [extract_message_content(message) for message in existing_msgs]
        incoming_contents = [extract_message_content(message) for message in messages]

        # Find the longest overlap: the tail of existing that matches a
        # prefix of incoming.  We want the earliest point in incoming
        # where the whole remaining existing tail has already been seen.
        #
        # Simpler approach: walk incoming backwards & find where the last
        # existing message appears. Everything after that is new.
        last_existing_content = existing_contents[-1]

        # Check from the END of incoming backwards to find the latest position
        # where last_existing_content appears (the client may have re-sent
        # partial history).
        cut_point = -1
        for i in range(len(incoming_contents) - 1, -1, -1):
            if incoming_contents[i] == last_existing_content:
                cut_point = i
                break

        if cut_point >= 0:
            new_msgs = messages[cut_point + 1 :]
            if not new_msgs:
                return messages
            if len(new_msgs) < len(messages):
                logger.debug(
                    "Dedup: stripped %d existing msgs, keeping %d new",
                    cut_point + 1,
                    len(new_msgs),
                )
            return new_msgs

        # No overlap found — client sent entirely new messages.
        # This is fine, pass them through.
        return messages

    def _build_state(
        self,
        messages: Sequence[BaseMessage],
        tenant_id: str,
        session_id: str,
        platform: str,
        metadata: ExecutionMetadata | None,
        max_iterations: int,
        allowed_tools: list[str] | None,
        denied_tools: list[str] | None,
        trace_id: str,
        access_token: ContextToken,
    ) -> DispatcherState:
        """Assemble the initial DispatcherState dict for a graph invocation.

        Merges message history, token-derived permissions, and execution
        constraints into a single state object.
        """
        return make_dispatcher_state(
            messages=messages,
            access_token=access_token,
            tenant_id=tenant_id,
            session_id=session_id,
            platform=platform,
            trace_id=trace_id,
            metadata=metadata,
            max_iterations=max_iterations,
            allowed_tools=self._resolve_allowed_tools(allowed_tools, access_token),
            denied_tools=denied_tools or [],
        )

    def _build_config(
        self,
        tenant_id: str,
        session_id: str,
        platform: str,
        metadata: ExecutionMetadata | None,
        access_token: ContextToken,
    ) -> RunnableConfig:
        """Build RunnableConfig with Langfuse + BrainAutoTracer callbacks.

        The checkpointer thread is isolated by tenant, trusted project,
        authenticated principal, and client session.
        """
        effective_user_id: str | None = getattr(access_token, "user_id", None)
        meta = metadata or ExecutionMetadata()
        project_id = meta.get("project_id", "")
        principal = (
            effective_user_id
            or getattr(access_token, "agent_id", None)
            or getattr(access_token, "token_id", None)
            or ""
        )
        checkpoint_scope = json_dumps(
            {
                "tenant_id": tenant_id,
                "project_id": project_id,
                "principal": principal,
                "session_id": session_id,
            },
            sort_keys=True,
        )
        thread_id = "dispatcher:" + hashlib.sha256(checkpoint_scope.encode()).hexdigest()

        langfuse_ctx = LangfuseRequestCtx.from_metadata(dict(meta))
        callbacks = list(
            get_langfuse_callbacks(
                session_id=session_id,
                user_id=effective_user_id,
                platform=platform,
                langfuse_ctx=langfuse_ctx,
            )
        )
        from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer

        callbacks.append(BrainAutoTracer())
        return RunnableConfig(
            configurable={"thread_id": thread_id},
            callbacks=callbacks,
        )

    async def invoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        access_token: ContextToken,
        tenant_id: str = "default",
        session_id: str = "default",
        platform: str = "api",
        metadata: ExecutionMetadata | None = None,
        max_iterations: int = 10,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        trace_id: str | None = None,
    ) -> DispatcherState:
        """Invoke the dispatcher agent.

        Args:
            messages: List of LangChain BaseMessage objects
            access_token: ContextToken for capability-based access control
            tenant_id: Tenant identifier for multi-tenant isolation
            session_id: Session identifier
            platform: Platform identifier (api, web, telegram, etc.)
            metadata: Additional metadata
            max_iterations: Maximum number of agent iterations
            allowed_tools: Explicit tool allowlist. ``None`` derives from token; empty denies all.
            denied_tools: List of denied tool names (blacklist)
            trace_id: Distributed trace ID (auto-generated if not provided)

        Returns:
            Final state dict from graph execution
        """
        import uuid

        self._ensure_initialized()

        from contextunity.router.service.mixins.execution.helpers import (
            resolve_dispatcher_tenant_id,
        )

        tenant_id = resolve_dispatcher_tenant_id(tenant_id, access_token)

        if trace_id is None:
            trace_id = uuid.uuid4().hex

        config = self._build_config(tenant_id, session_id, platform, metadata, access_token)

        # Deduplicate messages if checkpointer is active to avoid history explosion
        messages = await self._deduplicate_messages(messages, config)

        state = self._build_state(
            messages,
            tenant_id,
            session_id,
            platform,
            metadata,
            max_iterations,
            allowed_tools,
            denied_tools,
            trace_id,
            access_token,
        )

        effective_user_id: str | None = getattr(access_token, "user_id", None)
        meta = metadata or ExecutionMetadata()
        langfuse_ctx = LangfuseRequestCtx.from_metadata(dict(meta))

        with trace_context(
            session_id=session_id,
            platform=platform,
            name="dispatcher_invoke",
            user_id=effective_user_id,
            trace_id=trace_id,
            trace_input={"messages": messages},
            trace_metadata=dict(meta),
            tenant_id=tenant_id,
            agent_id=meta.get("agent_id"),
            graph_name=meta.get("graph_name"),
            langfuse_ctx=langfuse_ctx,
        ):
            result_raw = await self.graph.ainvoke(state, config=config)

        if is_dispatcher_state(result_raw):
            return result_raw
        raise ConfigurationError(
            message=(
                "Dispatcher graph returned invalid state "
                f"({type(result_raw).__name__}); refusing to discard graph output."
            )
        )

    async def stream(
        self,
        messages: Sequence[BaseMessage],
        *,
        access_token: ContextToken,
        tenant_id: str = "default",
        session_id: str = "default",
        platform: str = "api",
        metadata: ExecutionMetadata | None = None,
        max_iterations: int = 10,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[StateUpdate]:
        """Stream results from the dispatcher agent.

        Args:
            messages: List of messages (LangChain format)
            access_token: ContextToken for capability-based access control
            tenant_id: Tenant identifier for multi-tenant isolation
            session_id: Session identifier
            platform: Platform identifier
            metadata: Additional metadata
            max_iterations: Maximum number of agent iterations
            allowed_tools: Explicit tool allowlist. ``None`` derives from token; empty denies all.
            denied_tools: List of denied tool names (blacklist)

        Yields:
            Events from graph execution
        """
        self._ensure_initialized()

        import uuid

        from contextunity.router.service.mixins.execution.helpers import (
            resolve_dispatcher_tenant_id,
        )

        tenant_id = resolve_dispatcher_tenant_id(tenant_id, access_token)

        if trace_id is None:
            trace_id = uuid.uuid4().hex

        config = self._build_config(tenant_id, session_id, platform, metadata, access_token)

        # Deduplicate messages if checkpointer is active
        messages = await self._deduplicate_messages(messages, config)

        state = self._build_state(
            messages,
            tenant_id,
            session_id,
            platform,
            metadata,
            max_iterations,
            allowed_tools,
            denied_tools,
            trace_id,
            access_token,
        )

        effective_user_id: str | None = getattr(access_token, "user_id", None)
        meta = metadata or ExecutionMetadata()
        langfuse_ctx = LangfuseRequestCtx.from_metadata(dict(meta))

        with trace_context(
            session_id=session_id,
            platform=platform,
            name="dispatcher_stream",
            user_id=effective_user_id,
            trace_id=trace_id,
            trace_input={"messages": messages},
            trace_metadata=dict(meta),
            tenant_id=tenant_id,
            agent_id=meta.get("agent_id"),
            graph_name=meta.get("graph_name"),
            langfuse_ctx=langfuse_ctx,
        ):
            async for event in self.graph.astream(state, config=config):
                yield validated_dispatcher_stream_event(event)

    @staticmethod
    def _resolve_allowed_tools(
        explicit: list[str] | None,
        token: ContextToken,
    ) -> list[str]:
        """Resolve explicit allowlist and token-derived tool permissions.

        Priority: explicit list > token tool permissions > empty (no tools allowed).
        """
        if explicit is not None:
            return explicit
        if hasattr(token, "permissions") and token.permissions:
            from contextunity.core.permissions import extract_tool_names

            names = extract_tool_names(token.permissions)
            return sorted(names)
        return []

    def reset(self) -> None:
        """Reset the compiled graph, forcing recompilation on next use.

        Useful for testing or after configuration changes.
        """
        with self._lock:
            self._graph = None
            self._initialized = False
            logger.info("Dispatcher agent graph reset")


def get_dispatcher_service() -> DispatcherService:
    """Get or create the singleton dispatcher service.

    Returns:
        Singleton DispatcherService instance
    """
    global _dispatcher_instance

    if _dispatcher_instance is not None:
        return _dispatcher_instance

    with _dispatcher_lock:
        if _dispatcher_instance is None:
            _dispatcher_instance = DispatcherService()
            logger.info("Dispatcher service singleton created")
        return _dispatcher_instance


def reset_dispatcher_service() -> None:
    """Reset and destroy the singleton dispatcher service.

    Resets the internal graph state and clears the module-level singleton.
    Primarily used in tests.
    """
    global _dispatcher_instance
    with _dispatcher_lock:
        if _dispatcher_instance is not None:
            _dispatcher_instance.reset()
        _dispatcher_instance = None


__all__ = [
    "DispatcherService",
    "get_dispatcher_service",
    "reset_dispatcher_service",
]
