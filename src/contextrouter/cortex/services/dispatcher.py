"""Background service for always-active dispatcher agent."""

from __future__ import annotations

import logging
import threading
from typing import Any

from contextcore import ContextToken

from contextrouter.core import get_core_config
from contextrouter.cortex.graphs.dispatcher_agent import compile_dispatcher_graph
from contextrouter.modules.observability import get_langfuse_callbacks, trace_context

logger = logging.getLogger(__name__)

# Try to import checkpointing (optional — requires Redis)
try:
    from contextrouter.cortex.checkpointing.redis_saver import RedisCheckpointSaver

    CHECKPOINTING_AVAILABLE = True
except ImportError:
    CHECKPOINTING_AVAILABLE = False
    logger.debug("Checkpointing not available")


# Global dispatcher instance
_dispatcher_instance: "DispatcherService | None" = None
_dispatcher_lock = threading.Lock()


class DispatcherService:
    """Always-active dispatcher agent service.

    This service maintains a persistent graph instance and provides
    access to the dispatcher agent via API or Python import.
    """

    def __init__(self) -> None:
        """Initialize the dispatcher service."""
        self._graph = None
        self._initialized = False
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Ensure the graph is compiled and ready."""
        if self._initialized and self._graph is not None:
            return

        with self._lock:
            if self._initialized and self._graph is not None:
                return

            logger.info("Initializing dispatcher agent graph...")

            # Enable checkpointing for session persistence
            checkpointer = None
            if CHECKPOINTING_AVAILABLE:
                try:
                    config = get_core_config()
                    if config.redis.host:
                        checkpointer = RedisCheckpointSaver()
                        logger.info("Checkpointing enabled with Redis")
                except Exception as e:
                    logger.warning("Failed to initialize checkpointing: %s", e)

            self._graph = compile_dispatcher_graph(checkpointer=checkpointer)
            self._initialized = True
            logger.info("Dispatcher agent graph initialized")

    @property
    def graph(self) -> Any:
        """Get the compiled graph instance."""
        self._ensure_initialized()
        return self._graph

    async def _deduplicate_messages(
        self,
        messages: list[dict[str, Any]],
        config: dict,
    ) -> list[dict[str, Any]]:
        """Remove messages that already exist in the checkpoint.

        Strategy:
        1. Load existing messages from checkpoint.
        2. Extract content strings from both existing and incoming.
        3. Find the longest suffix of incoming that overlaps with existing.
        4. Return only the non-overlapping tail.

        If no checkpointer or no overlap → return original messages unchanged.
        """
        if not self._graph.checkpointer or not messages:
            return messages

        try:
            state_now = await self._graph.aget_state(config)
            existing_msgs = state_now.values.get("messages", [])
            if not existing_msgs:
                return messages
        except Exception as e:
            logger.warning("Dedup: failed to read state: %s", e)
            return messages

        def _content(m) -> str:
            """Extract comparable content string from any message format."""
            if hasattr(m, "content"):
                c = m.content
                # Guard against nested repr strings (e.g. content="content='...'")
                if isinstance(c, str):
                    return c
                return str(c)
            if isinstance(m, dict):
                return m.get("content", "")
            return str(m)

        existing_contents = [_content(m) for m in existing_msgs]
        incoming_contents = [_content(m) for m in messages]

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

    async def invoke(
        self,
        messages: list[dict[str, Any]],
        tenant_id: str = "default",
        session_id: str = "default",
        platform: str = "api",
        metadata: dict[str, Any] | None = None,
        max_iterations: int = 10,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        trace_id: str | None = None,
        access_token: ContextToken | None = None,
    ) -> dict[str, Any]:
        """Invoke the dispatcher agent.

        Args:
            messages: List of messages (LangChain format)
            tenant_id: Tenant identifier for multi-tenant isolation
            session_id: Session identifier
            platform: Platform identifier (api, web, telegram, etc.)
            metadata: Additional metadata
            max_iterations: Maximum number of agent iterations
            allowed_tools: List of allowed tool names (empty = all allowed)
            denied_tools: List of denied tool names (blacklist)
            trace_id: Distributed trace ID (auto-generated if not provided)

        Returns:
            Final state from graph execution
        """
        import uuid

        self._ensure_initialized()

        # Generate trace_id if not provided
        if trace_id is None:
            trace_id = uuid.uuid4().hex

        state = {
            "messages": messages,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "platform": platform,
            "metadata": metadata or {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "allowed_tools": self._resolve_allowed_tools(allowed_tools, access_token),
            "denied_tools": denied_tools or [],
            "trace_id": trace_id,
            "access_token": access_token,
        }

        # Prefix thread_id with tenant for checkpointing isolation
        config = {"configurable": {"thread_id": f"{tenant_id}:{session_id}"}}

        # Deduplicate messages if checkpointer is active to avoid history explosion
        messages = await self._deduplicate_messages(messages, config)

        state = {
            "messages": messages,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "platform": platform,
            "metadata": metadata or {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "allowed_tools": self._resolve_allowed_tools(allowed_tools, access_token),
            "denied_tools": denied_tools or [],
            "trace_id": trace_id,
            "access_token": access_token,
        }

        # Add Langfuse callbacks
        effective_user_id = getattr(access_token, "user_id", None) if access_token else None
        callbacks = get_langfuse_callbacks(
            session_id=session_id,
            user_id=effective_user_id,
            platform=platform,
        )
        config["callbacks"] = callbacks

        meta = metadata or {}
        with trace_context(
            session_id=session_id,
            platform=platform,
            name="dispatcher_invoke",
            user_id=effective_user_id,
            trace_id=trace_id,
            trace_input={"messages": messages},
            trace_metadata=meta,
            tenant_id=tenant_id,
            agent_id=meta.get("agent_id"),
            graph_name=meta.get("graph_name"),
        ):
            result = await self._graph.ainvoke(state, config=config)
        return result

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tenant_id: str = "default",
        session_id: str = "default",
        platform: str = "api",
        metadata: dict[str, Any] | None = None,
        max_iterations: int = 10,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        trace_id: str | None = None,
        access_token: ContextToken | None = None,
    ) -> Any:
        """Stream results from the dispatcher agent.

        Args:
            messages: List of messages (LangChain format)
            tenant_id: Tenant identifier for multi-tenant isolation
            session_id: Session identifier
            platform: Platform identifier
            metadata: Additional metadata
            max_iterations: Maximum number of agent iterations
            allowed_tools: List of allowed tool names (empty = all allowed)
            denied_tools: List of denied tool names (blacklist)

        Yields:
            Events from graph execution
        """
        self._ensure_initialized()

        import uuid

        # Generate trace_id if not provided
        if trace_id is None:
            trace_id = uuid.uuid4().hex

        # Prefix thread_id with tenant for checkpointing isolation
        config = {"configurable": {"thread_id": f"{tenant_id}:{session_id}"}}

        # Deduplicate messages if checkpointer is active
        messages = await self._deduplicate_messages(messages, config)

        state = {
            "messages": messages,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "platform": platform,
            "metadata": metadata or {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "allowed_tools": self._resolve_allowed_tools(allowed_tools, access_token),
            "denied_tools": denied_tools or [],
            "trace_id": trace_id,
            "access_token": access_token,
        }

        # Add Langfuse callbacks
        effective_user_id = getattr(access_token, "user_id", None) if access_token else None
        callbacks = get_langfuse_callbacks(
            session_id=session_id,
            user_id=effective_user_id,
            platform=platform,
        )
        config["callbacks"] = callbacks

        meta = metadata or {}
        with trace_context(
            session_id=session_id,
            platform=platform,
            name="dispatcher_stream",
            user_id=effective_user_id,
            trace_id=trace_id,
            trace_input={"messages": messages},
            trace_metadata=meta,
            tenant_id=tenant_id,
            agent_id=meta.get("agent_id"),
            graph_name=meta.get("graph_name"),
        ):
            async for event in self._graph.astream(state, config=config):
                yield event

    @staticmethod
    def _resolve_allowed_tools(
        explicit: list[str] | None,
        token: ContextToken | None,
    ) -> list[str]:
        """Merge explicit allowed_tools with token permissions.

        Priority: explicit list > token permissions > empty (all allowed).
        """
        if explicit:
            return explicit
        if token is not None and hasattr(token, "permissions") and token.permissions:
            try:
                from contextcore.permissions import extract_tool_names

                names = extract_tool_names(token.permissions)
                return sorted(names)  # deterministic ordering
            except ImportError:
                pass
        return []

    def reset(self) -> None:
        """Reset the graph (useful for testing or config reload)."""
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
        if _dispatcher_instance is not None:
            return _dispatcher_instance

        _dispatcher_instance = DispatcherService()
        logger.info("Dispatcher service singleton created")

    return _dispatcher_instance


def reset_dispatcher_service() -> None:
    """Reset the singleton (mainly for testing)."""
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
