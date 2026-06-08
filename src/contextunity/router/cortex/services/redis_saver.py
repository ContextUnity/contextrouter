"""Redis-based checkpoint saver for LangGraph.
Provides persistent state storage for dispatcher agent using Redis.
Implements the current LangGraph checkpoint API (aget_tuple/aput/alist/aput_writes).
IMPORTANT: Uses LangGraph's JsonPlusSerializer (via self.serde) for message
round-tripping. Plain json.dumps(default=str) MUST NOT be used — it destroys
BaseMessage objects by calling str() on them, which produces repr strings like
``content='...' additional_kwargs={}``. On the next read-back, this repr
becomes the new `.content`, causing recursive nesting.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, TypedDict, TypeGuard, final, override, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps, json_loads
from contextunity.core.types import is_object_dict
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)

from contextunity.router.core import get_core_config
from contextunity.router.cortex.checkpoint_guards import (
    _is_checkpoint,
    _is_checkpoint_metadata,
    _is_pending_writes,
)
from contextunity.router.modules.providers.redis import RedisProvider

logger = get_contextunit_logger(__name__)


class _SerdeEnvelope(TypedDict):
    type: str
    data: str


def _is_serde_envelope(value: object) -> TypeGuard[_SerdeEnvelope]:
    """Check whether a decoded JSON object matches the typed serde envelope."""
    return (
        is_object_dict(value)
        and isinstance(value.get("type"), str)
        and isinstance(value.get("data"), str)
    )


def _is_runnable_config(value: object) -> TypeGuard[RunnableConfig]:
    """Check whether an object looks like a RunnableConfig mapping."""
    return is_object_dict(value)


def _checkpoint_id_value(checkpoint: Checkpoint) -> str:
    """Extract a checkpoint id safely from a checkpoint dict."""
    checkpoint_id = checkpoint.get("id")
    return checkpoint_id or ""


@runtime_checkable
class _TypedSerializer(Protocol):
    """Typed subset of LangGraph's serializer contract used here."""

    def dumps_typed(self, obj: object) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> object: ...


@final
class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    """Redis-based checkpoint saver for LangGraph state persistence.

    Stores checkpoint state in Redis with configurable TTL.
    Supports multi-instance deployments with shared state.
    """

    redis: RedisProvider
    key_prefix: str
    default_ttl: int

    def __init__(
        self,
        redis: RedisProvider | None = None,
        key_prefix: str = "checkpoint:dispatcher:",
        default_ttl: int = 86400,  # 24 hours
    ) -> None:
        """Initialize the Redis checkpoint saver.

        Args:
            redis: Optional pre-configured RedisProvider. If None, one is created
                from the platform config.
            key_prefix: Prefix prepended to all Redis keys for namespace isolation.
            default_ttl: Default expiration in seconds for stored checkpoints (24h default).
        """
        super().__init__()
        if redis is None:
            config = get_core_config()
            if config.redis.enabled and config.redis.url:
                redis = RedisProvider.from_url(config.redis.url)
            else:
                redis = RedisProvider(host="")  # no-op provider

        self.redis = redis
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

    def _serializer(self) -> _TypedSerializer:
        """Return a typed serializer view over ``self.serde``."""
        serializer_obj: object = self.serde
        if not isinstance(serializer_obj, _TypedSerializer):
            raise TypeError("Checkpoint serializer does not support typed serde")
        return serializer_obj

    @staticmethod
    def _configurable(config: RunnableConfig) -> dict[str, object]:
        """Read the ``configurable`` dict from a RunnableConfig safely."""
        configurable = config.get("configurable")
        if is_object_dict(configurable):
            return configurable
        return {}

    # ── key helpers ──────────────────────────────────────────

    def _make_key(self, thread_id: str, checkpoint_ns: str = "") -> str:
        """Build the Redis key for a thread's latest checkpoint.

        Args:
            thread_id: The LangGraph thread identifier.
            checkpoint_ns: Optional namespace for sub-graph isolation.

        Returns:
            A prefixed Redis key string.
        """
        ns = checkpoint_ns or "default"
        return f"{self.key_prefix}{thread_id}:{ns}"

    def _writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Build the Redis key for pending writes linked to a specific checkpoint.

        Args:
            thread_id: The LangGraph thread identifier.
            checkpoint_ns: Namespace for sub-graph isolation.
            checkpoint_id: The specific checkpoint version identifier.

        Returns:
            A prefixed Redis key string for the writes hash.
        """
        ns = checkpoint_ns or "default"
        return f"{self.key_prefix}{thread_id}:{ns}:writes:{checkpoint_id}"

    @staticmethod
    def _thread_id(config: RunnableConfig) -> str:
        """Extract the thread ID from a RunnableConfig's configurable dict.

        Args:
            config: The LangGraph runnable configuration.

        Returns:
            The thread identifier, or an empty string if not set.
        """
        thread_id = RedisCheckpointSaver._configurable(config).get("thread_id")
        return thread_id if isinstance(thread_id, str) else ""

    @staticmethod
    def _checkpoint_ns(config: RunnableConfig) -> str:
        """Extract the checkpoint namespace from a RunnableConfig.

        Args:
            config: The LangGraph runnable configuration.

        Returns:
            The checkpoint namespace, or an empty string if not set.
        """
        checkpoint_ns = RedisCheckpointSaver._configurable(config).get("checkpoint_ns")
        return checkpoint_ns if isinstance(checkpoint_ns, str) else ""

    @staticmethod
    def _checkpoint_id(config: RunnableConfig) -> str | None:
        """Extract the checkpoint ID from a RunnableConfig, if present.

        Args:
            config: The LangGraph runnable configuration.

        Returns:
            The checkpoint identifier, or None if not set.
        """
        checkpoint_id = RedisCheckpointSaver._configurable(config).get("checkpoint_id")
        return checkpoint_id if isinstance(checkpoint_id, str) else None

    # ── serde helpers (proper message serialization) ─────────

    def _serialize(self, obj: object) -> str:
        """Serialize an object using LangGraph's JsonPlusSerializer for lossless round-tripping.

        Args:
            obj: The Python object to serialize (dicts, BaseMessage, etc.).

        Returns:
            A JSON string envelope containing the type tag and encoded data.
        """
        type_tag, data = self._serializer().dumps_typed(obj)
        # Store as a JSON envelope with type info for correct round-tripping
        envelope = json_dumps({"type": type_tag, "data": data.decode("latin-1")})
        return envelope

    def _deserialize(self, raw: bytes | str) -> object:
        """Deserialize data stored by ``_serialize``, with legacy fallback.

        Tries the typed envelope format first, then falls back to plain JSON
        for backward compatibility during migration.

        Args:
            raw: The raw bytes or string from Redis.

        Returns:
            The deserialized Python object, or None if decoding fails.
        """
        if isinstance(raw, str):
            raw = raw.encode()

        # Try new envelope format first
        try:
            envelope = json_loads(raw)
            if _is_serde_envelope(envelope):
                data_bytes = envelope["data"].encode("latin-1")
                return self._serializer().loads_typed((envelope["type"], data_bytes))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: legacy plain JSON (migration period)
        try:
            return json_loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        logger.warning("Could not deserialize checkpoint data, returning None")
        return None

    # ── required async API ──────────────────────────────────

    @override
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch the latest checkpoint tuple for a thread from Redis.

        Args:
            config: RunnableConfig containing the thread_id and optional checkpoint_ns.

        Returns:
            A CheckpointTuple with state, metadata, and pending writes, or None if
            no checkpoint exists.
        """
        thread_id = self._thread_id(config)
        if not thread_id:
            return None

        checkpoint_ns = self._checkpoint_ns(config)
        key = self._make_key(thread_id, checkpoint_ns)

        try:
            data_raw = await self.redis.get(key)
            if not data_raw:
                return None

            data = self._deserialize(data_raw)
            if not data or not is_object_dict(data):
                return None

            checkpoint = data.get("checkpoint")
            metadata = data.get("metadata", {})
            parent_config = data.get("parent_config")
            if not _is_checkpoint(checkpoint) or not _is_checkpoint_metadata(metadata):
                return None
            parent_config_value = parent_config if _is_runnable_config(parent_config) else None

            # Reconstruct the config that produced this checkpoint
            stored_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": _checkpoint_id_value(checkpoint),
                }
            }

            # Load pending writes
            pending_writes_value: list[PendingWrite] | None = None
            checkpoint_id = _checkpoint_id_value(checkpoint)
            if checkpoint_id:
                writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)
                writes_raw = await self.redis.get(writes_key)
                if writes_raw:
                    pending_writes_raw = self._deserialize(writes_raw)
                    if _is_pending_writes(pending_writes_raw):
                        pending_writes_value = pending_writes_raw

            return CheckpointTuple(
                config=stored_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config_value,
                pending_writes=pending_writes_value,
            )
        except Exception as e:  # graceful-degrade: corrupt/missing Redis checkpoint → cold start
            logger.error("Failed to get checkpoint for %s: %s", thread_id, e)
            return None

    @override
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint to Redis with the configured TTL.

        Args:
            config: RunnableConfig identifying the target thread.
            checkpoint: The LangGraph checkpoint state to store.
            metadata: Additional metadata associated with this checkpoint.
            new_versions: Channel version map for incremental state tracking.

        Returns:
            An updated RunnableConfig pointing to the newly stored checkpoint.
        """
        thread_id = self._thread_id(config)
        if not thread_id:
            logger.warning("No thread_id in config, skipping checkpoint save")
            return config

        checkpoint_ns = self._checkpoint_ns(config)
        key = self._make_key(thread_id, checkpoint_ns)

        checkpoint_data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_config": {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": self._checkpoint_id(config),
                }
            },
        }

        try:
            data_str = self._serialize(checkpoint_data)
            await self.redis.set(key, data_str, ex=self.default_ttl)
            logger.debug("Saved checkpoint for thread_id: %s", thread_id)
        except Exception as e:  # graceful-degrade: checkpoint save failure must not crash graph
            logger.error("Failed to save checkpoint for %s: %s", thread_id, e)

        # Return updated config pointing to the new checkpoint
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": _checkpoint_id_value(checkpoint),
            }
        }

    @override
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, object]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Append intermediate channel writes to a checkpoint's pending writes in Redis.

        Args:
            config: RunnableConfig identifying the thread and checkpoint.
            writes: Sequence of (channel_name, value) tuples to persist.
            task_id: The task identifier that produced these writes.
            task_path: Optional task path for sub-graph context.
        """
        thread_id = self._thread_id(config)
        checkpoint_id = self._checkpoint_id(config) or ""
        checkpoint_ns = self._checkpoint_ns(config)

        if not thread_id or not checkpoint_id:
            return

        key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        try:
            # Append writes (load existing + extend)
            existing_raw = await self.redis.get(key)
            existing: list[PendingWrite] = []
            if existing_raw:
                deserialized = self._deserialize(existing_raw)
                if _is_pending_writes(deserialized):
                    existing = list(deserialized)

            for channel, value in writes:
                existing.append((task_id, channel, value))
            if not existing:
                return

            data_str = self._serialize(existing)
            await self.redis.set(key, data_str, ex=self.default_ttl)
        except Exception as e:  # graceful-degrade: pending writes are best-effort
            logger.error("Failed to save writes for %s: %s", thread_id, e)

    @override
    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, object] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List available checkpoints. Returns only the current checkpoint (minimal implementation).

        Args:
            config: RunnableConfig identifying the thread, or None to skip.
            filter: Unused. Reserved for future metadata-based filtering.
            before: Unused. Reserved for future pagination.
            limit: Unused. Reserved for future result limiting.

        Yields:
            The current CheckpointTuple if one exists.
        """
        if config is None:
            return

        result = await self.aget_tuple(config)
        if result is not None:
            yield result


__all__ = ["RedisCheckpointSaver"]
