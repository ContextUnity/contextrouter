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
import logging
from typing import Any, AsyncIterator, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from contextrouter.core import get_core_config
from contextrouter.modules.providers.redis import RedisProvider

logger = logging.getLogger(__name__)


class RedisCheckpointSaver(BaseCheckpointSaver):
    """Redis-based checkpoint saver for LangGraph state persistence.

    Stores checkpoint state in Redis with configurable TTL.
    Supports multi-instance deployments with shared state.
    """

    def __init__(
        self,
        redis: RedisProvider | None = None,
        key_prefix: str = "checkpoint:dispatcher:",
        default_ttl: int = 86400,  # 24 hours
    ):
        super().__init__()
        if redis is None:
            config = get_core_config()
            redis = RedisProvider(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
            )

        self.redis = redis
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl

    # ── key helpers ──────────────────────────────────────────

    def _make_key(self, thread_id: str, checkpoint_ns: str = "") -> str:
        """Create Redis key for checkpoint."""
        ns = checkpoint_ns or "default"
        return f"{self.key_prefix}{thread_id}:{ns}"

    def _writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        ns = checkpoint_ns or "default"
        return f"{self.key_prefix}{thread_id}:{ns}:writes:{checkpoint_id}"

    @staticmethod
    def _thread_id(config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("thread_id", "")

    @staticmethod
    def _checkpoint_ns(config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("checkpoint_ns", "")

    @staticmethod
    def _checkpoint_id(config: RunnableConfig) -> str | None:
        return config.get("configurable", {}).get("checkpoint_id")

    # ── serde helpers (proper message serialization) ─────────

    def _serialize(self, obj: Any) -> bytes:
        """Serialize using LangGraph's JsonPlusSerializer.

        Returns bytes (msgpack) that can be stored in Redis.
        """
        type_tag, data = self.serde.dumps_typed(obj)
        # Store as a JSON envelope with type info for correct round-tripping
        envelope = json.dumps({"type": type_tag, "data": data.decode("latin-1")})
        return envelope.encode()

    def _deserialize(self, raw: bytes | str) -> Any:
        """Deserialize data stored by _serialize.

        Also handles legacy JSON-encoded data (plain json.dumps with default=str)
        for backward compatibility during migration.
        """
        if isinstance(raw, str):
            raw = raw.encode()

        # Try new envelope format first
        try:
            envelope = json.loads(raw)
            if isinstance(envelope, dict) and "type" in envelope and "data" in envelope:
                data_bytes = envelope["data"].encode("latin-1")
                return self.serde.loads_typed((envelope["type"], data_bytes))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: legacy plain JSON (migration period)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        logger.warning("Could not deserialize checkpoint data, returning None")
        return None

    # ── required async API ──────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from Redis."""
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
            if not data or not isinstance(data, dict):
                return None

            checkpoint = data["checkpoint"]
            metadata = data.get("metadata", {})
            parent_config = data.get("parent_config")

            # Reconstruct the config that produced this checkpoint
            stored_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint.get("id", ""),
                }
            }

            # Load pending writes
            pending_writes = None
            checkpoint_id = checkpoint.get("id", "")
            if checkpoint_id:
                writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)
                writes_raw = await self.redis.get(writes_key)
                if writes_raw:
                    pending_writes = self._deserialize(writes_raw)

            return CheckpointTuple(
                config=stored_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )
        except Exception as e:
            logger.error("Failed to get checkpoint for %s: %s", thread_id, e)
            return None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint in Redis."""
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
            data_bytes = self._serialize(checkpoint_data)
            await self.redis.set(key, data_bytes, ex=self.default_ttl)
            logger.debug("Saved checkpoint for thread_id: %s", thread_id)
        except Exception as e:
            logger.error("Failed to save checkpoint for %s: %s", thread_id, e)

        # Return updated config pointing to the new checkpoint
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint.get("id", ""),
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        thread_id = self._thread_id(config)
        checkpoint_id = self._checkpoint_id(config) or ""
        checkpoint_ns = self._checkpoint_ns(config)

        if not thread_id or not checkpoint_id:
            return

        key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        try:
            # Append writes (load existing + extend)
            existing_raw = await self.redis.get(key)
            existing: list = []
            if existing_raw:
                deserialized = self._deserialize(existing_raw)
                if isinstance(deserialized, list):
                    existing = deserialized

            for channel, value in writes:
                existing.append((task_id, channel, value))

            data_bytes = self._serialize(existing)
            await self.redis.set(key, data_bytes, ex=self.default_ttl)
        except Exception as e:
            logger.error("Failed to save writes for %s: %s", thread_id, e)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints. Minimal implementation — returns current only."""
        if config is None:
            return

        result = await self.aget_tuple(config)
        if result is not None:
            yield result


__all__ = ["RedisCheckpointSaver"]
