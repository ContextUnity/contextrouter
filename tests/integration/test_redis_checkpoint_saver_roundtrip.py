"""Integration: RedisCheckpointSaver round-trip with checkpoint guards."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from contextunity.router.cortex.services.redis_saver import RedisCheckpointSaver
from contextunity.router.modules.providers.redis import RedisProvider


class _InMemoryRedisClient:
    """Minimal redis.asyncio-shaped client for saver tests."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        _ = ex
        self._store[key] = value

    async def close(self) -> None:
        return None


def _valid_checkpoint() -> Checkpoint:
    return {
        "v": 1,
        "id": "ckpt-integration-1",
        "ts": "2026-06-04T12:00:00Z",
        "channel_values": {"messages": []},
        "channel_versions": {"messages": 1},
        "versions_seen": {"__start__": {"messages": 0}},
        "updated_channels": ["messages"],
    }


@pytest.mark.asyncio
async def test_redis_checkpoint_saver_aput_aget_round_trip() -> None:
    provider = RedisProvider(host="")
    provider._client = _InMemoryRedisClient()
    saver = RedisCheckpointSaver(redis=provider, key_prefix="test:ckpt:")

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-42",
            "checkpoint_ns": "dispatcher",
        }
    }
    checkpoint_obj: Checkpoint = _valid_checkpoint()  # pyright: narrow at guard boundary in saver
    metadata_obj: CheckpointMetadata = {"source": "loop", "step": 1}

    updated = await saver.aput(
        config,
        checkpoint_obj,
        metadata_obj,
        {"messages": 1},
    )

    loaded = await saver.aget_tuple(updated)
    assert loaded is not None
    assert loaded.checkpoint["id"] == "ckpt-integration-1"
    assert loaded.metadata.get("source") == "loop"


@pytest.mark.asyncio
async def test_redis_checkpoint_saver_rejects_invalid_stored_shape() -> None:
    provider = RedisProvider(host="")
    client = _InMemoryRedisClient()
    provider._client = client
    saver = RedisCheckpointSaver(redis=provider, key_prefix="test:ckpt:")

    key = saver._make_key("thread-bad", "default")
    client._store[key] = '{"checkpoint": {"v": 1}, "metadata": {}}'

    config: RunnableConfig = {"configurable": {"thread_id": "thread-bad"}}
    assert await saver.aget_tuple(config) is None
