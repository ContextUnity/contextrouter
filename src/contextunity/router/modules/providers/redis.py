"""Redis provider — shared async Redis pool for rate limiting, checkpointing, and revocation."""

from __future__ import annotations

from typing import Protocol

import redis.asyncio as redis
from contextunity.core import get_contextunit_logger

from contextunity.router.core.exceptions import RouterStorageError

logger = get_contextunit_logger(__name__)


class RedisRateLimitPipeline(Protocol):
    """Narrow redis-py pipeline surface used by ``RateLimiter``."""

    def zremrangebyscore(self, name: str, min: float, max: float) -> object: ...

    def zcard(self, name: str) -> object: ...

    def zadd(self, name: str, mapping: dict[str, float]) -> object: ...

    def expire(self, name: str, time: int) -> object: ...

    async def execute(self) -> list[object]: ...


class _RedisModule(Protocol):
    def from_url(self, url: str, **kwargs: object) -> redis.Redis: ...


async def _await_redis_ping(client: object) -> None:
    ping_attr: object = getattr(client, "ping", None)
    if not callable(ping_attr):
        return
    ping_call: object = ping_attr()
    from contextunity.core.narrowing import await_object

    await await_object(ping_call)


def _redis_from_url(url: str) -> redis.Redis:
    redis_module: _RedisModule = redis
    return redis_module.from_url(url, decode_responses=True)


class RedisProvider:
    """Redis provider for AI Gateway shared state.
    Used for token revocation, rate limiting, and LangGraph checkpointers.

    Respects the platform-level ``redis.enabled`` flag from ``SharedConfig``.
    When Redis is disabled, all operations are graceful no-ops.
    """

    _url: str | None
    host: str
    port: int
    db: int
    password: str | None
    _client: redis.Redis | None

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        *,
        url: str | None = None,
    ):
        """Store connection parameters; lazily connect on first operation."""
        self._url = url
        self.host = host if url is None else ""
        self.port = port if url is None else 0
        self.db = db if url is None else 0
        self.password = password if url is None else None
        self._client = None

    @classmethod
    def from_url(cls, url: str) -> RedisProvider:
        """Construct from a ``redis://`` URL (password and db encoded in the URL)."""
        return cls(url=url)

    async def connect(self):
        """Establish connection to the Redis server."""
        if not self.host and not self._url:
            return
        if not self._client:
            if self._url:
                self._client = _redis_from_url(self._url)
            else:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                )
            try:
                await _await_redis_ping(self._client)
                if self._url:
                    logger.info("Connected to Redis via URL")
                else:
                    logger.info("Connected to Redis at %s:%s", self.host, self.port)
            except Exception as e:
                logger.error("Failed to connect to Redis: %s", e)
                self._client = None

    def pipeline(self) -> RedisRateLimitPipeline:
        """Return a Redis pipeline for batching commands.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._client:
            raise RouterStorageError("Redis client not connected")
        pipe_obj: object = self._client.pipeline()
        return pipe_obj

    async def get(self, key: str) -> str | None:
        """Fetch a value by key, auto-connecting if needed. Returns ``None`` when unavailable."""
        if not self._client:
            await self.connect()
        return await self._client.get(key) if self._client else None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        """Set a value for a key in Redis with optional expiration.

        Args:
            key (str): The database or cache lookup key.
            value (str): The value to store or update.
            ex (Optional[int]): The optional expiration time in seconds.
        """
        if not self._client:
            await self.connect()
        if self._client:
            await self._client.set(key, value, ex=ex)

    async def delete(self, key: str):
        """Delete a key from Redis.

        Args:
            key (str): The database or cache lookup key.
        """
        if not self._client:
            await self.connect()
        if self._client:
            await self._client.delete(key)

    async def close(self):
        """Close the Redis client connection pool."""
        if self._client:
            await self._client.close()
            self._client = None


__all__ = ["RedisProvider", "RedisRateLimitPipeline"]
