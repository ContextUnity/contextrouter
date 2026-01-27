import logging
from typing import Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisProvider:
    """
    Redis provider for AI Gateway shared state.
    Used for token revocation, rate limiting, and LangGraph checkpointers.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._client: Optional[redis.Redis] = None

    async def connect(self):
        if not self._client:
            self._client = redis.Redis(
                host=self.host, port=self.port, db=self.db, decode_responses=True
            )
            try:
                await self._client.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._client = None

    async def get(self, key: str) -> Optional[str]:
        if not self._client:
            await self.connect()
        return await self._client.get(key) if self._client else None

    async def set(self, key: str, value: str, ex: Optional[int] = None):
        if not self._client:
            await self.connect()
        if self._client:
            await self._client.set(key, value, ex=ex)

    async def delete(self, key: str):
        if not self._client:
            await self.connect()
        if self._client:
            await self._client.delete(key)

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
