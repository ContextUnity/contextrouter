"""In-memory cache for dispatcher agent settings, sessions, and query results.

Provides fast in-memory caching with Redis as backing store for persistence.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from contextrouter.core import get_core_config
from contextrouter.modules.providers.redis import RedisProvider

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL."""

    def __init__(self, value: Any, ttl_seconds: int | None = None):
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds


class InMemoryCache:
    """In-memory cache with optional Redis backing."""

    def __init__(self, redis: RedisProvider | None = None):
        self._cache: dict[str, CacheEntry] = {}
        self._redis = redis
        self._max_size = 1000  # Maximum entries in memory

    async def get(self, key: str) -> Any | None:
        """Get value from cache (memory first, then Redis)."""
        # Check memory cache
        entry = self._cache.get(key)
        if entry:
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.value

        # Check Redis if available
        if self._redis:
            try:
                import json

                data_str = await self._redis.get(key)
                if data_str:
                    data = json.loads(data_str)
                    # Load into memory cache
                    self._cache[key] = CacheEntry(
                        data["value"], ttl_seconds=data.get("ttl_seconds")
                    )
                    return data["value"]
            except Exception as e:
                logger.debug("Redis get failed for %s: %s", key, e)

        return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set value in cache (memory and Redis)."""
        # Store in memory
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(value, ttl_seconds)

        # Store in Redis if available
        if self._redis:
            try:
                import json

                data = {"value": value, "ttl_seconds": ttl_seconds}
                await self._redis.set(key, json.dumps(data), ex=ttl_seconds)
            except Exception as e:
                logger.debug("Redis set failed for %s: %s", key, e)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        self._cache.pop(key, None)
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception as e:
                logger.debug("Redis delete failed for %s: %s", key, e)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]


# Global cache instances
_settings_cache: InMemoryCache | None = None
_session_cache: InMemoryCache | None = None
_query_cache: InMemoryCache | None = None


def get_settings_cache() -> InMemoryCache:
    """Get settings cache singleton."""
    global _settings_cache
    if _settings_cache is None:
        config = get_core_config()
        redis = None
        if config.redis.host:
            redis = RedisProvider(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
            )
        _settings_cache = InMemoryCache(redis=redis)
    return _settings_cache


def get_session_cache() -> InMemoryCache:
    """Get session cache singleton."""
    global _session_cache
    if _session_cache is None:
        config = get_core_config()
        redis = None
        if config.redis.host:
            redis = RedisProvider(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
            )
        _session_cache = InMemoryCache(redis=redis)
    return _session_cache


def get_query_cache() -> InMemoryCache:
    """Get query cache singleton."""
    global _query_cache
    if _query_cache is None:
        config = get_core_config()
        redis = None
        if config.redis.host:
            redis = RedisProvider(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                password=config.redis.password,
            )
        _query_cache = InMemoryCache(redis=redis)
    return _query_cache


def make_query_cache_key(query: str, tenant_id: str | None = None) -> str:
    """Create cache key for query."""
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"query:{tenant_prefix}{query_hash}"


def make_session_cache_key(session_id: str, tenant_id: str | None = None) -> str:
    """Create cache key for session."""
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"session:{tenant_prefix}{session_id}"


def make_settings_cache_key(setting_name: str, tenant_id: str | None = None) -> str:
    """Create cache key for settings."""
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"settings:{tenant_prefix}{setting_name}"


__all__ = [
    "InMemoryCache",
    "get_settings_cache",
    "get_session_cache",
    "get_query_cache",
    "make_query_cache_key",
    "make_session_cache_key",
    "make_settings_cache_key",
]
