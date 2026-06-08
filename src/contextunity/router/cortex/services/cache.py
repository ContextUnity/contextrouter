"""In-memory cache for dispatcher agent settings, sessions, and query results.
Provides fast in-memory caching with Redis as backing store for persistence.
"""

from __future__ import annotations

import hashlib
import time
from typing import TypedDict, final

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps, json_loads
from contextunity.core.types import JsonValue, is_json_dict, is_json_value

from contextunity.router.core import get_core_config
from contextunity.router.modules.providers.redis import RedisProvider

logger = get_contextunit_logger(__name__)


class _CacheEnvelope(TypedDict):
    value: JsonValue
    ttl_seconds: int | None


def _parse_cache_envelope(raw: str) -> _CacheEnvelope | None:
    """Parse Redis cache JSON into a typed envelope."""
    decoded = json_loads(raw)
    if not is_json_dict(decoded):
        return None

    if "value" not in decoded:
        if not is_json_value(decoded):
            return None
        return {"value": decoded, "ttl_seconds": None}

    raw_value = decoded.get("value")
    if not is_json_value(raw_value):
        return None
    ttl_seconds = decoded.get("ttl_seconds")
    if ttl_seconds is not None and not isinstance(ttl_seconds, int):
        ttl_seconds = None
    return {"value": raw_value, "ttl_seconds": ttl_seconds}


@final
class CacheEntry:
    """A single cache entry wrapping a value with an optional time-to-live."""

    value: JsonValue
    created_at: float
    ttl_seconds: int | None

    def __init__(self, value: JsonValue, ttl_seconds: int | None = None):
        """Initialize a cache entry.

        Args:
            value: The cached value.
            ttl_seconds: Optional expiration duration in seconds. None means no expiry.
        """
        self.value = value
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check whether the entry has exceeded its TTL.

        Returns:
            True if the entry is expired, False if still valid or TTL is unlimited.
        """
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds


@final
class InMemoryCache:
    """In-memory cache with optional Redis backing."""

    _cache: dict[str, CacheEntry]
    _redis: RedisProvider | None
    _max_size: int

    def __init__(self, redis: RedisProvider | None = None) -> None:
        """Initialize an in-memory cache with optional Redis write-through.

        Args:
            redis: Optional RedisProvider for persistence. If None, cache is memory-only.
        """
        self._cache = {}
        self._redis = redis
        self._max_size = 1000  # Maximum entries in memory

    async def get(self, key: str) -> JsonValue | None:
        """Retrieve a value by key, checking in-memory cache first, then Redis.

        Args:
            key: The cache key to look up.

        Returns:
            The cached value, or None if not found or expired.
        """
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
                data_str = await self._redis.get(key)
                if data_str:
                    data = _parse_cache_envelope(data_str)
                    if data is None:
                        return None
                    # Load into memory cache
                    self._cache[key] = CacheEntry(
                        data["value"], ttl_seconds=data.get("ttl_seconds")
                    )
                    return data["value"]
            except Exception as e:
                logger.debug("Redis get failed for %s: %s", key, e)

        return None

    async def set(self, key: str, value: JsonValue, ttl_seconds: int | None = None) -> None:
        """Store a value in both the in-memory cache and the Redis backing store.

        Evicts the oldest entry when the in-memory cache reaches max capacity.

        Args:
            key: The cache key to store.
            value: The value to cache.
            ttl_seconds: Optional expiration in seconds for both memory and Redis.
        """
        # Store in memory
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(value, ttl_seconds)

        # Store in Redis if available
        if self._redis:
            try:
                data: _CacheEnvelope = {"value": value, "ttl_seconds": ttl_seconds}
                await self._redis.set(key, json_dumps(data), ex=ttl_seconds)
            except Exception as e:
                logger.debug("Redis set failed for %s: %s", key, e)

    async def delete(self, key: str) -> None:
        """Remove a key from both in-memory and Redis caches.

        Args:
            key: The cache key to delete.
        """
        _ = self._cache.pop(key, None)
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception as e:
                logger.debug("Redis delete failed for %s: %s", key, e)

    def clear(self) -> None:
        """Clear all entries from the in-memory cache (does not affect Redis)."""
        self._cache.clear()

    def cleanup_expired(self) -> None:
        """Scan and remove all expired entries from the in-memory cache."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]


# Global cache instances
_settings_cache: InMemoryCache | None = None
_session_cache: InMemoryCache | None = None
_query_cache: InMemoryCache | None = None


def _make_redis_provider() -> RedisProvider | None:
    """Create a RedisProvider from platform config if Redis is enabled.

    Returns:
        A configured RedisProvider, or None if Redis is disabled or has no URL.
    """
    config = get_core_config()
    if not config.redis.enabled or not config.redis.url:
        return None
    return RedisProvider.from_url(config.redis.url)


def get_settings_cache() -> InMemoryCache:
    """Get or create the singleton cache instance for agent/dispatcher settings.

    Returns:
        The singleton InMemoryCache instance backed by Redis (if available).
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = InMemoryCache(redis=_make_redis_provider())
    return _settings_cache


def get_session_cache() -> InMemoryCache:
    """Get or create the singleton cache instance for session state.

    Returns:
        The singleton InMemoryCache instance backed by Redis (if available).
    """
    global _session_cache
    if _session_cache is None:
        _session_cache = InMemoryCache(redis=_make_redis_provider())
    return _session_cache


def get_query_cache() -> InMemoryCache:
    """Get or create the singleton cache instance for query result caching.

    Returns:
        The singleton InMemoryCache instance backed by Redis (if available).
    """
    global _query_cache
    if _query_cache is None:
        _query_cache = InMemoryCache(redis=_make_redis_provider())
    return _query_cache


def make_query_cache_key(query: str, tenant_id: str | None = None) -> str:
    """Build a deterministic cache key from a query string and optional tenant.

    Uses a truncated SHA-256 hash of the query to prevent overly long keys.

    Args:
        query: The raw query text to hash.
        tenant_id: Optional tenant prefix for multi-tenant isolation.

    Returns:
        A namespaced cache key string.
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"query:{tenant_prefix}{query_hash}"


def make_session_cache_key(session_id: str, tenant_id: str | None = None) -> str:
    """Build a cache key for session data.

    Args:
        session_id: The unique session identifier.
        tenant_id: Optional tenant prefix for multi-tenant isolation.

    Returns:
        A namespaced cache key string.
    """
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"session:{tenant_prefix}{session_id}"


def make_settings_cache_key(setting_name: str, tenant_id: str | None = None) -> str:
    """Build a cache key for a named configuration setting.

    Args:
        setting_name: The setting identifier (e.g., "model_config", "max_tokens").
        tenant_id: Optional tenant prefix for multi-tenant isolation.

    Returns:
        A namespaced cache key string.
    """
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
