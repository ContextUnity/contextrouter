"""Redis Memory Tool for Dispatcher Agent.

Provides memory operations with ContextUnit governance and safe caching strategies.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps, json_loads
from contextunity.core.types import JsonDict, is_json_dict

from contextunity.router.core import get_core_config
from contextunity.router.langchain_boundaries import tool
from contextunity.router.modules.providers.redis import RedisProvider
from contextunity.router.modules.tools.schemas import RedisMemoryResult

logger = get_contextunit_logger(__name__)

# Singleton Redis provider instance
_redis_provider: RedisProvider | None = None


def get_redis_provider() -> RedisProvider:
    """Get or create Redis provider singleton."""
    global _redis_provider
    if _redis_provider is None:
        config = get_core_config()
        if config.redis.enabled and config.redis.url:
            _redis_provider = RedisProvider.from_url(config.redis.url)
        else:
            _redis_provider = RedisProvider(host="")  # no-op provider
    return _redis_provider


def _parse_redis_payload(data_str: str) -> JsonDict:
    parsed = json_loads(data_str)
    if is_json_dict(parsed):
        return parsed
    return {}


def _iso_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _normalize_tenant_id(tenant_id: str | None) -> str:
    return tenant_id or "default"


# ============================================================================
# Cache Key Strategies
# ============================================================================


def _make_query_cache_key(query: str, tenant_id: str | None = None) -> str:
    """Create cache key for query results."""
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    tenant_id = _normalize_tenant_id(tenant_id)
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"cache:query:{tenant_prefix}{query_hash}"


def _make_session_key(session_id: str, tenant_id: str | None = None) -> str:
    """Create cache key for session data."""
    tenant_id = _normalize_tenant_id(tenant_id)
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"session:{tenant_prefix}{session_id}"


def _make_memory_key(key: str, session_id: str, tenant_id: str | None = None) -> str:
    """Create cache key for memory entries."""
    tenant_id = _normalize_tenant_id(tenant_id)
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"memory:{tenant_prefix}{session_id}:{key}"


@tool
async def store_memory(
    key: str,
    value: str,
    session_id: str,
    tenant_id: str | None = None,
    ttl_seconds: int = 3600,
) -> RedisMemoryResult:
    """Store a value in Redis memory with session and tenant isolation.

    Use this for user-specific or session-specific data that should persist
    across agent turns within a session.

    Args:
        key: Key for the memory entry
        value: Value to store (string)
        session_id: Session identifier for isolation
        tenant_id: Optional tenant identifier for multi-tenant isolation
        ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)

    Returns:
        Dict with success status and stored key
    """
    try:
        tenant_id = _normalize_tenant_id(tenant_id)
        redis = get_redis_provider()
        memory_key = _make_memory_key(key, session_id, tenant_id)

        # Store as JSON for structured data
        data: JsonDict = {
            "value": value,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "timestamp": _iso_timestamp(),
        }

        await redis.set(memory_key, json_dumps(data), ex=ttl_seconds)
        logger.debug("Stored memory: %s", memory_key)

        return RedisMemoryResult(
            success=True,
            key=key,
            memory_key=memory_key,
            ttl_seconds=ttl_seconds,
        )
    except Exception as e:
        logger.error("Failed to store memory: %s", e)
        return RedisMemoryResult(success=False, error=str(e))


@tool
async def retrieve_memory(
    key: str,
    session_id: str,
    tenant_id: str | None = None,
) -> RedisMemoryResult:
    """Retrieve a value from Redis memory.

    Retrieves previously stored memory by key and session.

    Args:
        key: Key of the memory entry to retrieve
        session_id: Session identifier
        tenant_id: Optional tenant identifier

    Returns:
        Dict with value if found, or error if not found
    """
    try:
        redis = get_redis_provider()
        memory_key = _make_memory_key(key, session_id, tenant_id)

        data_str = await redis.get(memory_key)
        if not data_str:
            return RedisMemoryResult(success=False, found=False, key=key)

        data = _parse_redis_payload(data_str)
        logger.debug("Retrieved memory: %s", memory_key)

        stored = RedisMemoryResult(success=True, found=True, key=key)
        value = _optional_str(data.get("value"))
        if value is not None:
            stored["value"] = value
        timestamp = _optional_str(data.get("timestamp"))
        if timestamp is not None:
            stored["timestamp"] = timestamp
        return stored
    except Exception as e:
        logger.error("Failed to retrieve memory: %s", e)
        return RedisMemoryResult(success=False, error=str(e))


@tool
async def cache_query_result(
    query: str,
    result: str,
    tenant_id: str | None = None,
    ttl_seconds: int = 1800,
) -> RedisMemoryResult:
    """Cache a query result for faster future retrieval.

    Use this to cache expensive query results (e.g., LLM responses, API calls).
    The cache key is derived from the query hash, so identical queries will
    return cached results.

    IMPORTANT: Only cache non-sensitive, reusable results.
    - Do NOT cache user-specific data (use store_memory instead)
    - Do NOT cache results that change frequently
    - Set appropriate TTL based on data freshness requirements

    Args:
        query: The query string (used to generate cache key)
        result: The result to cache (will be JSON-serialized)
        tenant_id: Optional tenant identifier for isolation
        ttl_seconds: Time-to-live in seconds (default: 1800 = 30 minutes)

    Returns:
        Dict with success status and cache key
    """
    try:
        redis = get_redis_provider()
        cache_key = _make_query_cache_key(query, tenant_id)

        data: JsonDict = {
            "query": query,
            "result": result,
            "tenant_id": tenant_id,
            "timestamp": _iso_timestamp(),
        }

        await redis.set(cache_key, json_dumps(data), ex=ttl_seconds)
        logger.debug("Cached query result: %s", cache_key)

        return RedisMemoryResult(
            success=True,
            cache_key=cache_key,
            ttl_seconds=ttl_seconds,
        )
    except Exception as e:
        logger.error("Failed to cache query result: %s", e)
        return RedisMemoryResult(success=False, error=str(e))


@tool
async def get_cached_query(
    query: str,
    tenant_id: str | None = None,
) -> RedisMemoryResult:
    """Retrieve a cached query result.

    Checks if a query result is cached and returns it if found.

    Args:
        query: The query string to look up
        tenant_id: Optional tenant identifier

    Returns:
        Dict with cached result if found, or not_found status
    """
    try:
        redis = get_redis_provider()
        cache_key = _make_query_cache_key(query, tenant_id)

        data_str = await redis.get(cache_key)
        if not data_str:
            return RedisMemoryResult(success=False, found=False, query=query)

        data = _parse_redis_payload(data_str)
        logger.debug("Retrieved cached query: %s", cache_key)

        cached = RedisMemoryResult(success=True, found=True, query=query)
        result_value = _optional_str(data.get("result"))
        if result_value is not None:
            cached["result"] = result_value
        timestamp = _optional_str(data.get("timestamp"))
        if timestamp is not None:
            cached["timestamp"] = timestamp
        return cached
    except Exception as e:
        logger.error("Failed to get cached query: %s", e)
        return RedisMemoryResult(success=False, error=str(e))


@tool
async def get_session_data(
    session_id: str,
    tenant_id: str | None = None,
) -> RedisMemoryResult:
    """Retrieve all session data from Redis.

    Gets all stored data for a session, including memory entries and metadata.

    Args:
        session_id: Session identifier
        tenant_id: Optional tenant identifier

    Returns:
        Dict with session data
    """
    try:
        redis = get_redis_provider()
        session_key = _make_session_key(session_id, tenant_id)

        data_str = await redis.get(session_key)
        if not data_str:
            return RedisMemoryResult(
                success=True,
                found=False,
                session_id=session_id,
                data={},
            )

        data = _parse_redis_payload(data_str)
        logger.debug("Retrieved session data: %s", session_key)

        return RedisMemoryResult(
            success=True,
            found=True,
            session_id=session_id,
            data=data,
        )
    except Exception as e:
        logger.error("Failed to get session data: %s", e)
        return RedisMemoryResult(success=False, error=str(e))


@tool
async def clear_memory(
    key: str | None = None,
    session_id: str | None = None,
    tenant_id: str | None = None,
) -> RedisMemoryResult:
    """Clear memory entries from Redis.

    Can clear a specific key, all keys for a session, or all keys for a tenant.

    Args:
        key: Optional specific key to clear
        session_id: Optional session to clear all keys for
        tenant_id: Optional tenant identifier

    Returns:
        Dict with success status and what was cleared
    """
    try:
        redis = get_redis_provider()
        tenant_id = tenant_id or "default"

        if key and session_id:
            memory_key = _make_memory_key(key, session_id, tenant_id)
            await redis.delete(memory_key)
            cleared = f"memory:{memory_key}"
        elif session_id:
            session_key = _make_session_key(session_id, tenant_id)
            await redis.delete(session_key)
            cleared = f"session:{session_key}"
        else:
            return RedisMemoryResult(
                success=False,
                error="Must provide key+session_id or session_id to clear",
            )

        logger.debug("Cleared memory: %s", cleared)
        return RedisMemoryResult(success=True, cleared=cleared)
    except Exception as e:
        logger.error("Failed to clear memory: %s", e)
        return RedisMemoryResult(success=False, error=str(e))
