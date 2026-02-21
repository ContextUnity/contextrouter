"""Redis Memory Tool for Dispatcher Agent.

Provides memory operations with ContextUnit governance and safe caching strategies.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain_core.tools import tool

from contextrouter.core import get_core_config
from contextrouter.modules.providers.redis import RedisProvider

logger = logging.getLogger(__name__)

# Singleton Redis provider instance
_redis_provider: RedisProvider | None = None


def get_redis_provider() -> RedisProvider:
    """Get or create Redis provider singleton."""
    global _redis_provider
    if _redis_provider is None:
        config = get_core_config()
        _redis_provider = RedisProvider(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
        )
    return _redis_provider


# ============================================================================
# Cache Key Strategies
# ============================================================================


def _make_query_cache_key(query: str, tenant_id: str | None = None) -> str:
    """Create cache key for query results."""
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"cache:query:{tenant_prefix}{query_hash}"


def _make_session_key(session_id: str, tenant_id: str | None = None) -> str:
    """Create cache key for session data."""
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"session:{tenant_prefix}{session_id}"


def _make_config_key(config_name: str, tenant_id: str | None = None) -> str:
    """Create cache key for configuration."""
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"config:{tenant_prefix}{config_name}"


def _make_memory_key(key: str, session_id: str, tenant_id: str | None = None) -> str:
    """Create cache key for memory storage."""
    tenant_prefix = f"{tenant_id}:" if tenant_id else ""
    return f"memory:{tenant_prefix}{session_id}:{key}"


# ============================================================================
# Memory Tools
# ============================================================================


@tool
async def store_memory(
    key: str,
    value: str,
    session_id: str,
    tenant_id: str | None = None,
    ttl_seconds: int = 3600,
) -> dict[str, Any]:
    """Store a value in Redis memory for later retrieval.

    This tool allows the agent to store information in memory that persists
    across tool calls within a session. Use this for:
    - Storing user preferences
    - Caching intermediate results
    - Remembering context from previous interactions

    IMPORTANT: Follow ContextUnit governance:
    - Only store data you have permission to store
    - Respect tenant isolation (use tenant_id when available)
    - Set appropriate TTL to avoid stale data
    - Never store sensitive credentials or tokens

    Args:
        key: Unique key for the memory entry
        value: Value to store (will be JSON-serialized)
        session_id: Session identifier for isolation
        tenant_id: Optional tenant identifier for multi-tenant isolation
        ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)

    Returns:
        Dict with success status and stored key
    """
    try:
        tenant_id = tenant_id or "default"
        redis = get_redis_provider()
        memory_key = _make_memory_key(key, session_id, tenant_id)

        # Store as JSON for structured data
        data = {
            "value": value,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "timestamp": str(__import__("datetime").datetime.now().isoformat()),
        }

        await redis.set(memory_key, json.dumps(data), ex=ttl_seconds)
        logger.debug("Stored memory: %s", memory_key)

        return {
            "success": True,
            "key": key,
            "memory_key": memory_key,
            "ttl_seconds": ttl_seconds,
        }
    except Exception as e:
        logger.error("Failed to store memory: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def retrieve_memory(
    key: str,
    session_id: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
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
            return {"success": False, "found": False, "key": key}

        data = json.loads(data_str)
        logger.debug("Retrieved memory: %s", memory_key)

        return {
            "success": True,
            "found": True,
            "key": key,
            "value": data.get("value"),
            "timestamp": data.get("timestamp"),
        }
    except Exception as e:
        logger.error("Failed to retrieve memory: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def cache_query_result(
    query: str,
    result: str,
    tenant_id: str | None = None,
    ttl_seconds: int = 1800,
) -> dict[str, Any]:
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

        data = {
            "query": query,
            "result": result,
            "tenant_id": tenant_id,
            "timestamp": str(__import__("datetime").datetime.now().isoformat()),
        }

        await redis.set(cache_key, json.dumps(data), ex=ttl_seconds)
        logger.debug("Cached query result: %s", cache_key)

        return {
            "success": True,
            "cache_key": cache_key,
            "ttl_seconds": ttl_seconds,
        }
    except Exception as e:
        logger.error("Failed to cache query result: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def get_cached_query(
    query: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
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
            return {"success": False, "found": False, "query": query}

        data = json.loads(data_str)
        logger.debug("Retrieved cached query: %s", cache_key)

        return {
            "success": True,
            "found": True,
            "query": query,
            "result": data.get("result"),
            "timestamp": data.get("timestamp"),
        }
    except Exception as e:
        logger.error("Failed to get cached query: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def get_session_data(
    session_id: str,
    tenant_id: str | None = None,
) -> dict[str, Any]:
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
            return {
                "success": True,
                "found": False,
                "session_id": session_id,
                "data": {},
            }

        data = json.loads(data_str)
        logger.debug("Retrieved session data: %s", session_key)

        return {
            "success": True,
            "found": True,
            "session_id": session_id,
            "data": data,
        }
    except Exception as e:
        logger.error("Failed to get session data: %s", e)
        return {"success": False, "error": str(e)}


@tool
async def clear_memory(
    key: str | None = None,
    session_id: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Clear memory entries.

    Clears specific memory entry or all memory for a session.

    IMPORTANT: Only clear memory you have permission to clear.
    Respect tenant isolation - never clear other tenants' data.

    Args:
        key: Specific key to clear (if None, clears all session memory)
        session_id: Session identifier (required if key is None)
        tenant_id: Optional tenant identifier

    Returns:
        Dict with success status
    """
    try:
        redis = get_redis_provider()

        if key and session_id:
            # Clear specific key
            memory_key = _make_memory_key(key, session_id, tenant_id)
            await redis.delete(memory_key)
            logger.debug("Cleared memory: %s", memory_key)
            return {"success": True, "cleared": key}
        elif session_id:
            # Clear all session memory (would need pattern matching)
            # For now, return error - implement pattern deletion if needed
            return {
                "success": False,
                "error": "Bulk session clear not implemented. Clear specific keys.",
            }
        else:
            return {
                "success": False,
                "error": "Either key+session_id or session_id required",
            }
    except Exception as e:
        logger.error("Failed to clear memory: %s", e)
        return {"success": False, "error": str(e)}


__all__ = [
    "store_memory",
    "retrieve_memory",
    "cache_query_result",
    "get_cached_query",
    "get_session_data",
    "clear_memory",
    "get_redis_provider",
]
