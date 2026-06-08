"""Rate limiter for contextunity.router.
Redis-based sliding window rate limiter for API and agent request throttling.
Supports per-token, per-tenant, and per-IP limiting.
Exception handling uses contextunity.core.exceptions hierarchy.
Usage:
    from contextunity.router.modules.providers.rate_limiter import RateLimiter
    limiter = RateLimiter(redis)
    if not await limiter.is_allowed("user:123", limit=100, window_seconds=60):
        raise RateLimitExceeded("Too many requests")
"""

from __future__ import annotations

import time
from typing import TypedDict

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ProviderError

from .redis import RedisProvider

logger = get_contextunit_logger(__name__)


class RateLimitRemaining(TypedDict):
    limit: int
    remaining: int
    reset: int


def _zcard_from_pipeline_results(results: list[object]) -> int:
    """Parse the ``zcard`` slot from a rate-limit pipeline ``execute`` result."""
    if len(results) < 2:
        return 0
    count_obj = results[1]
    if isinstance(count_obj, int):
        return count_obj
    if isinstance(count_obj, float):
        return int(count_obj)
    return 0


class RateLimitExceeded(ProviderError):
    """Raised when a rate limit is exceeded.

    Inherits from ProviderError (contextunity.core.exceptions) since
    rate limiting is a provider-level concern.
    """

    identifier: str
    limit: int
    window_seconds: int
    retry_after: int | None

    def __init__(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
        *,
        retry_after: int | None = None,
    ):
        """Store throttle context (identifier, limits) and format the error message."""
        self.identifier = identifier
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {identifier}: {limit} requests per {window_seconds}s",
            code="RATE_LIMIT_EXCEEDED",
        )


class RateLimiter:
    """Redis-based sliding window rate limiter for contextunity.router.

    Features:
        - Per-identifier limits (token_id, tenant_id, IP)
        - Sliding window algorithm (more accurate than fixed windows)
        - Fail-open on Redis errors (does not block users)
        - Returns remaining quota for response headers

    Args:
        redis: RedisProvider instance for state storage.
        key_prefix: Optional prefix for Redis keys.
    """

    _redis: RedisProvider
    _prefix: str

    def __init__(self, redis: RedisProvider, *, key_prefix: str = "rl"):
        """Bind the underlying Redis provider and key namespace."""
        self._redis = redis
        self._prefix = key_prefix

    def _key(self, identifier: str) -> str:
        """Build redis key for the given identifier."""
        return f"{self._prefix}:{identifier}"

    async def is_allowed(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> bool:
        """Check if the identifier is allowed to make a request.

        Args:
            identifier: Unique ID for the rate limit bucket
                (e.g., "token:abc123", "ip:1.2.3.4", "tenant:default").
            limit: Maximum number of requests allowed in the window.
            window_seconds: Time window in seconds.

        Returns:
            True if allowed, False if rate limited.

        Note:
            Fails open on Redis errors to avoid blocking legitimate traffic.
        """
        key = self._key(identifier)
        now = time.time()
        window_start = now - window_seconds

        try:
            # Sliding window: use sorted set with timestamps
            pipe = self._redis.pipeline()

            # Remove entries outside the window
            _ = pipe.zremrangebyscore(key, 0, window_start)
            # Count entries in the window
            _ = pipe.zcard(key)
            # Add current request
            _ = pipe.zadd(key, {str(now): now})
            # Set TTL so Redis auto-cleans
            _ = pipe.expire(key, window_seconds + 1)

            results = await pipe.execute()
            current_count = _zcard_from_pipeline_results(results)

            if current_count >= limit:
                logger.warning(
                    "Rate limit exceeded: identifier=%s count=%d limit=%d window=%ds",
                    identifier,
                    current_count,
                    limit,
                    window_seconds,
                )
                return False

            return True

        except Exception as e:
            # Fail open — do not block users on Redis failure
            logger.error("Rate limit check failed (fail-open): %s", e)
            return True

    async def check_or_raise(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> None:
        """Check rate limit and raise RateLimitExceeded if exceeded.

        This is the strict version — use in API middleware where you want
        to return 429 responses.

        Args:
            identifier: Rate limit bucket ID.
            limit: Max requests per window.
            window_seconds: Window duration.

        Raises:
            RateLimitExceeded: If the rate limit is exceeded.
        """
        if not await self.is_allowed(identifier, limit, window_seconds):
            raise RateLimitExceeded(
                identifier=identifier,
                limit=limit,
                window_seconds=window_seconds,
            )

    async def get_remaining(
        self,
        identifier: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitRemaining:
        """Return ``{limit, remaining, reset}`` for HTTP rate-limit response headers."""
        key = self._key(identifier)
        now = time.time()
        window_start = now - window_seconds

        try:
            pipe = self._redis.pipeline()
            _ = pipe.zremrangebyscore(key, 0, window_start)
            _ = pipe.zcard(key)
            results = await pipe.execute()
            current_count = _zcard_from_pipeline_results(results)

            return {
                "limit": limit,
                "remaining": max(0, limit - current_count),
                "reset": int(now + window_seconds),
            }

        except Exception:
            return {
                "limit": limit,
                "remaining": limit,
                "reset": int(now + window_seconds),
            }


__all__ = ["RateLimiter", "RateLimitExceeded", "RateLimitRemaining"]
