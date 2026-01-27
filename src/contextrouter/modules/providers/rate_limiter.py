import logging

from .redis import RedisProvider

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter for ContextRouter.
    """

    def __init__(self, redis: RedisProvider):
        self.redis = redis

    async def is_allowed(self, identifier: str, limit: int, window_seconds: int) -> bool:
        """
        Check if the identifier is allowed to make a request.
        Identifier can be a token_id or IP address.
        """
        key = f"rl:{identifier}"
        try:
            current_count = await self.redis.get(key)
            if current_count is None:
                await self.redis.set(key, "1", ex=window_seconds)
                return True

            if int(current_count) >= limit:
                logger.warning(f"Rate limit exceeded for {identifier}")
                return False

            # Increment
            # redis.asyncio might not have 'incr' directly exposed or it depends on version
            # Usually it's there
            await self.redis.set(key, str(int(current_count) + 1), ex=window_seconds)
            return True
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open to avoid blocking users on redis failure
