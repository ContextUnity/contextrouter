"""Async retry utility for transient external API failures.

Keep this dependency-free (no tenacity) to minimize runtime footprint.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_with_backoff_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.4,
    max_delay_s: float = 4.0,
    jitter_s: float = 0.2,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Run async fn() with exponential backoff (non-blocking sleep).

    Args:
        fn: Async callable to retry
        attempts: Total attempts including first try
        base_delay_s: Initial delay (doubles each retry: 0.4 → 0.8 → 1.6)
        max_delay_s: Cap on delay
        jitter_s: Random jitter added to each delay
        retry_on: Exception types to catch and retry
    """
    if attempts <= 1:
        return await fn()

    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return await fn()
        except retry_on as e:
            last_exc = e
            if i >= attempts - 1:
                break
            delay = min(max_delay_s, base_delay_s * (2**i)) + (random.random() * jitter_s)
            await asyncio.sleep(delay)

    if last_exc is None:
        raise RuntimeError("Retry failed but no exception was captured")
    raise last_exc
