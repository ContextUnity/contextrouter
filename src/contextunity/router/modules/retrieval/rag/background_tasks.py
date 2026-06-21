"""Strong references for fire-and-forget asyncio tasks in RAG retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)

_T = TypeVar("_T")

_background_tasks: set[asyncio.Task[object]] = set()


def spawn_background_task(coro: Coroutine[object, object, _T]) -> asyncio.Task[_T]:
    """Schedule *coro* and retain a strong reference until it completes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(done: asyncio.Task[object]) -> None:
        _background_tasks.discard(done)
        if done.cancelled():
            return
        exc = done.exception()
        if exc is not None:
            logger.error("Background task failed: %s", exc, exc_info=exc)

    task.add_done_callback(_on_done)
    return task


__all__ = ["spawn_background_task"]
