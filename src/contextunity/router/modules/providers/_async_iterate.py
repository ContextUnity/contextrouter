"""Shared async-iteration helper for vendor SDK objects at provider boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class _AsyncIterableObj(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...


async def async_iterate(stream_obj: object) -> AsyncIterator[object]:
    """Iterate an async stream from a vendor SDK object."""
    if not isinstance(stream_obj, _AsyncIterableObj):
        raise TypeError("Expected an async iterable vendor response")
    async for item in stream_obj:
        yield item


__all__ = ["async_iterate"]
