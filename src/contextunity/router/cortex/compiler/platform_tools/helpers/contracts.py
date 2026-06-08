"""Shared contracts for platform tool executors and adapters."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from contextunity.router.cortex.types import GraphState, StateUpdate

# Platform tool executors receive the full graph state.
PlatformState = GraphState
PlatformResult = StateUpdate


@runtime_checkable
class PlatformToolFunc(Protocol):
    """Tool body invoked by ``make_platform_executor`` with injected state."""

    def __call__(
        self,
        state: GraphState,
        /,
    ) -> PlatformResult | Awaitable[PlatformResult]: ...


@runtime_checkable
class PlatformExecutor(Protocol):
    """Compiled platform node entrypoint: ``(state, validated config) → state update``."""

    async def __call__(
        self,
        state: PlatformState,
        config: BaseModel,
        /,
    ) -> PlatformResult: ...


@runtime_checkable
class PlatformAdapter(Protocol):
    """Async adapter with a single graph-state argument."""

    async def __call__(self, state: PlatformState, /) -> PlatformResult: ...


__all__ = [
    "PlatformAdapter",
    "PlatformExecutor",
    "PlatformResult",
    "PlatformState",
    "PlatformToolFunc",
]
