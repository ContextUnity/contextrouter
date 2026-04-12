"""Core interfaces (ABCs + Protocols).

These interfaces are intentionally small and transport-agnostic.
Business logic lives in modules; orchestration lives in brain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Protocol,
    runtime_checkable,
)

from contextunity.core import get_contextunit_logger

if TYPE_CHECKING:
    from contextunity.core import ContextUnit
    from contextunity.core.tokens import ContextToken

    from contextunity.router.core.state import AgentState
else:
    ContextUnit = Any  # type: ignore[misc,assignment]
    ContextToken = Any  # type: ignore[misc,assignment]
    AgentState = Any  # type: ignore[misc,assignment]

logger = get_contextunit_logger(__name__)


class BaseAgent(ABC):
    """Base class for LangGraph nodes (strict: nodes are classes).

    Implementations must be async and return partial state updates.
    """

    def __init__(self, registry: object | None = None) -> None:
        # Registry access (agents can discover connectors/transformers/providers/models).
        self.registry = registry

    @abstractmethod
    async def process(self, state: AgentState) -> dict[str, Any]:
        raise NotImplementedError

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        return await self.process(state)


@runtime_checkable
class IRead(Protocol):
    """Read interface (optionally secured; enforced when security enabled)."""

    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: ContextToken,
    ) -> list[ContextUnit]: ...


@runtime_checkable
class IWrite(Protocol):
    """Write interface (optionally secured; enforced when security enabled)."""

    async def write(self, data: ContextUnit, *, token: ContextToken) -> None: ...


class BaseConnector(ABC):
    """Sources: produce raw data wrapped in ContextUnit."""

    @abstractmethod
    async def connect(self) -> AsyncIterator[ContextUnit]:
        raise NotImplementedError


class BaseTransformer(ABC):
    """Logic pipes: pure-ish transformation over ContextUnit."""

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}

    def configure(self, params: dict[str, Any] | None) -> None:
        """Optional configuration hook.

        FlowManager will call this when it cannot pass params via `__init__(**params)`.
        """

        self._params = dict(params or {})

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @abstractmethod
    async def transform(self, envelope: ContextUnit) -> ContextUnit:
        raise NotImplementedError


class BaseProvider(ABC):
    """Sinks: accept ContextUnit and persist/return it somewhere."""

    @abstractmethod
    async def sink(self, envelope: ContextUnit, *, token: ContextToken) -> Any:
        raise NotImplementedError


__all__ = [
    "BaseAgent",
    "BaseConnector",
    "BaseTransformer",
    "BaseProvider",
    "IRead",
    "IWrite",
]
