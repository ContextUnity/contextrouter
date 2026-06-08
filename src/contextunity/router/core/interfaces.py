"""Core interfaces (ABCs + Protocols).
These interfaces are intentionally small and transport-agnostic.
Business logic lives in modules; orchestration lives in brain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk.interfaces import BaseConnector, BaseTransformer
from contextunity.core.types import JsonDict

if TYPE_CHECKING:
    from contextunity.core import ContextUnit

    from contextunity.router.cortex.types import GraphState, StateUpdate

logger = get_contextunit_logger(__name__)


class BaseAgent(ABC):
    """Base class for LangGraph nodes (strict: nodes are classes).

    Implementations must be async and return partial state updates.
    """

    def __init__(self, registry: object | None = None) -> None:
        """Initialize with optional component registry.

        Args:
            registry: Component registry for discovering connectors,
                transformers, providers, and models at runtime.
        """
        # Registry access (agents can discover connectors/transformers/providers/models).
        self.registry: object | None = registry

    @abstractmethod
    async def process(self, state: GraphState) -> StateUpdate:
        """Execute agent logic on the current graph state.

        Subclasses must implement this to perform their domain-specific
        processing and return a partial state update.

        Args:
            state: Current LangGraph execution state.

        Returns:
            Partial state update dict to merge into the graph.
        """
        raise NotImplementedError

    async def __call__(self, state: GraphState) -> StateUpdate:
        """Callable shorthand — delegates to ``process``.

        Args:
            state: Current graph execution state.

        Returns:
            Result of ``self.process(state)``.
        """
        return await self.process(state)


@runtime_checkable
class IRead(Protocol):
    """Read interface for providers.

    Security is enforced at the SecureTool/secure_node boundary.
    """

    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: JsonDict | None = None,
    ) -> list[ContextUnit]:
        """Retrieve ContextUnits matching a query.

        Args:
            query: Natural-language or structured search expression.
            limit: Maximum results to return.
            filters: Optional provider-specific filter predicates.

        Returns:
            Ordered list of matching ContextUnits.
        """
        ...


@runtime_checkable
class IWrite(Protocol):
    """Write interface for providers.

    Security is enforced at the SecureTool/secure_node boundary.
    """

    async def write(self, data: ContextUnit) -> None:
        """Persist a ContextUnit to the underlying store.

        Args:
            data: ContextUnit payload to persist.
        """
        ...


class BaseProvider(ABC):
    """Sinks: accept ContextUnit and persist/return it somewhere.

    Security is enforced at the SecureTool/secure_node boundary.
    """

    @abstractmethod
    async def sink(self, unit: ContextUnit) -> None:
        """Persist a ContextUnit to the target storage.

        Args:
            unit: ContextUnit payload to persist.
        """
        raise NotImplementedError


__all__ = [
    "BaseAgent",
    "BaseConnector",
    "BaseTransformer",
    "BaseProvider",
    "IRead",
    "IWrite",
]
