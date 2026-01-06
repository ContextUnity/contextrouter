"""Core interfaces (ABCs + Protocols).

These interfaces are intentionally small and transport-agnostic.
Business logic lives in modules; orchestration lives in brain.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from contextrouter.core.bisquit import BisquitEnvelope
    from contextrouter.core.state import AgentState
    from contextrouter.core.tokens import BiscuitToken
else:
    BisquitEnvelope = Any  # type: ignore[misc,assignment]
    BiscuitToken = Any  # type: ignore[misc,assignment]
    AgentState = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def secured(permission: str | None = None) -> Callable:
    """Decorator to enforce Biscuit security on IRead/IWrite methods.

    If permission is None, it uses the default permission from core config.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            from contextrouter.core.exceptions import SecurityError
            from contextrouter.core.tokens import AccessManager

            token = kwargs.get("token")
            if token is None:
                # Try to find token in positional args if not in kwargs
                # (Assumes standard signature: read(query, limit, filters, token))
                # or write(data, token)
                for arg in args:
                    from contextrouter.core.tokens import BiscuitToken

                    if isinstance(arg, BiscuitToken):
                        token = arg
                        break

            if token is None and "sink" not in func.__name__:
                # We allow sink to be called without token if it delegates to write which is secured
                pass

            access = AccessManager.from_core_config()
            try:
                if "read" in func.__name__:
                    access.verify_read(token, permission=permission)
                elif "write" in func.__name__ or "sink" in func.__name__:
                    access.verify_write(token, permission=permission)
            except Exception as e:
                # Map to our internal exception hierarchy
                raise SecurityError(f"Security verification failed: {str(e)}") from e

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


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
        token: BiscuitToken,
    ) -> list[BisquitEnvelope]: ...


@runtime_checkable
class IWrite(Protocol):
    """Write interface (optionally secured; enforced when security enabled)."""

    async def write(self, data: BisquitEnvelope, *, token: BiscuitToken) -> None: ...


class BaseConnector(ABC):
    """Sources: produce raw data wrapped in BisquitEnvelope."""

    @abstractmethod
    async def connect(self) -> AsyncIterator[BisquitEnvelope]:
        raise NotImplementedError


class BaseTransformer(ABC):
    """Logic pipes: pure-ish transformation over BisquitEnvelope."""

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
    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        raise NotImplementedError


class BaseProvider(ABC):
    """Sinks: accept BisquitEnvelope and persist/return it somewhere."""

    @abstractmethod
    async def sink(self, envelope: BisquitEnvelope, *, token: BiscuitToken) -> Any:
        raise NotImplementedError


__all__ = [
    "BaseAgent",
    "BaseConnector",
    "BaseTransformer",
    "BaseProvider",
    "IRead",
    "IWrite",
]
