"""Router gRPC Service module."""

from .dispatcher_service import DispatcherService
from .server import serve

__all__ = ["DispatcherService", "serve"]
