"""gRPC error handling decorators for contextunity.router dispatcher service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import ParamSpec

from contextunity.core import contextunit_pb2
from contextunity.core.grpc_errors import (
    grpc_error_handler as core_unary_handler,
)
from contextunity.core.grpc_errors import (
    grpc_stream_error_handler as core_stream_handler,
)
from contextunity.core.types import (
    GrpcStreamErrorResponseFactory,
    GrpcUnaryErrorResponseFactory,
)

from contextunity.router.service.helpers import router_error_response_factory

P = ParamSpec("P")
ContextUnit = contextunit_pb2.ContextUnit
_RouterErrorFactory = GrpcUnaryErrorResponseFactory[ContextUnit]
_RouterStreamErrorFactory = GrpcStreamErrorResponseFactory[ContextUnit]


def grpc_error_handler(
    method: Callable[P, Coroutine[object, object, ContextUnit]],
) -> Callable[P, Coroutine[object, object, ContextUnit]]:
    """Wrap a unary gRPC method with the router's error response factory."""
    factory: _RouterErrorFactory = router_error_response_factory
    return core_unary_handler(method, response_factory=factory)


def grpc_stream_error_handler(
    method: Callable[P, AsyncIterator[ContextUnit]],
) -> Callable[P, AsyncIterator[ContextUnit]]:
    """Wrap a streaming gRPC method with the router's error response factory."""
    factory: _RouterStreamErrorFactory = router_error_response_factory
    return core_stream_handler(method, response_factory=factory)


__all__ = ["grpc_error_handler", "grpc_stream_error_handler"]
