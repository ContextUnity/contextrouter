"""gRPC error handling decorators for contextunity.router dispatcher service."""

from __future__ import annotations

import functools

from contextunity.core.exceptions import (
    grpc_error_handler as core_unary_handler,
)
from contextunity.core.exceptions import (
    grpc_stream_error_handler as core_stream_handler,
)

from contextunity.router.service.helpers import router_error_response_factory

# Pre-bind the router's response factory to the core exception decorators
# so all current usages in the router (`@grpc_error_handler`) Just Work™.

grpc_error_handler = functools.partial(
    core_unary_handler, response_factory=router_error_response_factory
)

grpc_stream_error_handler = functools.partial(
    core_stream_handler, response_factory=router_error_response_factory
)

__all__ = ["grpc_error_handler", "grpc_stream_error_handler"]
