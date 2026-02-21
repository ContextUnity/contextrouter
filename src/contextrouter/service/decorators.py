"""gRPC error handling decorators for ContextRouter dispatcher service."""

from __future__ import annotations

import grpc
from contextcore import get_context_unit_logger

from contextrouter.service.helpers import make_response, parse_unit

logger = get_context_unit_logger(__name__)


def grpc_error_handler(func):
    """Decorator for gRPC error handling."""

    async def wrapper(self, request, context):
        try:
            return await func(self, request, context)
        except ValueError as e:
            logger.error("Validation error in %s: %s", func.__name__, e)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            unit = parse_unit(request)
            return make_response(
                payload={"error": str(e), "error_type": "validation"},
                trace_id=str(unit.trace_id),
                provenance=list(unit.provenance) + [f"router:{func.__name__}:error"],
                security=unit.security,
            )
        except PermissionError as e:
            logger.warning("Permission denied in %s: %s", func.__name__, e)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(e))
            unit = parse_unit(request)
            return make_response(
                payload={"error": str(e), "error_type": "permission_denied"},
                trace_id=str(unit.trace_id),
                provenance=list(unit.provenance) + [f"router:{func.__name__}:permission_denied"],
                security=unit.security,
            )
        except Exception as e:
            err_msg = str(e) or repr(e)
            logger.exception("Unexpected error in %s: %s", func.__name__, err_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {err_msg}")
            unit = parse_unit(request)
            return make_response(
                payload={"error": err_msg, "error_type": type(e).__name__},
                trace_id=str(unit.trace_id),
                provenance=list(unit.provenance) + [f"router:{func.__name__}:error"],
                security=unit.security,
            )

    return wrapper


def grpc_stream_error_handler(func):
    """Decorator for gRPC streaming error handling."""

    async def wrapper(self, request, context):
        try:
            async for item in func(self, request, context):
                yield item
        except ValueError as e:
            logger.error("Validation error in %s: %s", func.__name__, e)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            yield make_response(
                payload={"error": str(e), "error_type": "validation"},
                provenance=[f"router:{func.__name__}:error"],
            )
        except PermissionError as e:
            logger.warning("Permission denied in %s: %s", func.__name__, e)
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(e))
            yield make_response(
                payload={"error": str(e), "error_type": "permission_denied"},
                provenance=[f"router:{func.__name__}:permission_denied"],
            )
        except Exception as e:
            err_msg = str(e) or repr(e)
            logger.exception("Unexpected error in %s: %s", func.__name__, err_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {err_msg}")
            yield make_response(
                payload={"error": err_msg, "error_type": type(e).__name__},
                provenance=[f"router:{func.__name__}:error"],
            )

    return wrapper


__all__ = ["grpc_error_handler", "grpc_stream_error_handler"]
