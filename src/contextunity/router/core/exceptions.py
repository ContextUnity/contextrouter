"""Exception hierarchy for contextunity.router.

Service-specific exceptions with stable error codes for gRPC mapping.
All codes use the ``ROUTER_`` prefix so the prefix-based fallback in
``core/grpc_errors.py`` maps them to ``grpc.StatusCode.INTERNAL`` by default.

Base class and infrastructure exceptions (ConfigurationError, SecurityError, etc.)
live in ``contextunity.core.exceptions`` — import them directly from there.

Usage::

    from contextunity.router.core.exceptions import (
        ContextrouterError,
        RouterLLMError,
        RouterStreamError,
    )
    from contextunity.core.exceptions import SecurityError
    from contextunity.core.grpc_errors import grpc_error_handler
"""

from __future__ import annotations

from contextunity.core.exceptions import ContextUnityError, register_error


@register_error("ROUTER_ERROR")
class ContextrouterError(ContextUnityError):
    """Base exception for contextunity.router.

    Inherits from ContextUnityError so that centralized gRPC error handlers
    in contextunity.core catch router-specific exceptions automatically.
    """

    code: str = "ROUTER_ERROR"
    message: str = "Router service error"


@register_error("ROUTER_GRAPH_BUILDER_ERROR")
class RouterGraphBuilderError(ContextrouterError):
    """Graph compilation or validation failed."""

    code: str = "ROUTER_GRAPH_BUILDER_ERROR"
    message: str = "Graph compilation failed"


@register_error("ROUTER_LLM_ERROR")
class RouterLLMError(ContextrouterError):
    """LLM invocation failed (model unavailable, rate-limited, malformed response)."""

    code: str = "ROUTER_LLM_ERROR"
    message: str = "LLM invocation failed"


@register_error("ROUTER_RETRIEVAL_ERROR")
class RouterRetrievalError(ContextrouterError):
    """Knowledge retrieval pipeline failed."""

    code: str = "ROUTER_RETRIEVAL_ERROR"
    message: str = "Knowledge retrieval failed"


@register_error("ROUTER_STREAM_ERROR")
class RouterStreamError(ContextrouterError):
    """Stream execution failed."""

    code: str = "ROUTER_STREAM_ERROR"
    message: str = "Stream execution failed"


@register_error("ROUTER_PII_ERROR")
class RouterPIIError(ContextrouterError):
    """Privacy (PII/anonymization) service failure."""

    code: str = "ROUTER_PII_ERROR"
    message: str = "Privacy service unavailable"


@register_error("ROUTER_TOOL_TIMEOUT")
class RouterToolTimeout(ContextrouterError):
    """Federated tool execution timed out."""

    code: str = "ROUTER_TOOL_TIMEOUT"
    message: str = "Federated tool execution timed out"


@register_error("ROUTER_PLUGIN_ERROR")
class RouterPluginError(ContextrouterError):
    """Router plugin failed."""

    code: str = "ROUTER_PLUGIN_ERROR"
    message: str = "Router plugin failed"


@register_error("ROUTER_REGISTRY_ERROR")
class RouterRegistryError(ContextrouterError):
    """Graph or agent registry operation failed."""

    code: str = "ROUTER_REGISTRY_ERROR"
    message: str = "Registry operation failed"


@register_error("ROUTER_CONNECTOR_ERROR")
class RouterConnectorError(ContextrouterError):
    """Data connector failure in router context."""

    code: str = "ROUTER_CONNECTOR_ERROR"
    message: str = "Data connector failed"


@register_error("ROUTER_VALIDATION_ERROR")
class RouterValidationError(ContextrouterError):
    """Input validation or configuration format failure."""

    code: str = "ROUTER_VALIDATION_ERROR"
    message: str = "Validation failed"


@register_error("ROUTER_STORAGE_ERROR")
class RouterStorageError(ContextrouterError):
    """Storage/provider backend failure."""

    code: str = "ROUTER_STORAGE_ERROR"
    message: str = "Storage operation failed"


@register_error("ROUTER_INTENT_DETECTION_ERROR")
class RouterIntentDetectionError(ContextrouterError):
    """Intent classification failed."""

    code: str = "ROUTER_INTENT_DETECTION_ERROR"
    message: str = "Intent detection failed"
