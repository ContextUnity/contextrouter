"""Exception hierarchy for contextrouter.

Service-specific base class. All shared exceptions, ErrorRegistry, and gRPC
error handlers live in contextcore.exceptions — import them directly from there.

Usage:
    from contextrouter.core.exceptions import ContextrouterError
    from contextcore.exceptions import ModelError, grpc_error_handler
"""

from __future__ import annotations

from contextcore.exceptions import ContextUnityError


class ContextrouterError(ContextUnityError):
    """Base exception for contextrouter.

    Inherits from ContextUnityError so that centralized gRPC error handlers
    in contextcore catch router-specific exceptions automatically.
    """

    pass
