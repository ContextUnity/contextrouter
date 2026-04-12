"""Exception hierarchy for cu.router.

Service-specific base class. All shared exceptions, ErrorRegistry, and gRPC
error handlers live in cu.core.exceptions — import them directly from there.

Usage:
    from contextunity.router.core.exceptions import ContextrouterError
    from contextunity.core.exceptions import ModelError, grpc_error_handler
"""

from __future__ import annotations

from contextunity.core.exceptions import ContextUnityError


class ContextrouterError(ContextUnityError):
    """Base exception for cu.router.

    Inherits from ContextUnityError so that centralized gRPC error handlers
    in cu.core catch router-specific exceptions automatically.
    """

    pass
