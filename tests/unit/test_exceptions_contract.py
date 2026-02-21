"""Contract tests for contextrouter exception hierarchy.

Verifies that:
1. ContextrouterError inherits from ContextUnityError (centralized hierarchy)
2. All error subclasses have stable error codes
3. ErrorRegistry contains expected base codes
4. gRPC error handlers are importable from contextcore
"""

from __future__ import annotations

from contextcore.exceptions import ContextUnityError, error_registry

from contextrouter.core.exceptions import ContextrouterError


def test_contextrouter_error_inherits_from_core() -> None:
    """ContextrouterError must be a subclass of ContextUnityError."""
    assert issubclass(ContextrouterError, ContextUnityError)


def test_contextrouter_error_has_code() -> None:
    """ContextrouterError must have a valid code from parent."""
    code = getattr(ContextrouterError, "code", None)
    assert isinstance(code, str) and code.strip()


def test_error_registry_contains_base_codes() -> None:
    reg = error_registry.all()
    for code in [
        "INTERNAL_ERROR",
        "CONFIGURATION_ERROR",
        "RETRIEVAL_ERROR",
        "INTENT_ERROR",
        "PROVIDER_ERROR",
        "CONNECTOR_ERROR",
        "MODEL_ERROR",
        "STORAGE_ERROR",
        "DB_CONNECTION_ERROR",
    ]:
        assert code in reg, f"missing {code} in error_registry"


def test_grpc_handlers_importable() -> None:
    """gRPC error handlers must be importable from contextcore."""
    from contextcore.exceptions import grpc_error_handler, grpc_stream_error_handler

    assert callable(grpc_error_handler)
    assert callable(grpc_stream_error_handler)
