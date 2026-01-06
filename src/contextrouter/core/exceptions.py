"""Exception hierarchy for contextrouter (vendor-neutral)."""

from __future__ import annotations

from typing import Any, Callable, TypeVar, cast


class ContextrouterError(Exception):
    """Base exception for contextrouter."""

    code: str = "INTERNAL_ERROR"
    message: str = "An internal error occurred"

    def __init__(self, message: str | None = None, code: str | None = None, **kwargs: Any) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = kwargs
        super().__init__(self.message)


class ConfigurationError(ContextrouterError):
    """Invalid or missing configuration."""

    code: str = "CONFIGURATION_ERROR"


class RetrievalError(ContextrouterError):
    """Retrieval pipeline failure."""

    code: str = "RETRIEVAL_ERROR"


class IntentDetectionError(ContextrouterError):
    """Intent classification failure."""

    code: str = "INTENT_ERROR"


class ProviderError(ContextrouterError):
    """Storage/Provider layer failure."""

    code: str = "PROVIDER_ERROR"


class SecurityError(ContextrouterError):
    """Authorization/security failure (token missing/invalid/expired)."""

    code: str = "SECURITY_ERROR"


class ConnectorError(ContextrouterError):
    """Data connector failure."""

    code: str = "CONNECTOR_ERROR"


class ModelError(ContextrouterError):
    """LLM or Embedding model failure."""

    code: str = "MODEL_ERROR"


class IngestionError(ContextrouterError):
    """Ingestion pipeline failure."""

    code: str = "INGESTION_ERROR"


class GraphBuilderError(ContextrouterError):
    """Graph building failure."""

    code: str = "GRAPH_BUILDER_ERROR"


class TransformerError(ContextrouterError):
    """Data transformation failure."""

    code: str = "TRANSFORMER_ERROR"


class StorageError(ProviderError):
    """Specific error for database or storage operations."""

    code: str = "STORAGE_ERROR"


class DatabaseConnectionError(StorageError):
    """Failed to connect to the database."""

    code: str = "DB_CONNECTION_ERROR"


# ---- Error Registry for Protocol Mapping ------------------------------------

_E = TypeVar("_E", bound=type[ContextrouterError])


class ErrorRegistry:
    """Registry for mapping internal errors to external protocol codes."""

    def __init__(self) -> None:
        self._errors: dict[str, type[ContextrouterError]] = {}

    def register(self, code: str, error_cls: type[ContextrouterError]) -> None:
        self._errors[code] = error_cls

    def get(self, code: str) -> type[ContextrouterError] | None:
        return self._errors.get(code)

    def all(self) -> dict[str, type[ContextrouterError]]:
        return dict(self._errors)


error_registry = ErrorRegistry()


def register_error(code: str) -> Callable[[_E], _E]:
    """Decorator to register a custom error type."""

    def decorator(cls: _E) -> _E:
        error_registry.register(code, cls)
        return cls

    return cast(Callable[[_E], _E], decorator)


# Register base errors
error_registry.register("INTERNAL_ERROR", ContextrouterError)
error_registry.register("CONFIGURATION_ERROR", ConfigurationError)
error_registry.register("RETRIEVAL_ERROR", RetrievalError)
error_registry.register("INTENT_ERROR", IntentDetectionError)
error_registry.register("PROVIDER_ERROR", ProviderError)
error_registry.register("SECURITY_ERROR", SecurityError)
error_registry.register("CONNECTOR_ERROR", ConnectorError)
error_registry.register("MODEL_ERROR", ModelError)
error_registry.register("INGESTION_ERROR", IngestionError)
error_registry.register("GRAPH_BUILDER_ERROR", GraphBuilderError)
error_registry.register("TRANSFORMER_ERROR", TransformerError)
