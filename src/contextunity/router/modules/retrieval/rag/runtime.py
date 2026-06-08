"""Per-request runtime settings for RAG (contextvars).
This is intentionally scoped to retrieval/RAG usage. The core framework should not
own product-capability settings.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from .types import RuntimeRagSettings

_runtime_settings: ContextVar[RuntimeRagSettings | None] = ContextVar(
    "contextunity.router_runtime_rag_settings", default=None
)


@contextmanager
def use_runtime_settings(settings: RuntimeRagSettings | None) -> Generator[None]:
    """Set *settings* as the active RAG runtime config for the current context; reset on exit."""
    token = _runtime_settings.set(settings)
    try:
        yield
    finally:
        # When async generators are closed from a different task/context (e.g. client disconnect),
        # contextvars may raise "Token was created in a different Context". In that case, there's
        # nothing useful to reset in the current context, so we swallow the error.
        try:
            _runtime_settings.reset(token)
        except ValueError:
            pass


def get_runtime_settings() -> RuntimeRagSettings:
    """Return the current task-local RAG settings, defaulting to an empty dict."""
    val = _runtime_settings.get()
    return val if isinstance(val, dict) else {}
