"""ContextVar-based runtime context for access tokens and provenance accumulation during graph execution."""

from __future__ import annotations

from contextvars import ContextVar, Token

from contextunity.core import ContextToken

_current_access_token: ContextVar[ContextToken | None] = ContextVar(
    "contextunity.router_current_access_token",
    default=None,
)

_provenance_accumulator: ContextVar[list[str | tuple[str, ...]] | None] = ContextVar(
    "contextunity.router_provenance_accum",
    default=None,
)


def init_provenance_accumulator() -> Token[list[str | tuple[str, ...]] | None]:
    """Set a fresh empty list into the ``_provenance_accumulator`` contextvar and return the reset token."""
    return _provenance_accumulator.set([])


def get_accumulated_provenance() -> list[str | tuple[str, ...]]:
    """Return a snapshot of all provenance entries collected in the current contextvar scope."""
    accum = _provenance_accumulator.get()
    return list(accum) if accum is not None else []


def reset_provenance_accumulator(token_ref: Token[list[str | tuple[str, ...]] | None]) -> None:
    """Restore ``_provenance_accumulator`` to its previous state via the contextvar reset token."""
    _provenance_accumulator.reset(token_ref)


def append_provenance(item: str | tuple[str, ...]) -> None:
    """Append a provenance step (node name, Shield path, or PII tag) to the active accumulator."""
    accum = _provenance_accumulator.get()
    if accum is not None:
        accum.append(item)


def set_current_access_token(token: ContextToken) -> Token[ContextToken | None]:
    """Set access token in current async context.

    Requires a valid token — callers must resolve the token before setting.
    The return type includes None because ``ContextVar`` default is ``None``.
    """
    return _current_access_token.set(token)


def reset_current_access_token(token_ref: Token[ContextToken | None]) -> None:
    """Restore ``_current_access_token`` to its previous value via the contextvar reset token."""
    _current_access_token.reset(token_ref)


def get_current_access_token() -> ContextToken | None:
    """Return the active ``ContextToken`` from the ``_current_access_token`` contextvar, or ``None`` if unset."""
    return _current_access_token.get()


def require_access_token() -> ContextToken:
    """Get the current access token or raise SecurityError.

    Use this in graph nodes and platform tools where the token
    is guaranteed by secure_node at the execution boundary.
    """
    token = _current_access_token.get()
    if token is None:
        from contextunity.core.exceptions import SecurityError

        raise SecurityError("No access token in execution context")
    return token


__all__ = [
    "set_current_access_token",
    "reset_current_access_token",
    "get_current_access_token",
    "require_access_token",
    "init_provenance_accumulator",
    "get_accumulated_provenance",
    "reset_provenance_accumulator",
    "append_provenance",
]
