"""Runtime context helpers for dispatcher execution."""

from __future__ import annotations

from contextvars import ContextVar, Token

from contextcore import ContextToken

_current_access_token: ContextVar[ContextToken | None] = ContextVar(
    "contextrouter_current_access_token",
    default=None,
)


def set_current_access_token(token: ContextToken | None) -> Token:
    """Set access token in current async context."""
    return _current_access_token.set(token)


def reset_current_access_token(token_ref: Token) -> None:
    """Reset access token context variable."""
    _current_access_token.reset(token_ref)


def get_current_access_token() -> ContextToken | None:
    """Get current access token from async context."""
    return _current_access_token.get()


__all__ = [
    "set_current_access_token",
    "reset_current_access_token",
    "get_current_access_token",
]
