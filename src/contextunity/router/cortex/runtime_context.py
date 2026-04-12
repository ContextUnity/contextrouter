"""Runtime context helpers for dispatcher execution."""

from __future__ import annotations

from contextvars import ContextVar, Token

from contextunity.core import ContextToken

_current_access_token: ContextVar[ContextToken | None] = ContextVar(
    "contextunity.router_current_access_token",
    default=None,
)

_provenance_accumulator: ContextVar[list[tuple[str, ...]] | list[str] | None] = ContextVar(
    "contextunity.router_provenance_accum",
    default=None,
)


def init_provenance_accumulator() -> Token:
    """Initialize a new provenance accumulator for the current execution context."""
    return _provenance_accumulator.set([])


def get_accumulated_provenance() -> list[tuple[str, ...]] | list[str]:
    """Get all collected provenance strings/paths from the accumulator."""
    accum = _provenance_accumulator.get()
    return list(accum) if accum is not None else []


def reset_provenance_accumulator(token_ref: Token) -> None:
    """Reset the provenance accumulator."""
    _provenance_accumulator.reset(token_ref)


def append_provenance(item: str | tuple[str, ...]) -> None:
    """Explicitly append an item/path to the provenance accumulator."""
    accum = _provenance_accumulator.get()
    if accum is not None:
        accum.append(item)


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
    "init_provenance_accumulator",
    "get_accumulated_provenance",
    "reset_provenance_accumulator",
    "append_provenance",
]
