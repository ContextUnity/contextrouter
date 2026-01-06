"""Processors for ingestion pipeline."""

from __future__ import annotations

from typing import Any


def generate_persona_profile(*args: Any, **kwargs: Any) -> Any:
    """Lazy import wrapper to avoid circular imports at module import time."""
    from .style import generate_persona_profile as _impl

    return _impl(*args, **kwargs)


__all__ = ["generate_persona_profile"]
