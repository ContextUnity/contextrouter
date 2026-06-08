"""Typed bridges between generic platform state and domain-specific states."""

from __future__ import annotations

from langchain_core.runnables.config import RunnableConfig

EMPTY_RUNNABLE_CONFIG: RunnableConfig = {}


__all__ = ["EMPTY_RUNNABLE_CONFIG"]
