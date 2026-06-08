"""Typed Protocol boundary for LangChain runnable tool-binding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeGuard, runtime_checkable


@runtime_checkable
class LangchainToolBinder(Protocol):
    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> object:
        """Return a runnable with tools bound."""
        ...

    def with_fallbacks(self, fallbacks: Sequence[object]) -> object:
        """Return a runnable with fallback models chained."""
        ...


def is_langchain_tool_binder(model: object) -> TypeGuard[LangchainToolBinder]:
    """Return True when *model* exposes LangChain tool-binding methods."""
    bind_tools = getattr(model, "bind_tools", None)
    with_fallbacks = getattr(model, "with_fallbacks", None)
    return callable(bind_tools) and callable(with_fallbacks)


def langchain_tool_binder(model: object) -> LangchainToolBinder | None:
    """Return *model* when it implements LangChain tool binding."""
    if is_langchain_tool_binder(model):
        return model
    return None


__all__ = ["LangchainToolBinder", "langchain_tool_binder"]
