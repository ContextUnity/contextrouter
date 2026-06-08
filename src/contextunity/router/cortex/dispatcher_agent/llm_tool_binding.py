"""Protocol bridge: router ``BaseModel`` LLMs expose LangChain ``bind_tools`` / ``ainvoke``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.tools import BaseTool


class RunnableToolBinding(Protocol):
    """Return type of ``bind_tools`` — LangChain runnable with async invoke."""

    async def ainvoke(self, input: object, /, **kwargs: object) -> object:
        """Asynchronously invoke the LLM with bound tools.

        Args:
            input: Messages or prompt to send to the LLM.
            **kwargs: Additional LangChain runnable configuration.

        Returns:
            LLM response (typically an ``AIMessage`` with tool calls).
        """


class LLMWithToolBind(Protocol):
    """LLM surface used by dispatcher ``agent_node`` (implemented by LangChain-backed providers)."""

    def bind_tools(self, tools: Sequence[BaseTool], /) -> RunnableToolBinding:
        """Bind a sequence of tools to the LLM for function-calling.

        Args:
            tools: Tools to make available for the LLM's tool-calling API.

        Returns:
            A runnable that supports ``ainvoke`` with the tools bound.
        """
        ...


__all__ = ["LLMWithToolBind", "RunnableToolBinding"]
