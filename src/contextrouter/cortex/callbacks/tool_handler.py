"""LangGraph callback handler for tool events.

This handler tracks tool calls for observability and future AG-UI event mapping.
Tool events are emitted by LangGraph as part of the event stream, and the API
layer maps them to AG-UI events using the AguiMapper.
"""

from __future__ import annotations

import logging

from langchain_core.callbacks import AsyncCallbackHandler

logger = logging.getLogger(__name__)


class ToolEventCallbackHandler(AsyncCallbackHandler):
    """Callback handler for tracking tool calls.

    This handler logs tool calls for observability. The actual AG-UI event
    emission happens in the API layer when processing LangGraph tool events.
    """

    def __init__(self) -> None:
        """Initialize tool event handler."""
        super().__init__()
        self.tool_calls: list[dict[str, object]] = []

    async def on_tool_start(
        self,
        serialized: dict[str, object],
        input_str: str,
        **kwargs: object,
    ) -> None:
        """Handle tool start event.

        Args:
            serialized: Serialized tool definition
            input_str: Tool input string
            **kwargs: Additional keyword arguments
        """
        tool_name = serialized.get("name", "unknown")
        logger.debug("Tool started: %s", tool_name)
        self.tool_calls.append(
            {
                "name": tool_name,
                "input": input_str,
                "status": "started",
            }
        )

    async def on_tool_end(
        self,
        output: str,
        **kwargs: object,
    ) -> None:
        """Handle tool end event.

        Args:
            output: Tool output string
            **kwargs: Additional keyword arguments
        """
        tool_name = kwargs.get("name", "unknown")
        logger.debug("Tool ended: %s", tool_name)
        # Update the last tool call with output
        if self.tool_calls:
            self.tool_calls[-1].update(
                {
                    "output": output,
                    "status": "completed",
                }
            )

    async def on_tool_error(
        self,
        error: Exception,
        **kwargs: object,
    ) -> None:
        """Handle tool error event.

        Args:
            error: Exception that occurred
            **kwargs: Additional keyword arguments
        """
        tool_name = kwargs.get("name", "unknown")
        logger.warning("Tool error: %s - %s", tool_name, error)
        if self.tool_calls:
            self.tool_calls[-1].update(
                {
                    "error": str(error),
                    "status": "error",
                }
            )

    # Chat model callbacks - no-op implementations since this handler only tracks tools
    async def on_chat_model_start(
        self,
        serialized: dict[str, object],
        messages: list[list[object]],
        **kwargs: object,
    ) -> None:
        """No-op: This handler only tracks tool events."""
        pass

    async def on_chat_model_end(self, response: object, **kwargs: object) -> None:
        """No-op: This handler only tracks tool events."""
        pass

    async def on_chat_model_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: object
    ) -> None:
        """No-op: This handler only tracks tool events."""
        pass

    async def on_chat_model_stream(self, chunk: object, **kwargs: object) -> None:
        """No-op: This handler only tracks tool events."""
        pass
