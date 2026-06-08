"""Brain trace callback handler — collects LLM step spans for Brain storage."""

from __future__ import annotations

from langchain_core.callbacks import AsyncCallbackHandler


class BrainTraceCallbackHandler(AsyncCallbackHandler):
    """Langchain callback that collects LLM step spans for Brain trace storage."""

    def __init__(self) -> None:
        """Initialize empty step and span accumulators."""
        super().__init__()
        self.steps: list[object] = []
        self._spans: dict[str, object] = {}

    # Implement callbacks...
