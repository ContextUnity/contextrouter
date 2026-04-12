import time
from typing import Any, Dict, List

from langchain_core.callbacks import AsyncCallbackHandler


class TraceCollectorHandler(AsyncCallbackHandler):
    def __init__(self):
        super().__init__()
        self.steps = []
        self._runs = {}

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        self._runs[run_id] = {
            "name": serialized.get("name", "chain"),
            "start": time.time(),
            "type": "chain",
        }

    async def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: str, parent_run_id: str = None, **kwargs: Any
    ) -> None:
        pass

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: str,
        parent_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        self._runs[run_id] = {
            "name": serialized.get("name", "llm"),
            "start": time.time(),
            "type": "llm",
        }

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: str = None,
        **kwargs: Any,
    ) -> None:
        self._runs[run_id] = {
            "name": serialized.get("name", "tool"),
            "start": time.time(),
            "type": "tool",
        }
