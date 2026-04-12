"""Universal automatic LangGraph tracer for cu.brain.

This AsyncCallbackHandler builds a hierarchical ''steps'' tree compatible
with the cu.view ``nested_steps`` dashboard architecture.

Tracing sources (provider-agnostic):
  • **Tools** — captured via LangChain on_tool_start/end (tools ARE BaseTool).
  • **Token usage** — extracted from node output ``_token_usage`` in
    on_chain_end, populated by ``invoke_model() → acc_tokens()``.
    Works for ANY LLM provider (OpenAI, Anthropic, Vertex, Ollama, etc.).
  • **ChatModel spans** — bonus detail when the provider uses a LangChain
    ChatModel internally.  Not required for correct token accounting.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from contextunity.router.modules.tools.schemas import ToolCallSummary, TraceStep


class BrainAutoTracer(AsyncCallbackHandler):
    """Automatically records all Langchain/LangGraph execution steps into a hierarchical
    tree structure that maps perfectly to cu.view's Graph Journey."""

    def __init__(self):
        super().__init__()
        self.root_spans = []  # Top-level spans (usually the outer graph)
        self._spans = {}  # Map run_id -> span dictionary
        # Track cumulative _token_usage from graph state to compute per-node deltas.
        # acc_tokens() accumulates across nodes, so each node's output has the
        # running total — we need the DELTA for per-node display.
        self._last_token_usage: dict[str, int | float] = {}

    def get_nested_steps(self) -> list[TraceStep]:
        """Return the fully built hierarchical steps tree."""
        return self.root_spans

    def get_tool_calls_summary(self) -> list[ToolCallSummary]:
        """Return a flat list of executed tools for high-level summary."""

        def walk(spans, acc):
            for s in spans:
                if s.get("type") == "tool_call" and not s.get("node", "").startswith("__"):
                    acc.append({"tool": s.get("tool", "unknown"), "status": s.get("status", "ok")})
                if s.get("children"):
                    walk(s["children"], acc)

        summary = []
        walk(self.root_spans, summary)
        return summary

    def get_token_usage(self) -> dict[str, float]:
        """Aggregate token usage and cost across all LLM spans."""
        input_tokens = 0
        output_tokens = 0
        total_cost = 0.0

        def walk(spans):
            nonlocal input_tokens, output_tokens, total_cost
            for s in spans:
                if "tokens_in" in s:
                    input_tokens += s["tokens_in"]
                if "tokens_out" in s:
                    output_tokens += s["tokens_out"]
                if "cost_usd" in s:
                    total_cost += s["cost_usd"]
                if s.get("children"):
                    walk(s["children"])

        walk(self.root_spans)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": round(total_cost, 8),
        }

    def _get_active_parent_list(self, parent_run_id: str | None) -> list:
        if not parent_run_id or parent_run_id not in self._spans:
            return self.root_spans
        return self._spans[parent_run_id].setdefault("children", [])

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # LangGraph nodes pass node names in kwargs["name"], serialized is often None
        name = kwargs.get("name") or (serialized.get("name") if serialized else None) or "chain"

        metadata = kwargs.get("metadata") or getattr(self, "metadata", {}) or {}
        if "langgraph_node" in metadata:
            name = metadata["langgraph_node"]
        elif tags:
            for tag in tags:
                if tag.startswith("graph:node:"):
                    name = tag.split(":")[-1]
                    break

        # We model LangGraph nodes as our visualizer "group" elements
        is_group = True
        step_type = "chain"

        # Small hack: if it's the top level LangGraph, skip creating a ui block
        # but keep track of it so children map to root.
        if name == "LangGraph":
            self._spans[run_id] = {"children": self.root_spans, "ignore": True}
            return

        span = {
            "id": run_id,
            "is_group": is_group,
            "node": name,
            "type": step_type,
            "status": "ok",
            "start_time": time.monotonic(),
            "args_json": str(inputs)[:10000] if inputs else "",
            "children": [],
            "cumulative_ms": 0,
            "cumulative_usd": 0.0,
            "cumulative_tokens": 0,
            "timing_ms": 0,
            "has_result": False,
        }
        self._spans[run_id] = span
        self._get_active_parent_list(parent_run_id).append(span)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.get(run_id)
        if not span or span.get("ignore"):
            return

        timing = int((time.monotonic() - span["start_time"]) * 1000)
        span["timing_ms"] = timing
        span["cumulative_ms"] += timing
        # Safely convert outputs to string
        try:
            span["result_json"] = (
                json.dumps(outputs, default=str, ensure_ascii=False) if outputs else ""
            )
        except Exception:
            span["result_json"] = str(outputs)[:10000] if outputs else ""

        span["has_result"] = bool(outputs)

        # Bubble up cumulative values to parent
        parent = self._spans.get(parent_run_id)
        if parent and not parent.get("ignore"):
            parent.setdefault("cumulative_ms", 0)
            parent["cumulative_ms"] += span.get("cumulative_ms", 0)
            parent.setdefault("cumulative_usd", 0.0)
            parent["cumulative_usd"] += span.get("cumulative_usd", 0.0)
            parent.setdefault("cumulative_tokens", 0)
            parent["cumulative_tokens"] += span.get("cumulative_tokens", 0)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool") if serialized else "tool"
        span = {
            "id": run_id,
            "is_group": False,
            "tool": name,
            "type": "tool_call",
            "status": "ok",
            "start_time": time.monotonic(),
            "args_json": str(input_str)[:5000],
            "result_json": "",
            "tokens": 0,
            "cost_usd": 0.0,
            "timing_ms": 0,
        }
        self._spans[run_id] = span
        self._get_active_parent_list(parent_run_id).append(span)

    async def on_tool_end(
        self, output: Any, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        span = self._spans.get(run_id)
        if not span:
            return

        timing = int((time.monotonic() - span["start_time"]) * 1000)
        span["timing_ms"] = timing
        span["result_json"] = str(output)[:10000]
        span["has_result"] = bool(output)

        # Add to parent cumulative
        parent = self._spans.get(parent_run_id)
        if parent and not parent.get("ignore"):
            parent.setdefault("cumulative_ms", 0)
            parent["cumulative_ms"] += timing

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "llm") if serialized else "llm"

        # Format full messages (including system) for cu.view observability
        msgs_fmt = []
        if messages:
            for msg_list in messages:
                for msg in msg_list:
                    role = (
                        getattr(msg, "type", "")
                        if hasattr(msg, "type")
                        else str(type(msg).__name__)
                    )
                    content = getattr(msg, "content", str(msg))
                    msgs_fmt.append(f"[{role.upper()}]\n{str(content)}")

        in_str = "\n\n".join(msgs_fmt)
        if len(in_str) > 15000:
            in_str = in_str[:15000] + "\n...[TRUNCATED]"

        span = {
            "id": run_id,
            "is_group": False,
            "tool": name,
            "type": "assistant",
            "status": "ok",
            "start_time": time.monotonic(),
            "args_json": in_str,
            "result_json": "",
            "tokens": 0,
            "cost_usd": 0.0,
            "timing_ms": 0,
        }
        self._spans[run_id] = span
        self._get_active_parent_list(parent_run_id).append(span)

    async def _handle_model_end(
        self, response: Any, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        """Shared logic for LLM completion — token extraction, timing, cost.

        Called by both on_chat_model_end and on_llm_end since LangChain
        dispatches to different handler methods depending on how the run
        was started (on_chat_model_start vs on_llm_start) while the
        response format (LLMResult) is identical.
        """
        span = self._spans.get(run_id)
        if not span:
            return

        timing = int((time.monotonic() - span["start_time"]) * 1000)
        span["timing_ms"] = timing

        content = ""
        if hasattr(response, "generations") and response.generations:
            content = (
                str(response.generations[0][0].text)
                if hasattr(response.generations[0][0], "text")
                else str(response.generations[0][0])
            )

        span["result_json"] = content[:10000]
        span["has_result"] = bool(content)

        # ── Token extraction: try all known LangChain formats ──
        usage: dict = {}

        # Path 1: llm_output.token_usage (classic LangChain)
        llm_output = getattr(response, "llm_output", {}) or {}
        token_usage = llm_output.get("token_usage", {})
        if token_usage:
            usage = token_usage

        # Path 2: AIMessage.usage_metadata (LangChain ≥0.2)
        # May be a dict OR a UsageMetadata dataclass — handle both
        if not usage and hasattr(response, "generations") and response.generations:
            msg = getattr(response.generations[0][0], "message", None)
            if msg:
                m = getattr(msg, "usage_metadata", None)
                if m:
                    # UsageMetadata object has .input_tokens, .output_tokens attrs
                    inp = getattr(m, "input_tokens", None) or (
                        m.get("input_tokens") if isinstance(m, dict) else None
                    )
                    out = getattr(m, "output_tokens", None) or (
                        m.get("output_tokens") if isinstance(m, dict) else None
                    )
                    tot = getattr(m, "total_tokens", None) or (
                        m.get("total_tokens") if isinstance(m, dict) else None
                    )
                    if inp is not None or out is not None:
                        usage = {
                            "prompt_tokens": int(inp or 0),
                            "completion_tokens": int(out or 0),
                            "total_tokens": int(tot or 0) or int(inp or 0) + int(out or 0),
                        }

        # Path 3: AIMessage.response_metadata.token_usage (langchain-openai >=0.1.8)
        # Also check for direct total_cost in response_metadata (our custom invoke_model output)
        if not usage and hasattr(response, "generations") and response.generations:
            msg = getattr(response.generations[0][0], "message", None)
            if msg:
                rm = getattr(msg, "response_metadata", {}) or {}
                if isinstance(rm, dict):
                    tu = rm.get("token_usage", {})
                    if tu:
                        usage = tu
                    if "total_cost" in rm:
                        usage["total_cost"] = rm["total_cost"]
        elif hasattr(response, "generations") and response.generations:
            # We already have usage, but let's see if total_cost is in rm
            msg = getattr(response.generations[0][0], "message", None)
            if msg:
                rm = getattr(msg, "response_metadata", {}) or {}
                if isinstance(rm, dict) and "total_cost" in rm:
                    usage["total_cost"] = rm["total_cost"]

        total_tokens = usage.get("total_tokens", 0)
        span["tokens"] = total_tokens
        span["tokens_in"] = usage.get("prompt_tokens", 0)
        span["tokens_out"] = usage.get("completion_tokens", 0)

        # ── Cost estimation for ChatModel span ──
        # Use provided cost if available
        if usage.get("total_cost"):
            span["cost_usd"] = usage["total_cost"]
        elif total_tokens:
            model_name = span.get("tool", "")
            # Try to get model from llm_output or invocation_params
            actual_model = (
                llm_output.get("model_name", "")
                or kwargs.get("invocation_params", {}).get("model", "")
                or model_name
            )
            if actual_model:
                from contextunity.router.modules.models.types import UsageStats

                est = UsageStats(
                    input_tokens=span["tokens_in"],
                    output_tokens=span["tokens_out"],
                    total_tokens=total_tokens,
                ).estimate_cost(actual_model)
                if est.total_cost:
                    span["cost_usd"] = est.total_cost

        # Bubble up tracking values to parent
        parent = self._spans.get(parent_run_id)
        if parent and not parent.get("ignore"):
            parent.setdefault("cumulative_ms", 0)
            parent["cumulative_ms"] += timing
            parent.setdefault("cumulative_tokens", 0)
            parent["cumulative_tokens"] += span.get("tokens", 0)
            if span.get("cost_usd"):
                parent.setdefault("cumulative_usd", 0.0)
                parent["cumulative_usd"] += span["cost_usd"]

    async def on_chat_model_end(
        self, response: Any, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        await self._handle_model_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_llm_end(
        self, response: Any, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        await self._handle_model_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_tool_error(
        self, error: Exception, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        span = self._spans.get(run_id)
        if span:
            span["status"] = "error"
            span["result_json"] = str(error)
            span["has_result"] = True
            parent = self._spans.get(parent_run_id)
            if parent and not parent.get("ignore"):
                parent["status"] = "error"

    async def on_chain_error(
        self, error: Exception, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        span = self._spans.get(run_id)
        if span and not span.get("ignore"):
            span["status"] = "error"
            parent = self._spans.get(parent_run_id)
            if parent and not parent.get("ignore"):
                parent["status"] = "error"

    async def on_chat_model_error(
        self, error: Exception, *, run_id: str, parent_run_id: str | None = None, **kwargs: Any
    ) -> None:
        span = self._spans.get(run_id)
        if span:
            span["status"] = "error"
            span["result_json"] = str(error)
            span["has_result"] = True
            parent = self._spans.get(parent_run_id)
            if parent and not parent.get("ignore"):
                parent["status"] = "error"
