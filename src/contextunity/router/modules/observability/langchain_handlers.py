"""LangChain callback handler mixin for BrainAutoTracer.

Handles standard LangChain lifecycle events:
  - Chain start/end
  - Tool start/end
  - ChatModel start/end
  - LLM end (shared logic with ChatModel)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps
from contextunity.core.types import is_object_dict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

from contextunity.router.modules.observability.contracts import (
    LangchainCallbackMixinHost,
    SpanDict,
    copy_object_dict,
    run_key,
)
from contextunity.router.modules.observability.langchain_usage import (
    extract_generation_text,
    extract_langchain_usage,
)

logger = get_contextunit_logger(__name__)


class LangchainCallbackMixin:
    """Mixin providing standard LangChain AsyncCallbackHandler methods.

    Requires the host class to provide:
      - ``self.root_spans``: list of top-level spans
      - ``self.spans``: dict mapping run_id → span
      - ``self.last_token_usage``: dict for cumulative token tracking
      - ``self.get_active_parent_list(parent_run_id)``: method
    """

    last_token_usage: dict[str, int | float]

    def __init__(self) -> None:
        """Initialize per-tracer token delta tracking."""
        self.last_token_usage = {}

    @staticmethod
    def read_token_delta(outputs: object) -> tuple[int, int, float] | None:
        """Read cumulative token usage from ``outputs`` (or its ``update`` envelope).

        Returns ``(input_tokens, output_tokens, total_cost)`` or ``None`` when
        ``_token_usage`` is absent.
        """
        if not is_object_dict(outputs):
            return None
        candidate: object = outputs.get("_token_usage")
        if not is_object_dict(candidate):
            update_obj: object = outputs.get("update")
            if is_object_dict(update_obj):
                candidate = update_obj.get("_token_usage")
        if not is_object_dict(candidate):
            return None
        in_raw = candidate.get("input_tokens")
        out_raw = candidate.get("output_tokens")
        cost_raw = candidate.get("total_cost")
        in_v = int(in_raw) if isinstance(in_raw, (int, float)) else 0
        out_v = int(out_raw) if isinstance(out_raw, (int, float)) else 0
        cost_v = float(cost_raw) if isinstance(cost_raw, (int, float)) else 0.0
        return in_v, out_v, cost_v

    async def on_chain_start(
        self: LangchainCallbackMixinHost,
        serialized: dict[str, object],
        inputs: dict[str, object],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        """On chain start."""
        # LangGraph nodes pass node names in kwargs["name"], serialized is often None
        name_raw = kwargs.get("name")
        name = (
            str(name_raw)
            if name_raw is not None
            else (serialized.get("name") if serialized else None) or "chain"
        )

        metadata = copy_object_dict(kwargs.get("metadata"))
        if "langgraph_node" in metadata:
            name = str(metadata["langgraph_node"])
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
            self.spans[run_key(run_id)] = {"children": self.root_spans, "ignore": True}
            return

        child_spans: list[SpanDict] = []
        span: SpanDict = {
            "id": run_id,
            "is_group": is_group,
            "node": name,
            "type": step_type,
            "status": "ok",
            "start_time": time.monotonic(),
            "args_json": str(inputs)[:10000] if inputs else "",
            "children": child_spans,
            "cumulative_ms": 0,
            "cumulative_usd": 0.0,
            "cumulative_tokens": 0,
            "timing_ms": 0,
            "has_result": False,
        }
        self.spans[run_key(run_id)] = span
        self.get_active_parent_list(parent_run_id).append(span)

    async def on_chain_end(
        self: LangchainCallbackMixinHost,
        outputs: dict[str, object],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On chain end."""
        _ = kwargs
        span = self.spans.get(run_key(run_id))
        if not span or span.get("ignore"):
            return

        start_time = span.get("start_time")
        if not isinstance(start_time, (int, float)):
            return
        timing = int((time.monotonic() - start_time) * 1000)
        span["timing_ms"] = timing
        cumulative_ms = span.get("cumulative_ms")
        span["cumulative_ms"] = (
            int(cumulative_ms) if isinstance(cumulative_ms, (int, float)) else 0
        ) + timing
        # Safely convert outputs to string
        try:
            span["result_json"] = (
                json_dumps(outputs, default=str, ensure_ascii=False) if outputs else ""
            )
        except Exception:  # graceful-degrade: JSON dump failure must not break tracing
            span["result_json"] = str(outputs)[:10000] if outputs else ""

        span["has_result"] = bool(outputs)

        # Bubble up cumulative values to parent
        parent = self.spans.get(run_key(parent_run_id))
        if parent and not parent.get("ignore"):
            _ = parent.setdefault("cumulative_ms", 0)
            child_cumulative_ms = span.get("cumulative_ms")
            if isinstance(child_cumulative_ms, (int, float)):
                parent_cumulative_ms = parent.get("cumulative_ms")
                parent["cumulative_ms"] = (
                    int(parent_cumulative_ms)
                    if isinstance(parent_cumulative_ms, (int, float))
                    else 0
                ) + int(child_cumulative_ms)
            _ = parent.setdefault("cumulative_usd", 0.0)
            child_cumulative_usd = span.get("cumulative_usd")
            if isinstance(child_cumulative_usd, (int, float)):
                parent_cumulative_usd = parent.get("cumulative_usd")
                parent["cumulative_usd"] = (
                    float(parent_cumulative_usd)
                    if isinstance(parent_cumulative_usd, (int, float))
                    else 0.0
                ) + float(child_cumulative_usd)
            _ = parent.setdefault("cumulative_tokens", 0)
            child_cumulative_tokens = span.get("cumulative_tokens")
            if isinstance(child_cumulative_tokens, (int, float)):
                parent_cumulative_tokens = parent.get("cumulative_tokens")
                parent["cumulative_tokens"] = (
                    int(parent_cumulative_tokens)
                    if isinstance(parent_cumulative_tokens, (int, float))
                    else 0
                ) + int(child_cumulative_tokens)

        # Per-node token-usage delta. ``acc_tokens()`` accumulates across
        # nodes, so the running total grows monotonically. Compare against
        # ``last_token_usage`` to derive this node's contribution and record it
        # on the span. The running total stays on the tracer for the next end.
        token_delta = self.read_token_delta(outputs)
        if token_delta is not None:
            current_in, current_out, current_cost = token_delta
            last = self.last_token_usage
            last_in_raw = last.get("input_tokens")
            last_out_raw = last.get("output_tokens")
            last_cost_raw = last.get("total_cost")
            last_in = int(last_in_raw) if isinstance(last_in_raw, (int, float)) else 0
            last_out = int(last_out_raw) if isinstance(last_out_raw, (int, float)) else 0
            last_cost = float(last_cost_raw) if isinstance(last_cost_raw, (int, float)) else 0.0
            span["tokens_in"] = max(0, current_in - last_in)
            span["tokens_out"] = max(0, current_out - last_out)
            span["cost_usd"] = max(0.0, current_cost - last_cost)
            self.last_token_usage = {
                "input_tokens": current_in,
                "output_tokens": current_out,
                "total_cost": current_cost,
            }

    async def on_tool_start(
        self: LangchainCallbackMixinHost,
        serialized: dict[str, object],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On tool start."""
        _ = kwargs
        name_raw = serialized.get("name", "tool") if serialized else "tool"
        name = str(name_raw)
        span: SpanDict = {
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
        self.spans[run_key(run_id)] = span
        self.get_active_parent_list(parent_run_id).append(span)

    async def on_tool_end(
        self: LangchainCallbackMixinHost,
        output: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On tool end."""
        _ = kwargs
        span = self.spans.get(run_key(run_id))
        if not span:
            return

        start_time = span.get("start_time")
        if not isinstance(start_time, (int, float)):
            return
        timing = int((time.monotonic() - start_time) * 1000)
        span["timing_ms"] = timing
        span["result_json"] = str(output)[:10000]
        span["has_result"] = bool(output)

        # Add to parent cumulative
        parent = self.spans.get(run_key(parent_run_id))
        if parent and not parent.get("ignore"):
            _ = parent.setdefault("cumulative_ms", 0)
            parent_cumulative_ms = parent.get("cumulative_ms")
            parent["cumulative_ms"] = (
                int(parent_cumulative_ms) if isinstance(parent_cumulative_ms, (int, float)) else 0
            ) + timing

    async def on_chat_model_start(
        self: LangchainCallbackMixinHost,
        serialized: dict[str, object],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On chat model start."""
        name_raw = serialized.get("name", "llm") if serialized else "llm"
        name = str(name_raw)

        # Format full messages (including system) for contextunity.view observability
        msgs_fmt: list[str] = []
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

        metadata = copy_object_dict(kwargs.get("metadata"))
        invocation_params = copy_object_dict(kwargs.get("invocation_params"))
        prompt_version = None
        prompt_meta = metadata.get("prompt_version")
        if prompt_meta is not None:
            prompt_version = prompt_meta
        if prompt_version is None:
            prompt_inv = invocation_params.get("prompt_version")
            if prompt_inv is not None:
                prompt_version = prompt_inv
        span: SpanDict = {
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
        if prompt_version is not None:
            span["prompt_version"] = prompt_version
        self.spans[run_key(run_id)] = span
        self.get_active_parent_list(parent_run_id).append(span)

    async def handle_model_end(
        self: LangchainCallbackMixinHost,
        response: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """Shared logic for LLM completion — token extraction, timing, cost.

        Args:
            response: The response payload containing results.
        """
        span = self.spans.get(run_key(run_id))
        if not span:
            return

        start_time = span.get("start_time")
        if not isinstance(start_time, (int, float)):
            return
        timing = int((time.monotonic() - start_time) * 1000)
        span["timing_ms"] = timing
        content = extract_generation_text(response)
        span["result_json"] = content[:10000]
        span["has_result"] = bool(content)

        usage = extract_langchain_usage(response)

        total_tokens = int(usage.get("total_tokens", 0))
        span["tokens"] = total_tokens
        span["tokens_in"] = int(usage.get("prompt_tokens", 0))
        span["tokens_out"] = int(usage.get("completion_tokens", 0))

        # ── Cost estimation for ChatModel span ──
        total_cost = usage.get("total_cost")
        if total_cost is not None:
            span["cost_usd"] = total_cost
        elif total_tokens:
            model_name = span.get("tool", "")
            invocation_params = copy_object_dict(kwargs.get("invocation_params"))
            model_inv = invocation_params.get("model")
            actual_model = (
                str(model_inv) if isinstance(model_inv, str) and model_inv else str(model_name)
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
        parent = self.spans.get(run_key(parent_run_id))
        if parent and not parent.get("ignore"):
            _ = parent.setdefault("cumulative_ms", 0)
            parent_cumulative_ms = parent.get("cumulative_ms")
            parent["cumulative_ms"] = (
                int(parent_cumulative_ms) if isinstance(parent_cumulative_ms, (int, float)) else 0
            ) + timing
            _ = parent.setdefault("cumulative_tokens", 0)
            parent_cumulative_tokens = parent.get("cumulative_tokens")
            span_tokens = span.get("tokens")
            parent["cumulative_tokens"] = (
                int(parent_cumulative_tokens)
                if isinstance(parent_cumulative_tokens, (int, float))
                else 0
            ) + (int(span_tokens) if isinstance(span_tokens, (int, float)) else 0)
            span_cost = span.get("cost_usd")
            if span_cost:
                _ = parent.setdefault("cumulative_usd", 0.0)
                parent_cumulative_usd = parent.get("cumulative_usd")
                parent["cumulative_usd"] = (
                    float(parent_cumulative_usd)
                    if isinstance(parent_cumulative_usd, (int, float))
                    else 0.0
                ) + (float(span_cost) if isinstance(span_cost, (int, float)) else 0.0)

    async def on_chat_model_end(
        self: LangchainCallbackMixinHost,
        response: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On chat model end."""
        await self.handle_model_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def on_llm_end(
        self: LangchainCallbackMixinHost,
        response: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On llm end."""
        await self.handle_model_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
