"""BrainEvent custom event handler mixin for BrainAutoTracer.

Handles platform model registry LLM calls that bypass standard LangChain
callbacks. These come through as ``dispatch_custom_event("brain_event", ...)``.

Event types:
  - ``llm_start`` — Create child LLM span
  - ``llm_end`` — Close LLM span with tokens/cost
  - ``llm_error`` — Mark span as errored
  - ``tool_result`` — Create completed tool span from federated/platform calls
"""

from __future__ import annotations

import time
import uuid
from uuid import UUID

from contextunity.core import get_contextunit_logger

from contextunity.router.modules.observability.brain_event_parse import parse_brain_custom_event
from contextunity.router.modules.observability.contracts import (
    BrainEventMixinHost,
    SpanDict,
    run_key,
    span_children,
)

logger = get_contextunit_logger(__name__)


class BrainEventMixin:
    """Mixin providing BrainEvent custom event handling.

    Requires the host class to provide:
      - ``self.root_spans``: list of top-level spans
      - ``self.spans``: dict mapping run_id → span
      - ``self.active_custom_spans``: dict mapping node_name → span_id
      - ``self.find_span_by_node(node_name)``: method
    """

    async def on_custom_event(
        self: BrainEventMixinHost,
        name: str,
        data: object,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Handle BrainEvent custom events from graph node executors."""
        _ = tags, metadata, kwargs
        if name != "brain_event":
            return

        parsed = parse_brain_custom_event(data)
        if parsed is None:
            return

        event_type, node_name_str, payload = parsed

        if event_type == "llm_start":
            self.handle_llm_start(node_name_str, payload, run_id)
        elif event_type == "llm_end":
            self.handle_llm_end(node_name_str, payload, run_id)
        elif event_type == "llm_error":
            self.handle_llm_error(node_name_str, payload, run_id)
        elif event_type == "tool_result":
            self.handle_tool_result(node_name_str, payload, run_id)

    def handle_llm_start(
        self: BrainEventMixinHost,
        node_name: str | None,
        data: dict[str, object],
        run_id: UUID,
    ) -> None:
        """Create a child LLM span under the active node group."""
        model_val = data.get("model")
        model = str(model_val) if isinstance(model_val, str) else "llm"
        span_id = str(uuid.uuid4())

        span: SpanDict = {
            "id": span_id,
            "is_group": False,
            "tool": model,
            "type": "assistant",
            "status": "ok",
            "start_time": time.monotonic(),
            "args_json": str(data.get("args")) if data.get("args") is not None else "",
            "result_json": "",
            "tokens": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "timing_ms": 0,
        }
        prompt_ver = data.get("prompt_version")
        if prompt_ver is not None:
            span["prompt_version"] = str(prompt_ver)
        self.spans[span_id] = span

        # Map to parent: run_id from dispatch_custom_event is the node's chain run_id
        parent = self.spans.get(run_key(run_id))
        if parent and not parent.get("ignore"):
            span_children(parent).append(span)
        else:
            # Fallback: find parent by node name
            parent = self.find_span_by_node(node_name) if node_name else None
            if parent:
                span_children(parent).append(span)
            else:
                self.root_spans.append(span)

        # Track active span for this node so llm_end can find it
        if node_name:
            self.active_custom_spans[node_name] = span_id

    def handle_llm_end(
        self: BrainEventMixinHost,
        node_name: str | None,
        data: dict[str, object],
        run_id: UUID,
    ) -> None:
        """Close the active LLM span with timing, tokens, and cost."""
        span_id = self.active_custom_spans.pop(node_name, None) if node_name else None
        if not span_id:
            return

        span = self.spans.get(span_id)
        if not span:
            return

        start_time = span.get("start_time")
        if not isinstance(start_time, (int, float)):
            return
        timing = int((time.monotonic() - start_time) * 1000)
        span["timing_ms"] = timing
        span["has_result"] = True

        if "result" in data:
            span["result_json"] = str(data["result"])

        _Numeric = (int, float, str)
        tokens_in = int(v) if isinstance(v := data.get("input_tokens"), _Numeric) else 0
        tokens_out = int(v) if isinstance(v := data.get("output_tokens"), _Numeric) else 0
        total_cost = float(v) if isinstance(v := data.get("total_cost"), _Numeric) else 0.0
        total_tokens = tokens_in + tokens_out

        span["tokens"] = total_tokens
        span["tokens_in"] = tokens_in
        span["tokens_out"] = tokens_out

        if total_cost:
            span["cost_usd"] = total_cost
        elif total_tokens:
            model_val = data.get("model")
            tool_val = span.get("tool", "")
            model_name = (
                str(model_val) if isinstance(model_val, str) else str(tool_val) if tool_val else ""
            )
            if model_name:
                try:
                    from contextunity.router.modules.models.types import UsageStats

                    est = UsageStats(
                        input_tokens=tokens_in,
                        output_tokens=tokens_out,
                        total_tokens=total_tokens,
                    ).estimate_cost(model_name)
                    if est.total_cost:
                        span["cost_usd"] = est.total_cost
                except Exception:
                    pass

        # Bubble up to parent node group
        parent = self.spans.get(run_key(run_id))
        if parent and not parent.get("ignore"):
            _ = parent.setdefault("cumulative_tokens", 0)
            parent_cumulative_tokens = parent.get("cumulative_tokens")
            parent["cumulative_tokens"] = (
                int(parent_cumulative_tokens)
                if isinstance(parent_cumulative_tokens, (int, float))
                else 0
            ) + total_tokens
            span_cost = span.get("cost_usd")
            if span_cost:
                _ = parent.setdefault("cumulative_usd", 0.0)
                parent_cumulative_usd = parent.get("cumulative_usd")
                parent["cumulative_usd"] = (
                    float(parent_cumulative_usd)
                    if isinstance(parent_cumulative_usd, (int, float))
                    else 0.0
                ) + (float(span_cost) if isinstance(span_cost, (int, float)) else 0.0)

    def handle_llm_error(
        self: BrainEventMixinHost,
        node_name: str | None,
        data: dict[str, object],
        run_id: UUID,
    ) -> None:
        """Mark the active LLM span as errored."""
        _ = run_id
        span_id = self.active_custom_spans.pop(node_name, None) if node_name else None
        if not span_id:
            return

        span = self.spans.get(span_id)
        if not span:
            return

        start_time = span.get("start_time")
        if not isinstance(start_time, (int, float)):
            return
        timing = int((time.monotonic() - start_time) * 1000)
        span["timing_ms"] = timing
        span["status"] = "error"
        span["result_json"] = (
            str(data.get("error")) if data.get("error") is not None else "LLM error"
        )
        span["has_result"] = True

    def handle_tool_result(
        self: BrainEventMixinHost,
        node_name: str | None,
        data: dict[str, object],
        run_id: UUID,
    ) -> None:
        """Create a completed tool span from a federated/platform tool_result event."""
        tool_binding_val = data.get("tool_binding")
        tool_binding = (
            str(tool_binding_val) if isinstance(tool_binding_val, str) else (node_name or "tool")
        )
        duration_val = data.get("duration_ms")
        duration_ms = int(duration_val) if isinstance(duration_val, (int, float, str)) else 0
        status_val = data.get("status")
        status = str(status_val) if isinstance(status_val, str) else "ok"
        error = data.get("error")
        span_id = str(uuid.uuid4())

        span: SpanDict = {
            "id": span_id,
            "is_group": False,
            "tool": tool_binding,
            "type": "tool_call",
            "status": status,
            "start_time": time.monotonic(),
            "args_json": str(data.get("args", "")) if data.get("args") is not None else "",
            "result_json": str(error)
            if error
            else (str(data.get("result", "")) if data.get("result") is not None else ""),
            "tokens": 0,
            "cost_usd": 0.0,
            "timing_ms": duration_ms,
            "has_result": True,
        }
        self.spans[span_id] = span

        # Map to parent node group
        parent = self.spans.get(run_key(run_id))
        if parent and not parent.get("ignore"):
            span_children(parent).append(span)
        else:
            parent = self.find_span_by_node(node_name) if node_name else None
            if parent:
                span_children(parent).append(span)
            else:
                self.root_spans.append(span)

        # Bubble up timing
        parent = self.spans.get(run_key(run_id))
        if parent and not parent.get("ignore"):
            _ = parent.setdefault("cumulative_ms", 0)
            parent_cumulative_ms = parent.get("cumulative_ms")
            parent["cumulative_ms"] = (
                int(parent_cumulative_ms) if isinstance(parent_cumulative_ms, (int, float)) else 0
            ) + duration_ms
