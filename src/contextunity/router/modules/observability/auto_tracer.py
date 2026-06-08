"""Universal automatic LangGraph tracer for contextunity.brain.
This AsyncCallbackHandler builds a hierarchical ''steps'' tree compatible
with the contextunity.view ``nested_steps`` dashboard architecture.
Tracing sources (provider-agnostic):
  • **Tools** — captured via LangChain on_tool_start/end (tools ARE BaseTool).
  • **Token usage** — extracted from node output ``_token_usage`` in
    on_chain_end, populated by ``invoke_model() → acc_tokens()``.
    Works for ANY LLM provider (OpenAI, Anthropic, Vertex, Ollama, etc.).
  • **ChatModel spans** — bonus detail when the provider uses a LangChain
    ChatModel internally.  Not required for correct token accounting.
  • **Custom BrainEvents** — captured via on_custom_event from graph node
    executors (LLM, Federated, Platform) that use the platform model
    registry instead of LangChain ChatModel directly.
"""

from __future__ import annotations

from typing import override
from uuid import UUID

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_list
from langchain_core.callbacks import AsyncCallbackHandler

from contextunity.router.modules.observability.brain_event_handlers import BrainEventMixin
from contextunity.router.modules.observability.contracts import (
    SpanDict,
    is_span_dict,
    span_children,
)
from contextunity.router.modules.observability.langchain_handlers import LangchainCallbackMixin

logger = get_contextunit_logger(__name__)


class BrainAutoTracer(LangchainCallbackMixin, BrainEventMixin, AsyncCallbackHandler):
    """Automatically records all Langchain/LangGraph execution steps into a hierarchical
    tree structure that maps perfectly to contextunity.view's Graph Journey."""

    def __init__(self) -> None:
        """Initialize the hierarchical span tree and per-node token-usage delta tracker."""
        super().__init__()
        self.root_spans: list[SpanDict] = []  # Top-level spans (usually the outer graph)
        self.spans: dict[str, SpanDict] = {}  # Map run_id -> span dictionary
        # Track cumulative _token_usage from graph state to compute per-node deltas.
        # acc_tokens() accumulates across nodes, so each node's output has the
        # running total — we need the DELTA for per-node display.
        self.last_token_usage: dict[str, int | float] = {}
        # Track active custom event spans (llm_start without llm_end yet).
        # Maps node_name -> span_id for in-flight LLM calls.
        self.active_custom_spans: dict[str, str] = {}

    def get_nested_steps(self) -> list[SpanDict]:
        """Return the fully built hierarchical steps tree."""
        return self.root_spans

    def get_tool_calls_summary(self) -> list[dict[str, str]]:
        """Return a flat list of executed tool calls for summary."""

        summary: list[dict[str, str]] = []

        def walk(spans: list[SpanDict], acc: list[dict[str, str]]) -> None:
            """Walk."""
            for s in spans:
                node_raw = s.get("node", "")
                node = node_raw if isinstance(node_raw, str) else ""
                if s.get("type") == "tool_call" and not node.startswith("__"):
                    tool_raw = s.get("tool", "unknown")
                    status_raw = s.get("status", "ok")
                    acc.append(
                        {
                            "tool": tool_raw if isinstance(tool_raw, str) else "unknown",
                            "status": status_raw if isinstance(status_raw, str) else "ok",
                        }
                    )
                children_raw = s.get("children")
                if is_object_list(children_raw):
                    child_spans: list[SpanDict] = []
                    for child in children_raw:
                        if is_span_dict(child):
                            child_spans.append(child)
                    walk(child_spans, acc)

        walk(self.root_spans, summary)
        return summary

    def get_token_usage(self) -> dict[str, float]:
        """Aggregate token usage and cost across all llm spans."""
        input_tokens = 0
        output_tokens = 0
        total_cost = 0.0

        def walk(spans: list[SpanDict]) -> None:
            """Walk."""
            nonlocal input_tokens, output_tokens, total_cost
            for s in spans:
                tokens_in_raw = s.get("tokens_in")
                if isinstance(tokens_in_raw, (int, float)):
                    input_tokens += int(tokens_in_raw)
                tokens_out_raw = s.get("tokens_out")
                if isinstance(tokens_out_raw, (int, float)):
                    output_tokens += int(tokens_out_raw)
                cost_raw = s.get("cost_usd")
                if isinstance(cost_raw, (int, float)):
                    total_cost += float(cost_raw)
                children_raw = s.get("children")
                if is_object_list(children_raw):
                    child_spans: list[SpanDict] = []
                    for child in children_raw:
                        if is_span_dict(child):
                            child_spans.append(child)
                    walk(child_spans)

        walk(self.root_spans)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": round(total_cost, 8),
        }

    def get_active_parent_list(self, parent_run_id: UUID | None) -> list[SpanDict]:
        """Resolve the active parent span's children list for nesting."""
        parent_key = str(parent_run_id) if parent_run_id is not None else ""
        if not parent_key or parent_key not in self.spans:
            return self.root_spans
        parent_span = self.spans[parent_key]
        if parent_span.get("ignore"):
            # LangGraph root span aliases ``children`` to ``root_spans``; never
            # reassign via ``span_children`` or nested steps stay empty.
            return self.root_spans
        return span_children(parent_span)

    def find_span_by_node(self, node_name: str) -> SpanDict | None:
        """Find an active group span by its node name."""
        for span in reversed(list(self.spans.values())):
            if span.get("node") == node_name and span.get("is_group") and not span.get("ignore"):
                return span
        return None

    # ── Error handlers ──────────────────────────────────────────────

    @override
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On tool error."""
        _ = kwargs
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else ""
        span = self.spans.get(run_key)
        if span:
            span["status"] = "error"
            span["result_json"] = str(error)
            span["has_result"] = True
            parent = self.spans.get(parent_key)
            if parent and not parent.get("ignore"):
                parent["status"] = "error"

    @override
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On chain error."""
        _ = kwargs
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else ""
        span = self.spans.get(run_key)
        if span and not span.get("ignore"):
            span["status"] = "error"
            parent = self.spans.get(parent_key)
            if parent and not parent.get("ignore"):
                parent["status"] = "error"

    async def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None:
        """On chat model error."""
        _ = kwargs
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else ""
        span = self.spans.get(run_key)
        if span:
            span["status"] = "error"
            span["result_json"] = str(error)
            span["has_result"] = True
            parent = self.spans.get(parent_key)
            if parent and not parent.get("ignore"):
                parent["status"] = "error"
