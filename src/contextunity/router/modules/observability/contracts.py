"""Shared contracts for observability tracer mixins."""

from __future__ import annotations

from typing import Protocol, TypeAlias, TypeGuard
from uuid import UUID

from contextunity.core.types import is_object_dict, is_object_list

SpanDict: TypeAlias = dict[str, object]


def copy_object_dict(raw: object) -> dict[str, object]:
    """Copy a runtime object mapping into ``dict[str, object]`` when possible."""
    if is_object_dict(raw):
        return dict(raw)
    return {}


def is_span_dict(value: object) -> TypeGuard[SpanDict]:
    """Narrow callback span payloads; validates known keys when present."""
    if not is_object_dict(value):
        return False
    if not any(key in value for key in ("node", "tool", "type", "ignore", "children")):
        return False
    node = value.get("node")
    if node is not None and not isinstance(node, str):
        return False
    tool = value.get("tool")
    if tool is not None and not isinstance(tool, str):
        return False
    span_type = value.get("type")
    if span_type is not None and not isinstance(span_type, str):
        return False
    ignore = value.get("ignore")
    if ignore is not None and not isinstance(ignore, bool):
        return False
    children_raw = value.get("children")
    if children_raw is not None:
        if not is_object_list(children_raw):
            return False
        if not all(is_span_dict(item) for item in children_raw):
            return False
    return True


def span_children(parent: SpanDict) -> list[SpanDict]:
    """Return (and initialize) the mutable ``children`` list on a span dict."""
    children_raw = parent.get("children")
    if is_object_list(children_raw):
        typed_children = [item for item in children_raw if is_span_dict(item)]
        parent["children"] = typed_children
        return typed_children
    children: list[SpanDict] = []
    parent["children"] = children
    return children


class TracerHost(Protocol):
    """Minimum host interface required by tracer mixins."""

    root_spans: list[SpanDict]
    spans: dict[str, SpanDict]
    active_custom_spans: dict[str, str]
    # Running total of token usage across nodes, used to compute per-node
    # deltas in ``on_chain_end``. Initialized to an empty dict by
    # ``BrainAutoTracer.__init__`` and mutated as nodes finish.
    last_token_usage: dict[str, int | float]

    def get_active_parent_list(self, parent_run_id: UUID | None) -> list[SpanDict]: ...

    def find_span_by_node(self, node_name: str) -> SpanDict | None: ...


class LangchainCallbackMixinHost(TracerHost, Protocol):
    """Host protocol for ``LangchainCallbackMixin`` callback methods."""

    @staticmethod
    def read_token_delta(outputs: object) -> tuple[int, int, float] | None: ...

    async def handle_model_end(
        self,
        response: object,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: object,
    ) -> None: ...


class BrainEventMixinHost(TracerHost, Protocol):
    """Host protocol for ``BrainEventMixin`` custom event handlers."""

    def handle_llm_start(
        self, node_name: str | None, data: dict[str, object], run_id: UUID
    ) -> None: ...

    def handle_llm_end(
        self, node_name: str | None, data: dict[str, object], run_id: UUID
    ) -> None: ...

    def handle_llm_error(
        self, node_name: str | None, data: dict[str, object], run_id: UUID
    ) -> None: ...

    def handle_tool_result(
        self, node_name: str | None, data: dict[str, object], run_id: UUID
    ) -> None: ...


def run_key(run_id: UUID | str | None) -> str:
    """Normalize callback run ids to dict-safe string keys."""
    return str(run_id) if run_id is not None else ""


__all__ = [
    "BrainEventMixinHost",
    "LangchainCallbackMixinHost",
    "SpanDict",
    "copy_object_dict",
    "is_span_dict",
    "span_children",
    "TracerHost",
    "run_key",
]
