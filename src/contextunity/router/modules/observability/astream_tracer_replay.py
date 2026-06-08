"""Replay LangGraph ``astream_events`` v2 payloads into ``BrainAutoTracer``.

``StreamAgent`` consumes ``graph.astream_events`` for SSE progress. Callback
handlers attached to ``RunnableConfig`` are not reliably invoked on that path,
so trace logging would otherwise record ``0 steps``. Replaying stream events
into the tracer keeps Brain traces aligned with graph execution.
"""

from __future__ import annotations

from uuid import UUID

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.messages import BaseMessage

from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.modules.observability.contracts import copy_object_dict

logger = get_contextunit_logger(__name__)


def _as_uuid(value: object) -> UUID | None:
    """Coerce *value* to ``UUID``; return ``None`` for empty / non-coercible input.

    A missing or empty ``run_id`` on a replayed event must not abort the
    surrounding production loop — the event is skipped and the caller logs
    the gap rather than crashing the stream with ``ValueError``.
    """
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None


def _parent_run_id(event: dict[str, object]) -> UUID | None:
    parent_ids = event.get("parent_ids")
    if not is_object_list(parent_ids) or not parent_ids:
        return None
    return _as_uuid(parent_ids[0])


def _event_data(event: dict[str, object]) -> dict[str, object]:
    data = event.get("data")
    return dict(data) if is_object_dict(data) else {}


def _event_error(data: dict[str, object]) -> BaseException:
    """Coerce LangGraph error event payload to an exception for tracer callbacks."""
    err = data.get("error")
    if isinstance(err, BaseException):
        return err
    if err is not None:
        return RuntimeError(str(err))
    return RuntimeError("unknown stream execution error")


def _optional_object_dict(raw: object) -> dict[str, object] | None:
    copied = copy_object_dict(raw)
    return copied or None


def _coerce_message_batches(raw: object) -> list[list[BaseMessage]]:
    if not is_object_list(raw):
        return []
    batches: list[list[BaseMessage]] = []
    for batch in raw:
        if not is_object_list(batch):
            continue
        messages = [item for item in batch if isinstance(item, BaseMessage)]
        if messages:
            batches.append(messages)
    return batches


async def replay_astream_event_to_tracer(
    tracer: BrainAutoTracer,
    event: dict[str, object],
) -> None:
    """Feed one ``astream_events`` v2 event into ``BrainAutoTracer`` callbacks."""
    kind = str(event.get("event", ""))
    if not kind:
        return

    run_id = _as_uuid(event.get("run_id", ""))
    if run_id is None:
        logger.debug("astream replay: skipping event with missing run_id (kind=%s)", kind)
        return
    parent_run_id = _parent_run_id(event)
    tags_raw = event.get("tags")
    tags = [str(tag) for tag in tags_raw] if is_object_list(tags_raw) else None
    metadata = _optional_object_dict(event.get("metadata"))
    name = str(event.get("name", ""))
    data = _event_data(event)

    run_key = str(run_id)

    if kind == "on_chain_start":
        if run_key in tracer.spans:
            return
        inputs = copy_object_dict(data.get("input"))
        await tracer.on_chain_start(
            {},
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            name=name,
            metadata=metadata,
        )
        return

    if kind == "on_chain_end":
        outputs = copy_object_dict(data.get("output"))
        await tracer.on_chain_end(
            outputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        return

    if kind == "on_chain_error":
        await tracer.on_chain_error(
            _event_error(data),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        return

    if kind == "on_chat_model_start":
        if run_key in tracer.spans:
            return
        serialized = copy_object_dict(data.get("serialized"))
        messages: list[list[BaseMessage]] = []
        input_raw = data.get("input")
        if is_object_dict(input_raw):
            messages = _coerce_message_batches(input_raw.get("messages"))
        await tracer.on_chat_model_start(
            serialized,
            messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
        )
        return

    if kind in {"on_chat_model_end", "on_llm_end"}:
        output = data.get("output")
        if output is None:
            return
        await tracer.on_chat_model_end(
            output,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        return

    if kind in {"on_chat_model_error", "on_llm_error"}:
        await tracer.on_chat_model_error(
            _event_error(data),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        return

    if kind == "on_tool_start":
        if run_key in tracer.spans:
            return
        serialized = copy_object_dict(data.get("serialized"))
        input_val = data.get("input")
        input_str = str(input_val)[:5000] if input_val is not None else ""
        await tracer.on_tool_start(
            serialized,
            input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
        )
        return

    if kind == "on_tool_end":
        output = data.get("output")
        if output is None:
            return
        await tracer.on_tool_end(
            output,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        return

    if kind == "on_tool_error":
        await tracer.on_tool_error(
            _event_error(data),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        return

    if kind == "on_custom_event" and name == "brain_event":
        # ``dispatch_custom_event`` attributes brain events to the active node run.
        contextual_run_id = parent_run_id if parent_run_id is not None else run_id
        await tracer.on_custom_event(
            name,
            data,
            run_id=contextual_run_id,
            tags=tags,
            metadata=metadata,
        )


__all__ = ["replay_astream_event_to_tracer"]
