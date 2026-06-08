"""Typed Protocol boundary for optional ``openai`` / ``langfuse.openai`` AsyncOpenAI clients."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias, overload, runtime_checkable

from contextunity.core.types import is_object_list

from ..boundary_common import (
    await_call,
    build_kwargs,
    invoke_openai_chat_create,
    load_sdk_factory,
    openai_choice_text,
    openai_finish_reason,
    openai_stream_delta_text,
)

OpenAIMessage: TypeAlias = dict[str, object]

# Backward-compatible alias used by OpenAI chat completion paths.
build_chat_kwargs = build_kwargs


class _OpenAIChatMessage(Protocol):
    content: str | None


class _OpenAIChatChoice(Protocol):
    message: _OpenAIChatMessage


class _OpenAIChatCompletion(Protocol):
    choices: list[_OpenAIChatChoice]
    usage: object | None


class _OpenAIChatDelta(Protocol):
    content: str | None


class _OpenAIChatStreamChoice(Protocol):
    delta: _OpenAIChatDelta


class _OpenAIChatStreamChunk(Protocol):
    choices: list[_OpenAIChatStreamChoice]


class _OpenAIChatCompletions(Protocol):
    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        **kwargs: object,
    ) -> _OpenAIChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        stream: Literal[True],
        **kwargs: object,
    ) -> AsyncIterator[_OpenAIChatStreamChunk]: ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        **kwargs: object,
    ) -> _OpenAIChatCompletion | AsyncIterator[_OpenAIChatStreamChunk]: ...


class _OpenAIChat(Protocol):
    @property
    def completions(self) -> _OpenAIChatCompletions: ...


@runtime_checkable
class OpenAIAsyncClient(Protocol):
    @property
    def chat(self) -> _OpenAIChat: ...


def load_async_openai_client(**kwargs: object) -> object:
    """Load ``AsyncOpenAI`` from Langfuse wrapper, falling back to ``openai``."""
    return load_sdk_factory(
        module_names=("langfuse.openai", "openai"),
        factory_name="AsyncOpenAI",
        error_message=(
            "OpenAI-compatible client requires `openai` package. Install with `pip install openai`."
        ),
        **kwargs,
    )


def openai_chat_content(choice: _OpenAIChatChoice) -> str:
    """Extract string content from a chat completion choice."""
    return openai_choice_text(choice)


def openai_stream_delta(chunk: _OpenAIChatStreamChunk) -> str | None:
    """Extract incremental text from a streaming chunk, if present."""
    return openai_stream_delta_text(chunk)


def responses_output_text(response: object) -> str:
    """Extract assistant text from an OpenAI Responses API payload."""
    output_obj: object = getattr(response, "output", None)
    if not is_object_list(output_obj):
        return ""
    for item in output_obj:
        item_type: object = getattr(item, "type", None)
        if item_type != "message":
            continue
        item_content: object = getattr(item, "content", None)
        if not is_object_list(item_content):
            continue
        for content in item_content:
            content_type: object = getattr(content, "type", None)
            if content_type != "output_text":
                continue
            text_value: object = getattr(content, "text", "")
            return text_value if isinstance(text_value, str) else str(text_value)
    return ""


def choice_finish_reason(choice: object) -> str | None:
    """Read ``finish_reason`` from a chat completion choice object."""
    return openai_finish_reason(choice)


__all__ = [
    "OpenAIAsyncClient",
    "OpenAIMessage",
    "await_call",
    "build_chat_kwargs",
    "build_kwargs",
    "choice_finish_reason",
    "invoke_openai_chat_create",
    "load_async_openai_client",
    "openai_chat_content",
    "openai_stream_delta",
    "responses_output_text",
]
