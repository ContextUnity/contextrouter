"""Shared helpers for optional third-party SDK boundary modules."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from importlib import import_module
from typing import Protocol, runtime_checkable

from contextunity.core.exceptions import ConfigurationError
from contextunity.core.narrowing import await_object
from contextunity.core.types import is_object_dict, is_object_list


def build_kwargs(**kwargs: object) -> dict[str, object]:
    """Drop ``None`` values so SDK optional params are omitted, not sent as null."""
    return {k: v for k, v in kwargs.items() if v is not None}


async def await_call(fn: object, /, *args: object, **kwargs: object) -> object:
    """Invoke *fn* and await the result when it returns an awaitable."""
    if not callable(fn):
        raise ConfigurationError("SDK call target is not callable")
    result: object = fn(*args, **build_kwargs(**kwargs))
    return await await_object(result)


async def await_method(
    target: object,
    name: str,
    /,
    *args: object,
    **kwargs: object,
) -> object:
    """Call ``getattr(target, name)(*args, **kwargs)`` through :func:`await_call`."""
    fn: object = getattr(target, name, None)
    if fn is None:
        raise ConfigurationError(f"{name!r} is unavailable on SDK client")
    return await await_call(fn, *args, **kwargs)


def load_sdk_factory(
    *,
    module_names: tuple[str, ...],
    factory_name: str,
    error_message: str,
    **kwargs: object,
) -> object:
    """Import the first available module and construct ``factory_name(**kwargs)``."""
    factory_obj = load_sdk_attribute(
        module_names=module_names,
        attribute_name=factory_name,
        error_message=error_message,
    )
    if not callable(factory_obj):
        raise ConfigurationError(error_message)
    return factory_obj(**kwargs)


def load_sdk_attribute(
    *,
    module_names: tuple[str, ...],
    attribute_name: str,
    error_message: str,
) -> object:
    """Import the first available module and return ``attribute_name``."""
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        attr: object = getattr(module, attribute_name, None)
        if attr is not None:
            return attr
    raise ConfigurationError(error_message)


def resolve_request_field[T](
    request_value: T | None,
    provider_value: T | None,
) -> T | None:
    """Prefer ``ModelRequest`` field over ``provider_config`` when set."""
    return request_value if request_value is not None else provider_value


def resolve_json_object_mode(
    *,
    request_response_format: object,
    provider_response_format: object,
) -> bool:
    """Return True when either request or provider config requests JSON output."""
    fmt = resolve_request_field(request_response_format, provider_response_format)
    return fmt == "json_object"


def resolve_max_output_tokens(
    *,
    request_max_output_tokens: int | None,
    provider_max_tokens: int | None,
) -> int | None:
    """Prefer ``ModelRequest.max_output_tokens`` over provider config."""
    return resolve_request_field(request_max_output_tokens, provider_max_tokens)


def resolve_temperature(
    *,
    request_temperature: float | None,
    provider_temperature: float | None,
) -> float | None:
    """Prefer ``ModelRequest.temperature`` over provider config."""
    return resolve_request_field(request_temperature, provider_temperature)


def ensure_json_hint_in_openai_messages(messages: Sequence[object]) -> None:
    """OpenAI requires the word ``json`` in messages when using json_object format."""
    dict_messages: list[dict[str, object]] = []
    for raw in messages:
        if is_object_dict(raw):
            dict_messages.append(dict(raw))
    all_text = " ".join(
        str(m.get("content")) for m in dict_messages if isinstance(m.get("content"), str)
    )
    if "json" in all_text.lower():
        return
    for m in reversed(dict_messages):
        content_value = m.get("content")
        if m.get("role") == "user" and isinstance(content_value, str):
            m["content"] = content_value + "\nRespond in JSON."
            return


async def invoke_openai_chat_create(client: object, **kwargs: object) -> object:
    """Call ``client.chat.completions.create`` with None-stripped kwargs."""
    chat_obj: object = getattr(client, "chat", None)
    completions_obj: object = getattr(chat_obj, "completions", None)
    create_fn: object = getattr(completions_obj, "create", None)
    return await await_call(create_fn, **kwargs)


def openai_choice_text(choice: object) -> str:
    """Extract assistant text from a single OpenAI-compatible completion choice."""
    message_obj: object = getattr(choice, "message", None)
    content: object = getattr(message_obj, "content", None) if message_obj is not None else None
    return str(content or "")


def first_list_item(items: object) -> object | None:
    """Return the first element when *items* is a non-empty list."""
    if not is_object_list(items) or not items:
        return None
    return items[0]


def openai_first_choice_text(response: object) -> str:
    """Extract assistant text from an OpenAI-compatible chat completion response."""
    choices_obj: object = getattr(response, "choices", None)
    first_choice = first_list_item(choices_obj)
    if first_choice is None:
        return ""
    return openai_choice_text(first_choice)


def openai_stream_delta_text(chunk: object) -> str | None:
    """Extract incremental text from an OpenAI-compatible stream chunk."""
    choices_obj: object = getattr(chunk, "choices", None)
    first_choice = first_list_item(choices_obj)
    if first_choice is None:
        return None
    delta_obj: object = getattr(first_choice, "delta", None)
    if delta_obj is None:
        return None
    delta: object = getattr(delta_obj, "content", None)
    return str(delta) if delta else None


@runtime_checkable
class _AsyncObjectStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...


async def iter_openai_stream_text(stream: object) -> AsyncIterator[str]:
    """Yield non-empty text deltas from an OpenAI-compatible stream."""
    if not isinstance(stream, _AsyncObjectStream):
        raise ConfigurationError("OpenAI stream response is not async iterable")
    async for chunk_raw in stream:
        chunk_obj: object = chunk_raw
        delta = openai_stream_delta_text(chunk_obj)
        if delta:
            yield delta


def openai_finish_reason(choice: object) -> str | None:
    """Read ``finish_reason`` from an OpenAI-compatible completion choice."""
    reason: object = getattr(choice, "finish_reason", None)
    return reason if isinstance(reason, str) else None


__all__ = [
    "await_call",
    "await_method",
    "build_kwargs",
    "invoke_openai_chat_create",
    "iter_openai_stream_text",
    "load_sdk_attribute",
    "load_sdk_factory",
    "openai_choice_text",
    "openai_finish_reason",
    "openai_first_choice_text",
    "openai_stream_delta_text",
    "ensure_json_hint_in_openai_messages",
    "resolve_json_object_mode",
    "resolve_max_output_tokens",
    "resolve_request_field",
    "resolve_temperature",
]
