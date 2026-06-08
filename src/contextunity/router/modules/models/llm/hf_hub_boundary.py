"""Typed Protocol boundary for optional ``huggingface_hub.AsyncInferenceClient``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, overload

from contextunity.core.exceptions import ConfigurationError

from ..boundary_common import await_method, load_sdk_factory


class HFHubAsyncClient(Protocol):
    @overload
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Literal[False] = False,
    ) -> object: ...

    @overload
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Literal[True],
    ) -> AsyncIterator[object]: ...

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> object | AsyncIterator[object]: ...

    async def text_generation(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> object: ...

    async def automatic_speech_recognition(self, data: bytes) -> object: ...

    async def text_classification(self, prompt: str) -> object: ...

    async def visual_question_answering(self, image: bytes, question: str) -> object: ...

    async def image_to_text(self, image: bytes) -> object: ...


class _HFHubAsyncClientAdapter:
    """Adapter narrowing ``AsyncInferenceClient`` at the import boundary."""

    _inner: object

    def __init__(self, inner: object) -> None:
        self._inner = inner

    @overload
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Literal[False] = False,
    ) -> object: ...

    @overload
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: Literal[True],
    ) -> AsyncIterator[object]: ...

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
    ) -> object | AsyncIterator[object]:
        return await await_method(
            self._inner,
            "chat_completion",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

    async def text_generation(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> object:
        return await await_method(
            self._inner,
            "text_generation",
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    async def automatic_speech_recognition(self, data: bytes) -> object:
        return await await_method(self._inner, "automatic_speech_recognition", data)

    async def text_classification(self, prompt: str) -> object:
        return await await_method(self._inner, "text_classification", prompt)

    async def visual_question_answering(self, image: bytes, question: str) -> object:
        return await await_method(
            self._inner,
            "visual_question_answering",
            image,
            question,
        )

    async def image_to_text(self, image: bytes) -> object:
        return await await_method(self._inner, "image_to_text", image)


def load_async_inference_client(**kwargs: object) -> HFHubAsyncClient:
    """Construct a typed ``AsyncInferenceClient`` when the optional dependency is installed."""
    try:
        inner = load_sdk_factory(
            module_names=("huggingface_hub",),
            factory_name="AsyncInferenceClient",
            error_message="huggingface_hub.AsyncInferenceClient is unavailable",
            **kwargs,
        )
    except ConfigurationError as exc:
        raise ImportError(
            "HuggingFaceHubLLM requires `contextunity.router[models-hf-hub]`."
        ) from exc
    return _HFHubAsyncClientAdapter(inner)


__all__ = ["HFHubAsyncClient", "load_async_inference_client"]
