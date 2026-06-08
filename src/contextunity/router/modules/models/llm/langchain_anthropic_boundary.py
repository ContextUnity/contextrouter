"""Typed boundary for optional ``langchain_anthropic.ChatAnthropic``."""

from __future__ import annotations

from typing import Protocol

from contextunity.core.exceptions import ConfigurationError
from langchain_core.language_models.chat_models import BaseChatModel

from ..boundary_common import load_sdk_attribute


class _ChatAnthropicFactory(Protocol):
    def __call__(self, **kwargs: object) -> BaseChatModel: ...


def load_chat_anthropic_factory() -> _ChatAnthropicFactory:
    """Return a callable that constructs ``ChatAnthropic`` instances."""
    try:
        factory_obj = load_sdk_attribute(
            module_names=("langchain_anthropic",),
            attribute_name="ChatAnthropic",
            error_message="langchain_anthropic.ChatAnthropic is unavailable",
        )
    except ConfigurationError as exc:
        raise ModuleNotFoundError(
            "Anthropic provider requires `contextunity.router[models-anthropic]`."
        ) from exc

    if not callable(factory_obj):
        raise ConfigurationError("langchain_anthropic.ChatAnthropic is unavailable")

    def _construct(**kwargs: object) -> BaseChatModel:
        instance: object = factory_obj(**kwargs)
        if isinstance(instance, BaseChatModel):
            return instance
        raise ConfigurationError("ChatAnthropic returned an incompatible instance")

    return _construct


__all__ = ["load_chat_anthropic_factory"]
