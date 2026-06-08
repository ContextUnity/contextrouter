"""Typed Protocol boundary for optional ``rlm`` package imports."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from contextunity.core.exceptions import ConfigurationError


@runtime_checkable
class RLMEngine(Protocol):
    custom_tools: dict[str, object] | None

    def completion(self, prompt: str) -> object: ...


@runtime_checkable
class RLMUsageSummary(Protocol):
    total_input_tokens: int
    total_output_tokens: int


def ensure_rlm_installed() -> None:
    """Import ``rlm`` or raise ``ImportError`` with install guidance."""
    _ = importlib.import_module("rlm")


def load_rlm_engine(**kwargs: object) -> RLMEngine:
    """Construct an ``RLM`` runtime instance."""
    module = importlib.import_module("rlm")
    rlm_cls: object = getattr(module, "RLM", None)
    if not callable(rlm_cls):
        raise ConfigurationError("rlm.RLM is unavailable")
    instance: object = rlm_cls(**kwargs)
    if not isinstance(instance, RLMEngine):
        raise ConfigurationError("rlm.RLM returned an incompatible instance")
    return instance


def load_rlm_logger(log_dir: str) -> object:
    """Construct an ``RLMLogger`` for trajectory logging."""
    logger_module = importlib.import_module("rlm.logger")
    logger_cls: object = getattr(logger_module, "RLMLogger", None)
    if not callable(logger_cls):
        raise ConfigurationError("rlm.logger.RLMLogger is unavailable")
    return logger_cls(log_dir=log_dir)


def rlm_response_text(result: object) -> str:
    """Extract completion text from an RLM result object."""
    response_obj: object = getattr(result, "response", None)
    if isinstance(response_obj, str):
        return response_obj
    return str(result)


def rlm_usage_tokens(result: object) -> tuple[int, int] | None:
    """Extract input/output token counts from an RLM result object."""
    usage_summary: object = getattr(result, "usage_summary", None)
    if not isinstance(usage_summary, RLMUsageSummary):
        return None
    input_tokens = int(usage_summary.total_input_tokens or 0)
    output_tokens = int(usage_summary.total_output_tokens or 0)
    return input_tokens, output_tokens


__all__ = [
    "RLMEngine",
    "ensure_rlm_installed",
    "load_rlm_engine",
    "load_rlm_logger",
    "rlm_response_text",
    "rlm_usage_tokens",
]
