"""Shared registration helpers for platform tool modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeGuard

from pydantic import BaseModel

from .contracts import PlatformExecutor


def _is_platform_executor(value: object) -> TypeGuard[PlatformExecutor]:
    return callable(value)


@dataclass(frozen=True)
class ToolRegistrationSpec:
    """Declarative spec for one registry entry."""

    binding: str
    executor: object
    config_schema: type[BaseModel]
    required_scopes: list[str]


class PlatformRegistry(Protocol):
    """Structural registry boundary used by platform tool modules."""

    def register(
        self,
        binding: str,
        executor: PlatformExecutor,
        config_schema: type[BaseModel],
        required_scopes: list[str],
    ) -> None:
        """Register a platform tool entry."""


def register_tool_specs(registry: PlatformRegistry, specs: list[ToolRegistrationSpec]) -> None:
    """Register all tool specs into PlatformToolRegistry."""
    for spec in specs:
        executor = spec.executor
        if not _is_platform_executor(executor):
            msg = f"Tool '{spec.binding}' executor is not callable"
            raise TypeError(msg)
        registry.register(
            binding=spec.binding,
            executor=executor,
            config_schema=spec.config_schema,
            required_scopes=spec.required_scopes,
        )


__all__ = ["PlatformRegistry", "ToolRegistrationSpec", "register_tool_specs"]
