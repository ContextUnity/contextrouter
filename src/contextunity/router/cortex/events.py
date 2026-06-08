"""Internal brain event types (host-neutral).
These events are yielded directly by graph execution nodes.
They form the universal protocol for streaming graph responses to any transport layer
(e.g., gRPC streaming, AG-UI protocol, or raw SDK clients).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from contextunity.core.types import JsonDict

from contextunity.router.cortex.types import (
    LLMUsageData,
    ToolErrorData,
    ToolTelemetryPayload,
)

BrainEventType = Literal[
    "token",
    "citation",
    "suggestion",
    "node_start",
    "node_end",
    "tool_start",
    "tool_end",
    "tool_error",
    "tool_result",
    "llm_start",
    "llm_end",
    "llm_error",
    "error",
]

BrainEventPayload = (
    Mapping[str, object] | LLMUsageData | ToolTelemetryPayload | ToolErrorData | JsonDict
)


def _empty_event_data() -> dict[str, object]:
    return {}


@dataclass(frozen=True)
class BrainEvent:
    """A standardized, host-neutral telemetry event emitted during graph execution.

    These events form the universal protocol for streaming graph responses to any transport
    layer (e.g., gRPC stream responses, AG-UI protocol, or raw SDK client callbacks).

    Attributes:
        type: The type of brain event being dispatched.
        node: The name of the node where the event originated.
        data: Payload containing event-specific data (e.g. chunk text, tool name, usage stats).
        timestamp: Epoch timestamp indicating when the event was created.
    """

    type: BrainEventType
    node: str | None = None
    data: BrainEventPayload = field(default_factory=_empty_event_data)
    timestamp: float = field(default_factory=time.time)
