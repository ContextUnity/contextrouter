"""Internal brain event types (host-neutral).

The API transport layer can map these to AG-UI events. Telegram can map to message edits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

BrainEventType = Literal[
    "token",
    "citation",
    "suggestion",
    "node_start",
    "node_end",
    "tool_start",
    "tool_end",
    "tool_error",
    "error",
]


@dataclass(frozen=True)
class BrainEvent:
    type: BrainEventType
    node: str | None = None
    data: dict[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
