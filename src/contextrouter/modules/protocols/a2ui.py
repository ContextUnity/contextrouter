"""A2UI protocol (agent-to-UI) stubs."""

from __future__ import annotations

from contextcore import ContextUnit
from pydantic import BaseModel, ConfigDict


class A2UIWidget(BaseModel):
    """Structured UI widget emitted by agents."""

    model_config = ConfigDict(extra="ignore")

    widget_type: str
    data: ContextUnit
    token_id: str


__all__ = ["A2UIWidget"]
