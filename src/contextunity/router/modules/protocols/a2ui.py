"""A2UI protocol -- agent-to-UI event streaming stubs for real-time frontend updates."""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import ContextUnit
from pydantic import BaseModel, ConfigDict


class A2UIWidget(BaseModel):
    """Structured UI widget emitted by agents."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    widget_type: str
    data: ContextUnit
    token_id: str


__all__ = ["A2UIWidget"]
