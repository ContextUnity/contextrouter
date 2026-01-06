"""A2UI protocol (agent-to-UI) stubs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from contextrouter.core.bisquit import BisquitEnvelope


class A2UIWidget(BaseModel):
    """Structured UI widget emitted by agents."""

    model_config = ConfigDict(extra="ignore")

    widget_type: str
    data: BisquitEnvelope
    token_id: str


__all__ = ["A2UIWidget"]
