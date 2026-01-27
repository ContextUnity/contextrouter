"""A2A protocol (agent-to-agent) stubs."""

from __future__ import annotations

from contextcore import ContextUnit
from pydantic import BaseModel, ConfigDict


class A2AMessage(BaseModel):
    """Delegation message between agents."""

    model_config = ConfigDict(extra="ignore")

    from_agent: str
    to_agent: str
    payload: ContextUnit
    token_id: str
    delegation_type: str


__all__ = ["A2AMessage"]
