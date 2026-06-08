"""A2A protocol -- agent-to-agent communication stubs for federated multi-agent workflows."""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import ContextUnit
from pydantic import BaseModel, ConfigDict


class A2AMessage(BaseModel):
    """Delegation message between agents."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    from_agent: str
    to_agent: str
    payload: ContextUnit
    token_id: str
    delegation_type: str


__all__ = ["A2AMessage"]
