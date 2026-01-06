"""Bisquit transport envelope (Pydantic).

Required contract (principal spec):
- id
- content
- token_id
- provenance: List[str]
- metadata

Backward compatibility:
- Existing code uses `data` and `citations`. We keep them and normalize `content`↔`data`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BisquitEnvelope(BaseModel):
    """Envelope for data passing through the pipeline."""

    model_config = ConfigDict(extra="ignore")

    # Principal contract
    id: str | None = None
    content: Any = None

    # Backward compatibility
    data: Any = None
    provenance: list[str] = Field(default_factory=list)
    # Keep citations untyped here to avoid importing brain models from core
    # (prevents circular imports during module initialization).
    citations: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_id: str | None = None

    @model_validator(mode="after")
    def _sync_content_data(self) -> "BisquitEnvelope":
        # Allow callers to set either `data` (legacy) or `content` (new).
        if self.content is None and self.data is not None:
            self.content = self.data
        if self.data is None and self.content is not None:
            self.data = self.content
        return self

    def sign(self, token_id: str) -> "BisquitEnvelope":
        self.token_id = token_id
        return self

    def add_trace(self, stage: str) -> "BisquitEnvelope":
        # Minimal tracing hook: append stage marker into provenance trail.
        s = stage.strip()
        if s:
            self.provenance.append(s)
        return self


__all__ = ["BisquitEnvelope"]
