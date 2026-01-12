"""Model layer (LLM + embeddings) decoupled from agents."""

from __future__ import annotations

from .base import BaseEmbeddings, BaseModel
from .registry import ModelRegistry, model_registry

__all__ = [
    "BaseModel",
    "BaseEmbeddings",
    "ModelRegistry",
    "model_registry",
]
