"""Model layer (LLM + embeddings) decoupled from agents."""

from __future__ import annotations

from .base import BaseEmbeddings, BaseLLM, BaseModel
from .registry import ModelRegistry, model_registry

__all__ = [
    "BaseModel",
    "BaseLLM",
    "BaseEmbeddings",
    "ModelRegistry",
    "model_registry",
]
