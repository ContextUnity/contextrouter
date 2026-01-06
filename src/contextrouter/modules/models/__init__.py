"""Model layer (LLM + embeddings) decoupled from agents."""

from __future__ import annotations

from .base import BaseEmbeddings, BaseLLM
from .registry import ModelRegistry, model_registry

__all__ = [
    "BaseLLM",
    "BaseEmbeddings",
    "ModelRegistry",
    "model_registry",
]
