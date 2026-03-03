"""Model registry for LLMs and embeddings."""

from __future__ import annotations

from .core import ModelKey, Registry
from .fallback import FallbackModel
from .main import ModelRegistry, model_registry
from .types import ModelSelectionStrategy

__all__ = [
    "ModelRegistry",
    "model_registry",
    "ModelKey",
    "FallbackModel",
    "Registry",
    "ModelSelectionStrategy",
]
