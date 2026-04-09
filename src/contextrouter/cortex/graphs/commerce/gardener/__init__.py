"""
Gardener v2 subgraph package.

Gardener: Normalizes products via deterministic + LLM two-pass pipeline.
Model is injected via graph config at registration time (template-based).
"""

from .graph import build_gardener_graph
from .state import GardenerState

__all__ = [
    "build_gardener_graph",
    "GardenerState",
]
