"""Knowledge graph package.

IMPORTANT: keep imports in this module **lazy**.

The runtime cortex imports `contextrouter.modules.ingestion.rag.graph.serialization` to load the
persisted graph. Importing heavy ingestion dependencies (like `networkx`) at package import time
breaks minimal installs (e.g., API-only deployments).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["GRAPH_EXTRACTION_PROMPT", "GraphBuilder", "GraphEnricher"]


def __getattr__(name: str) -> Any:
    if name == "GraphBuilder":
        return importlib.import_module(".builder", __name__).GraphBuilder
    if name == "GraphEnricher":
        return importlib.import_module(".lookup", __name__).GraphEnricher
    if name == "GRAPH_EXTRACTION_PROMPT":
        return importlib.import_module(".prompts", __name__).GRAPH_EXTRACTION_PROMPT
    raise AttributeError(name)
