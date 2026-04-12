"""Retrieval module (orchestration and multi-source knowledge retrieval).

This package contains the core orchestration layer for fetching data from any source
(Web, SQL, Vector, RSS) through bidirectional pipelines.
"""

from __future__ import annotations

from .orchestrator import (
    RetrievalOrchestrator,
    RetrievalResult,
)
from .pipeline import BaseRetrievalPipeline, PipelineResult

__all__ = [
    "RetrievalOrchestrator",
    "RetrievalResult",
    "BaseRetrievalPipeline",
    "PipelineResult",
]
