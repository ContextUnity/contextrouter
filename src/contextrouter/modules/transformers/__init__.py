"""Framework adapters (Transformers) for ingestion stages.

These are thin wrappers for FlowManager-style pipelines. Deep stage logic remains in
`contextrouter.modules.ingestion.rag.*`.
"""

from __future__ import annotations

__all__ = [
    "taxonomy",
    "graph",
    "shadow",
    "ontology",
    "ner",
    "keyphrases",
]
