"""Framework adapters (Transformers) for ingestion stages.

These are thin wrappers for FlowManager-style pipelines. Deep stage logic remains in
`contextunity.brain.ingestion.rag.*`.
"""

from __future__ import annotations

from . import graph, keyphrases, ner, ontology, shadow, taxonomy

__all__ = [
    "taxonomy",
    "graph",
    "shadow",
    "ontology",
    "ner",
    "keyphrases",
]
