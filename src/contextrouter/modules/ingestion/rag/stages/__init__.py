"""Ingestion staged pipeline.

Stages are explicit, persisted, and composable:
- preprocess: Raw -> CleanText
- persona: CleanText -> persona.txt (optional)
- taxonomy: CleanText -> taxonomy.json
- enrich: CleanText -> NER + keyphrases (optional)
- graph: CleanText + taxonomy -> knowledge_graph.pickle
- shadow: CleanText + taxonomy + graph -> ShadowRecords (per type)
- export: ShadowRecords -> per-type JSONL (Vertex import format)
- deploy: JSONL -> upload + index
"""

from .enrich import enrich_clean_text

__all__ = ["enrich_clean_text"]
