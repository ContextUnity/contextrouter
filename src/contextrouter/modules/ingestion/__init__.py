"""Ingestion modules for various data processing pipelines.

This package provides modular ingestion engines for different use cases:

- `rag/`: RAG (Retrieval-Augmented Generation) ingestion pipeline
  - Processes content for knowledge base retrieval
  - Builds taxonomy, ontology, and knowledge graphs
  - Generates shadow records for semantic search

- `etl/`: ETL (Extract, Transform, Load) pipelines (stub)
  - Generic data extraction from multiple sources
  - Transformation pipelines for analytics
  - Loading into data warehouses

- `datasets/`: Dataset generation pipelines (stub)
  - Training/evaluation dataset preparation
  - Annotation and labeling workflows
  - Dataset export formats

- `indexing/`: Search index ingestion (stub)
  - Content normalization for search engines
  - Index schema mapping
  - Incremental updates and reindexing

- `observability/`: Log/event ingestion (stub)
  - Log aggregation and parsing
  - Metrics extraction
  - Storage in observability backends

All ingestion modules follow the modular architecture:
- Connectors: Raw data sources
- Transformers: Processing stages
- Providers: Storage sinks
"""

from __future__ import annotations

__all__: list[str] = []
