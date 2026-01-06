"""Ingestion pipeline configuration."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RAGConfig(BaseModel):
    """Retrieval-Augmented Generation configuration.

    Controls the full ingestion pipeline from raw data to indexed knowledge.
    """

    model_config = ConfigDict(extra="ignore")

    # Source data discovery
    data_path: str = "data"

    # Processing controls
    skip_preprocess: bool = False
    skip_structure: bool = False
    skip_index: bool = False
    skip_deploy: bool = False

    # Performance tuning
    workers: int = Field(default_factory=lambda: max(1, int(((os.cpu_count()) or 2) // 2)))

    # Output paths (relative to data_path)
    assets_path: str = "assets"
    clean_text_path: str = "clean_text"
    taxonomy_path: str = "taxonomy.json"
    ontology_path: str = "ontology.json"
    graph_path: str = "knowledge_graph.pickle"
    shadow_path: str = "shadow_records"

    # Content processing
    max_file_size_mb: int = 100
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".html", ".json", ".xml"]
    )

    # Taxonomy generation
    taxonomy_enabled: bool = True
    taxonomy_samples: int = 100
    taxonomy_categories: dict[str, str] = Field(default_factory=dict)

    # Ontology generation
    ontology_enabled: bool = True

    # Graph building
    graph_enabled: bool = True
    graph_builder_mode: Literal["llm", "local", "hybrid"] = "llm"
    graph_model: str = "vertex/gemini-2.5-pro"

    # Shadow records
    shadow_enabled: bool = True

    # Deployment
    deploy_enabled: bool = True
    deploy_provider: str = "vertex"  # vertex, postgres, etc.
    deploy_wait: bool = True

    # Reporting
    report_enabled: bool = True
