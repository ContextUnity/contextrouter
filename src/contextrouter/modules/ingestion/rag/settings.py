"""Typed ingestion configuration (Pydantic).

This module defines the single source of truth for the RAG ingestion pipeline config.
All ingestion code should accept `RagIngestionConfig` instead of `dict[str, Any]`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _default_assets_folder() -> Path:
    # Default to a user-writable folder relative to the current working directory.
    # This avoids writing into site-packages when the library is installed.
    return Path.cwd() / "assets" / "ingestion"


class IngestionSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    workers: int = Field(default_factory=lambda: max(1, int(((os.cpu_count()) or 2) // 2)))


class TaxonomySection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    philosophy_focus: str = (
        "Extract core concepts, terminology, and relationships from the material."
    )
    include_types: list[str] = Field(default_factory=lambda: ["video", "book", "qa", "knowledge"])
    max_samples: int = Field(default=100, ge=1)
    scan_model: str = "vertex/gemini-2.5-flash"
    hard_cap_samples: int = Field(default=500, ge=1)
    categories: dict[str, str] = Field(default_factory=dict)

    @field_validator("scan_model")
    @classmethod
    def _validate_model_key(cls, v: str) -> str:
        vv = (v or "").strip()
        if "/" not in vv:
            raise ValueError("taxonomy.scan_model must be a model key: 'provider/name'")
        return vv


class GraphSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    include_types: list[str] = Field(default_factory=lambda: ["video", "book", "qa", "knowledge"])
    incremental: bool = False
    model: str = "vertex/gemini-2.5-pro"
    builder_mode: Literal["llm", "local", "hybrid"] = "llm"
    cognee_enabled: bool = True

    @field_validator("model")
    @classmethod
    def _validate_model_key(cls, v: str) -> str:
        vv = (v or "").strip()
        if "/" not in vv:
            raise ValueError("graph.model must be a model key: 'provider/name'")
        return vv


class BookSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    llm_topic_extraction_enabled: bool = True


class QASection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    llm_speaker_detect_enabled: bool = False
    llm_question_filter_enabled: bool = True
    llm_question_validation_enabled: bool = True
    llm_answer_validation_enabled: bool = True
    corrections: dict[str, str] = Field(default_factory=dict)


class VideoSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    llm_clean_enabled: bool = False
    llm_clean_batch_size: int = Field(default=40, ge=1)
    corrections: dict[str, str] = Field(default_factory=dict)
    llm_summary_enabled: bool = False
    llm_summary_batch_size: int = Field(default=15, ge=1)
    llm_summary_max_sentences: int = Field(default=3, ge=1, le=10)
    llm_segment_validation_enabled: bool = True
    llm_segment_validation_batch_size: int = Field(default=50, ge=1)


class WebSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url_file: str = "url.toml"
    force_reindex: bool = False
    user_agent: str = "ContextrouterIngestionBot/1.0 (+https://example.com/bot)"
    llm_summary_enabled: bool = True
    llm_summary_max_chars: int = Field(default=8000, ge=100)
    llm_summary_output_chars: int = Field(default=240, ge=10)
    crawl_enabled: bool = False
    crawl_max_pages: int = Field(default=200, ge=1)
    crawl_max_depth: int = Field(default=2, ge=0)
    crawl_include_subdomains: bool = False
    crawl_concurrency: int = Field(default=4, ge=1, le=64)
    timeout_s: float = Field(default=20.0, ge=1.0)
    crawl_skip_url_substrings: list[str] = Field(
        default_factory=lambda: ["/cart", "/checkout", "/wp-admin", "/login", "/wp-login"]
    )


class PersonaSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    output_path: str = "persona.txt"
    persona_name: str = "Speaker Name"
    tone_sample_globs: list[str] = Field(
        default_factory=lambda: ["clean_text/qa.jsonl", "clean_text/video.jsonl"]
    )
    tone_sample_count: int = Field(default=50, ge=1)
    tone_max_chars_per_sample: int = Field(default=500, ge=50)
    bio_globs: list[str] = Field(default_factory=lambda: ["clean_text/knowledge.jsonl"])
    bio_include_full_text: bool = True
    max_output_tokens: int = Field(default=8192, ge=256, le=65536)


class UploadGCloudSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project_id: str | None = None
    location: str | None = None
    gcs_bucket: str | None = None
    data_store_id: str | None = None


class UploadSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    provider: Literal["gcloud"] = "gcloud"
    db_name: str = "green"
    include_date: bool = True
    gcloud: UploadGCloudSection = Field(default_factory=UploadGCloudSection)


class ValidationSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min_chunk_length: int = Field(default=50, ge=1)
    require_uppercase_start: bool = True


class PluginDir(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dir: str


class PluginsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    video: PluginDir = Field(default_factory=lambda: PluginDir(dir="video"))
    book: PluginDir = Field(default_factory=lambda: PluginDir(dir="book"))
    qa: PluginDir = Field(default_factory=lambda: PluginDir(dir="qa"))
    web: PluginDir = Field(default_factory=lambda: PluginDir(dir="web"))
    knowledge: PluginDir = Field(default_factory=lambda: PluginDir(dir="knowledge"))


class PathsSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    assets_folder: Path = Field(default_factory=_default_assets_folder)
    source_dir: str = "source"
    clean_text_dir: str = "clean_text"
    shadow_dir: str = "shadow"
    upload_dir: str = "output"
    jsonl_dir: str = "jsonl"
    processing_dir: str = "_processing"

    @field_validator("assets_folder", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v.strip():
            return Path(v)
        return _default_assets_folder()


class RagIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ingestion: IngestionSection = Field(default_factory=IngestionSection)
    taxonomy: TaxonomySection = Field(default_factory=TaxonomySection)
    graph: GraphSection = Field(default_factory=GraphSection)
    book: BookSection = Field(default_factory=BookSection)
    qa: QASection = Field(default_factory=QASection)
    video: VideoSection = Field(default_factory=VideoSection)
    web: WebSection = Field(default_factory=WebSection)
    persona: PersonaSection = Field(default_factory=PersonaSection)
    upload: UploadSection = Field(default_factory=UploadSection)
    validation: ValidationSection = Field(default_factory=ValidationSection)
    plugins: PluginsSection = Field(default_factory=PluginsSection)
    paths: PathsSection = Field(default_factory=PathsSection)

    def assets_paths(self) -> dict[str, Path]:
        assets_folder = self.paths.assets_folder
        if not assets_folder.is_absolute():
            assets_folder = Path.cwd() / assets_folder

        upload_dir = assets_folder / self.paths.upload_dir
        jsonl_dir = upload_dir / self.paths.jsonl_dir
        return {
            "assets_folder": assets_folder,
            "source": assets_folder / self.paths.source_dir,
            "clean_text": assets_folder / self.paths.clean_text_dir,
            "shadow": assets_folder / self.paths.shadow_dir,
            "upload": upload_dir,
            "jsonl": jsonl_dir,
            "processing": upload_dir / self.paths.processing_dir,
            "taxonomy": assets_folder / "taxonomy.json",
            "graph": assets_folder / "knowledge_graph.pickle",
            "ontology": assets_folder / "ontology.json",
        }


__all__ = ["RagIngestionConfig"]
