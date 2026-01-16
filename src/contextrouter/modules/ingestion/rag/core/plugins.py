"""Base class for ingestion plugins.

This defines the plugin contract for the ingestion pipeline
(`contextrouter.modules.ingestion.rag.*`). It is ingestion-specific and intentionally
separate from the framework-level `contextrouter.core.interfaces`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from contextrouter.core import Config

from ..settings import RagIngestionConfig
from .types import RawData, ShadowRecord
from .utils import (
    build_enriched_input_text,
    get_graph_enrichment,
    load_taxonomy_safe,
    normalize_clean_text,
)


class IngestionPlugin(ABC):
    """Abstract base class for ingestion plugins.

    Each plugin handles a specific content type (video, book, qa, web, knowledge)
    and implements two phases: Load and Transform.

    Plugins can specify a default source directory name, which can be overridden
    in settings.toml via [plugins.{source_type}].dir
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Returns the source type string (e.g., 'video', 'book')."""

    @property
    def default_source_dir(self) -> str:
        """Returns the default source directory name for this plugin.

        Can be overridden in settings.toml via [plugins.{source_type}].dir
        Defaults to source_type if not overridden.

        Returns:
            Default directory name (e.g., 'video', 'q&a')
        """
        return self.source_type

    @abstractmethod
    def load(self, assets_path: str) -> list[RawData]:
        """Scans the assets directory and loads raw content."""

    @abstractmethod
    def transform(
        self,
        data: list[RawData],
        enrichment_func: Callable[[str], dict[str, Any]],
        *,
        taxonomy_path: Path | None = None,
        config: RagIngestionConfig | None = None,
        core_cfg: Config | None = None,
    ) -> list[ShadowRecord]:
        """Chunks the raw data and transforms it into ShadowRecords."""

    # ---- Optional shared helpers (recommended for consistency across plugins) ----

    def _load_taxonomy(self, taxonomy_path: Path | None = None) -> dict[str, Any] | None:
        return load_taxonomy_safe(taxonomy_path)

    def _graph_enrichment(
        self, *, text: str, enrichment_func: Callable[[str], dict[str, Any]]
    ) -> tuple[list[str], str, list[str]]:
        return get_graph_enrichment(text=text, enrichment_func=enrichment_func)

    def _build_input_text(
        self,
        *,
        content: str,
        keywords: list[str] | None = None,
        summary: str | None = None,
        parent_categories: list[str] | None = None,
    ) -> str:
        return build_enriched_input_text(
            content=content,
            keywords=keywords,
            summary=summary,
            parent_categories=parent_categories,
        )

    def _clean_ui_text(self, text: str) -> str:
        return normalize_clean_text(text)


__all__ = ["IngestionPlugin"]
