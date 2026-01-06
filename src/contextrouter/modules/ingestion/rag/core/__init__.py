"""Core ingestion pipeline components."""

from .batch import (
    BatchResult,
    batch_transform,
    batch_validate,
    chunked,
    filter_by_indices,
)
from .loaders import (
    FileLoaderMixin,
    LoadedFile,
    iter_files,
    load_text_files,
    read_text_file,
)
from .plugins import IngestionPlugin
from .registry import get_all_plugins, get_plugin_class, register_plugin
from .types import IngestionMetadata, RawData, ShadowRecord
from .utils import (
    clean_str_list,
    get_graph_enrichment,
    load_taxonomy_safe,
    normalize_ambiguous_unicode,
    normalize_clean_text,
    parallel_map,
    resolve_workers,
)

__all__ = [
    "IngestionMetadata",
    "RawData",
    "ShadowRecord",
    "IngestionPlugin",
    "register_plugin",
    "get_plugin_class",
    "get_all_plugins",
    "BatchResult",
    "batch_transform",
    "batch_validate",
    "chunked",
    "filter_by_indices",
    "FileLoaderMixin",
    "LoadedFile",
    "iter_files",
    "load_text_files",
    "read_text_file",
    "normalize_ambiguous_unicode",
    "normalize_clean_text",
    "parallel_map",
    "resolve_workers",
    "clean_str_list",
    "get_graph_enrichment",
    "load_taxonomy_safe",
]
