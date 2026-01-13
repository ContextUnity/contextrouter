"""Common file loading utilities and mixins for ingestion plugins.

This module provides reusable helpers for file operations during ingestion:

**Standalone Functions**:

- `iter_files(directory, extensions, recursive, skip_hidden)` - Iterate
  over files matching extensions. Skips temp files (~$, .swp, etc.).

- `read_text_file(path, encodings, errors)` - Read text with encoding
  fallback chain (utf-8 → cp1252 → latin-1). Returns LoadedFile or None.

- `load_text_files(directory, extensions, transform)` - Load all matching
  files, optionally applying transform function.

- `find_alternative_dir(base_path, alternatives)` - Find first existing
  alternative directory (e.g., "qa" vs "q&a").

**FileLoaderMixin**:

Mixin class for plugins providing:
- `_resolve_source_dir(path, alternatives)` - Resolve directory with fallbacks
- `_load_text_files(directory, extensions)` - Load files with encoding fallback
- `_iter_files(directory, extensions)` - Iterate over files

**Usage in Plugin**:

```python
@register_plugin("qa")
class QAPlugin(IngestionPlugin, FileLoaderMixin):
    def load(self, assets_path: str) -> list[RawData]:
        if not (source_dir := self._resolve_source_dir(assets_path, alternatives=("q&a",))):
            return []

        return [
            RawData(content=f.content, source_type="qa", metadata={"title": f.path.stem})
            for f in self._load_text_files(source_dir, extensions=(".txt", ".md"))
        ]
```
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Files to skip (temp files, system files)
SKIP_PREFIXES = ("~$", ".", "__")
SKIP_SUFFIXES = (".tmp", ".bak", ".swp")


@dataclass
class LoadedFile:
    """Result of loading a single file."""

    path: Path
    content: str
    encoding: str


def iter_files(
    directory: Path,
    *,
    extensions: tuple[str, ...] = (".txt", ".md"),
    recursive: bool = False,
    skip_hidden: bool = True,
) -> Iterator[Path]:
    """Iterate over files matching extensions in directory.

    Args:
        directory: Directory to scan
        extensions: File extensions to include (with dot)
        recursive: Whether to scan subdirectories
        skip_hidden: Skip files starting with . or ~$

    Yields:
        Path objects for matching files, sorted by name
    """
    if not directory.exists():
        return

    glob_pattern = "**/*" if recursive else "*"
    all_files = sorted(directory.glob(glob_pattern))

    for path in all_files:
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if skip_hidden and _should_skip(path.name):
            continue
        yield path


def _should_skip(filename: str) -> bool:
    """Check if file should be skipped based on name patterns."""
    return any(filename.startswith(p) for p in SKIP_PREFIXES) or any(
        filename.endswith(s) for s in SKIP_SUFFIXES
    )


def read_text_file(
    path: Path,
    *,
    encodings: tuple[str, ...] = ("utf-8", "cp1252", "latin-1"),
    errors: str = "ignore",
) -> LoadedFile | None:
    """Read text file with encoding fallback.

    Args:
        path: File path
        encodings: Encodings to try in order
        errors: Error handling for final fallback

    Returns:
        LoadedFile with content and detected encoding, or None on failure
    """
    for encoding in encodings:
        try:
            content = path.read_text(encoding=encoding)
            return LoadedFile(path=path, content=content, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning("Failed to read %s with %s: %s", path.name, encoding, e)
            return None

    # Final attempt with error handling
    try:
        content = path.read_text(encoding=encodings[0], errors=errors)
        return LoadedFile(path=path, content=content, encoding=f"{encodings[0]}+{errors}")
    except Exception as e:
        logger.warning("Failed to read %s: %s", path.name, e)
        return None


def load_text_files(
    directory: Path,
    *,
    extensions: tuple[str, ...] = (".txt", ".md"),
    transform: Callable[[LoadedFile], Any] | None = None,
) -> list[Any]:
    """Load all text files from directory.

    Args:
        directory: Directory to scan
        extensions: File extensions to include
        transform: Optional function to transform each LoadedFile

    Returns:
        List of loaded content (or transformed results)
    """
    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []

    results = []
    for path in iter_files(directory, extensions=extensions):
        if loaded := read_text_file(path):
            result = transform(loaded) if transform else loaded
            if result is not None:
                results.append(result)

    return results


def find_alternative_dir(base_path: Path, alternatives: tuple[str, ...]) -> Path | None:
    """Find first existing alternative directory.

    Args:
        base_path: Original path that doesn't exist
        alternatives: Alternative directory names to try

    Returns:
        First existing alternative path, or None
    """
    parent = base_path.parent
    for alt_name in alternatives:
        alt_path = parent / alt_name
        if alt_path.exists():
            logger.info("Using alternative directory: %s", alt_path)
            return alt_path
    return None


class FileLoaderMixin:
    """Mixin providing common file loading utilities for plugins.

    Usage in plugin:
        class MyPlugin(IngestionPlugin, FileLoaderMixin):
            def load(self, assets_path: str) -> list[RawData]:
                source_dir = self._resolve_source_dir(assets_path, alternatives=("alt_name",))
                if not source_dir:
                    return []

                return [
                    self._to_raw_data(loaded, "my_type", metadata_fn)
                    for loaded in self._load_text_files(source_dir)
                ]
    """

    def _resolve_source_dir(
        self,
        assets_path: str,
        *,
        alternatives: tuple[str, ...] = (),
    ) -> Path | None:
        """Resolve source directory, trying alternatives if primary doesn't exist."""
        source_dir = Path(assets_path)
        if source_dir.exists():
            return source_dir

        if alternatives and (alt := find_alternative_dir(source_dir, alternatives)):
            return alt

        logger.warning("Source directory does not exist: %s", assets_path)
        return None

    def _load_text_files(
        self,
        directory: Path,
        *,
        extensions: tuple[str, ...] = (".txt", ".md"),
    ) -> list[LoadedFile]:
        """Load text files from directory with encoding fallback."""
        return load_text_files(directory, extensions=extensions)

    def _iter_files(
        self,
        directory: Path,
        *,
        extensions: tuple[str, ...],
    ) -> Iterator[Path]:
        """Iterate over files in directory."""
        return iter_files(directory, extensions=extensions)
