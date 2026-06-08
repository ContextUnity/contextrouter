"""File connector -- local and cloud file ingestion adapter (PDF, DOCX, CSV, etc.)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

from contextunity.core import ContextUnit

from contextunity.router.core.interfaces import BaseConnector


class FileConnector(BaseConnector):
    """Walk a directory tree and yield each matching file as a binary ``ContextUnit``."""

    def __init__(
        self,
        *,
        root: str | Path = ".",
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> None:
        """Set the scan *root*, optional extension filter, and recursion flag."""
        self._root: Path = Path(root)
        self._extensions: list[str] = [
            extension.lower() for extension in (extensions or []) if extension.strip()
        ]
        self._recursive: bool = recursive

    @override
    def connect(self) -> AsyncIterator[ContextUnit]:
        return self._connect()

    async def _connect(self) -> AsyncIterator[ContextUnit]:
        """Glob the root directory, read each matching file as raw bytes, and yield a ``ContextUnit`` per file."""
        # Minimal async generator wrapper; downstream decides how to parse bytes/text.
        it = self._root.rglob("*") if self._recursive else self._root.glob("*")
        for p in sorted(it):
            if not p.is_file():
                continue
            if self._extensions and p.suffix.lower() not in self._extensions:
                continue
            data = p.read_bytes()
            unit = ContextUnit(
                payload={"data": data, "path": str(p)},
                provenance=[f"connector:file:{p.name}"],
                modality="binary",
            )
            yield unit


__all__ = ["FileConnector"]
