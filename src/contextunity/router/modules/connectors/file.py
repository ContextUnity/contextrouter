"""File connector -- local and cloud file ingestion adapter (PDF, DOCX, CSV, etc.)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

from contextunity.core import ContextUnit
from contextunity.core.exceptions import SecurityError

from contextunity.router.core.interfaces import BaseConnector


class FileConnector(BaseConnector):
    """Walk a directory tree and yield each matching file as a binary ``ContextUnit``."""

    def __init__(
        self,
        *,
        root: str | Path = ".",
        extensions: list[str] | None = None,
        recursive: bool = True,
        allowed_root: str | Path | None = None,
    ) -> None:
        """Set the scan *root*, optional extension filter, and recursion flag.

        Args:
            allowed_root: Boundary the scan may never leave. ``root`` must
                resolve inside it, and files whose resolved path (after
                symlinks) escapes it are skipped. Defaults to the resolved
                ``root`` itself, so symlinks cannot lead reads out of the tree.

        Raises:
            SecurityError: If ``root`` resolves outside ``allowed_root``.
        """
        self._root: Path = Path(root)
        resolved_root = self._root.resolve()
        self._boundary: Path = (
            Path(allowed_root).resolve() if allowed_root is not None else resolved_root
        )
        if not resolved_root.is_relative_to(self._boundary):
            raise SecurityError(
                f"FileConnector root '{resolved_root}' escapes allowed_root '{self._boundary}'"
            )
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
            # Re-resolve per file: a symlink inside the tree must not pull
            # reads from outside the boundary.
            if not p.resolve().is_relative_to(self._boundary):
                continue
            data = p.read_bytes()
            unit = ContextUnit(
                payload={"data": data, "path": str(p)},
                provenance=[f"connector:file:{p.name}"],
                modality="binary",
            )
            yield unit


__all__ = ["FileConnector"]
