"""File connector (source)."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseConnector


class FileConnector(BaseConnector):
    def __init__(
        self,
        *,
        root: str | Path = ".",
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> None:
        self._root = Path(root)
        self._extensions = [
            e.lower() for e in (extensions or []) if isinstance(e, str) and e.strip()
        ]
        self._recursive = bool(recursive)

    async def connect(self) -> AsyncIterator[BisquitEnvelope]:
        # Minimal async generator wrapper; downstream decides how to parse bytes/text.
        it = self._root.rglob("*") if self._recursive else self._root.glob("*")
        for p in sorted(it):
            if not p.is_file():
                continue
            if self._extensions and p.suffix.lower() not in self._extensions:
                continue
            data = p.read_bytes()
            yield BisquitEnvelope(
                data=data,
                provenance=[f"connector:file:{p.name}"],
                metadata={"path": str(p)},
            )


__all__ = ["FileConnector"]
