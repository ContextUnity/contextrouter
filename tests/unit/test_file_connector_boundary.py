"""Path-boundary tests for FileConnector.

Pins the contract: the scan never reads outside allowed_root — neither via a
root that escapes the boundary nor via symlinks resolving outside the tree.
"""

from __future__ import annotations

import pytest
from contextunity.core.exceptions import SecurityError

from contextunity.router.modules.connectors.file import FileConnector


async def _collect_paths(connector: FileConnector) -> list[str]:
    return [str(unit.payload["path"]) async for unit in connector.connect()]


@pytest.mark.asyncio
async def test_files_inside_root_are_yielded(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("world")

    paths = await _collect_paths(FileConnector(root=tmp_path))
    assert len(paths) == 2


def test_root_escaping_allowed_root_rejected(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with pytest.raises(SecurityError, match="escapes allowed_root"):
        FileConnector(root="/", allowed_root=sandbox)


def test_parent_traversal_root_rejected(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with pytest.raises(SecurityError, match="escapes allowed_root"):
        FileConnector(root=sandbox / ".." / "..", allowed_root=sandbox)


@pytest.mark.asyncio
async def test_symlink_escaping_boundary_is_skipped(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do-not-read")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "ok.txt").write_text("fine")
    (sandbox / "escape.txt").symlink_to(secret)

    paths = await _collect_paths(FileConnector(root=sandbox))
    assert len(paths) == 1
    assert paths[0].endswith("ok.txt")


@pytest.mark.asyncio
async def test_symlink_inside_boundary_is_allowed(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("inside")
    (tmp_path / "alias.txt").symlink_to(target)

    paths = await _collect_paths(FileConnector(root=tmp_path))
    # Both the real file and the in-boundary symlink are readable.
    assert len(paths) == 2
