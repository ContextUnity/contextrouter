"""Tests for ingestion file loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from contextrouter.modules.ingestion.rag import (
    FileLoaderMixin,
    LoadedFile,
    iter_files,
    load_text_files,
    read_text_file,
)
from contextrouter.modules.ingestion.rag.core.loaders import (
    _should_skip,
    find_alternative_dir,
)


class TestShouldSkip:
    """Tests for _should_skip helper."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            # Should skip (temp files, hidden files)
            ("~$document.txt", True),
            (".hidden", True),
            ("__pycache__", True),
            ("file.tmp", True),
            ("file.bak", True),
            ("file.swp", True),
            # Should not skip (normal files)
            ("document.txt", False),
            ("README.md", False),
            ("data.json", False),
            ("test_file.py", False),
        ],
    )
    def test_skip_patterns(self, filename: str, expected: bool) -> None:
        assert _should_skip(filename) == expected


class TestIterFiles:
    """Tests for iter_files function."""

    def test_returns_empty_for_nonexistent_directory(self) -> None:
        result = list(iter_files(Path("/nonexistent/path")))
        assert result == []

    def test_filters_by_extension(self, tmp_path: Path) -> None:
        # Create test files
        (tmp_path / "doc.txt").write_text("txt content")
        (tmp_path / "doc.md").write_text("md content")
        (tmp_path / "doc.json").write_text("{}")

        result = list(iter_files(tmp_path, extensions=(".txt", ".md")))
        names = {p.name for p in result}

        assert names == {"doc.txt", "doc.md"}

    def test_skips_hidden_files_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "visible.txt").write_text("content")
        (tmp_path / ".hidden.txt").write_text("hidden")
        (tmp_path / "~$temp.txt").write_text("temp")

        result = list(iter_files(tmp_path, extensions=(".txt",)))
        names = {p.name for p in result}

        assert names == {"visible.txt"}

    def test_includes_hidden_when_disabled(self, tmp_path: Path) -> None:
        (tmp_path / "visible.txt").write_text("content")
        (tmp_path / ".hidden.txt").write_text("hidden")

        result = list(iter_files(tmp_path, extensions=(".txt",), skip_hidden=False))
        names = {p.name for p in result}

        assert ".hidden.txt" in names

    def test_recursive_mode(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("root")
        (subdir / "nested.txt").write_text("nested")

        # Non-recursive
        result = list(iter_files(tmp_path, extensions=(".txt",), recursive=False))
        assert len(result) == 1
        assert result[0].name == "root.txt"

        # Recursive
        result = list(iter_files(tmp_path, extensions=(".txt",), recursive=True))
        names = {p.name for p in result}
        assert names == {"root.txt", "nested.txt"}

    def test_returns_sorted_paths(self, tmp_path: Path) -> None:
        (tmp_path / "c.txt").write_text("c")
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        result = list(iter_files(tmp_path, extensions=(".txt",)))
        names = [p.name for p in result]

        assert names == ["a.txt", "b.txt", "c.txt"]


class TestReadTextFile:
    """Tests for read_text_file function."""

    def test_reads_utf8_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "utf8.txt"
        file_path.write_text("Hello 世界", encoding="utf-8")

        result = read_text_file(file_path)

        assert result is not None
        assert result.content == "Hello 世界"
        assert result.encoding == "utf-8"
        assert result.path == file_path

    def test_falls_back_to_cp1252(self, tmp_path: Path) -> None:
        file_path = tmp_path / "cp1252.txt"
        # Write bytes that are valid cp1252 but invalid utf-8
        file_path.write_bytes("Caf\xe9".encode("cp1252"))

        result = read_text_file(file_path)

        assert result is not None
        assert "Caf" in result.content
        assert result.encoding in ("cp1252", "utf-8+ignore")

    def test_returns_none_for_nonexistent_file(self, tmp_path: Path) -> None:
        result = read_text_file(tmp_path / "nonexistent.txt")
        assert result is None

    def test_custom_encodings(self, tmp_path: Path) -> None:
        file_path = tmp_path / "custom.txt"
        file_path.write_text("test", encoding="utf-8")

        result = read_text_file(file_path, encodings=("utf-8",))

        assert result is not None
        assert result.encoding == "utf-8"


class TestLoadTextFiles:
    """Tests for load_text_files function."""

    def test_loads_all_matching_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.txt").write_text("content b")
        (tmp_path / "c.json").write_text("{}")

        result = load_text_files(tmp_path, extensions=(".txt",))

        assert len(result) == 2
        contents = {f.content for f in result}
        assert contents == {"content a", "content b"}

    def test_returns_empty_for_nonexistent_dir(self) -> None:
        result = load_text_files(Path("/nonexistent"))
        assert result == []

    def test_applies_transform_function(self, tmp_path: Path) -> None:
        (tmp_path / "test.txt").write_text("hello")

        def transform(loaded: LoadedFile) -> str:
            return loaded.content.upper()

        result = load_text_files(tmp_path, extensions=(".txt",), transform=transform)

        assert result == ["HELLO"]

    def test_skips_none_from_transform(self, tmp_path: Path) -> None:
        (tmp_path / "keep.txt").write_text("keep")
        (tmp_path / "skip.txt").write_text("skip")

        def transform(loaded: LoadedFile) -> str | None:
            return None if "skip" in loaded.content else loaded.content

        result = load_text_files(tmp_path, extensions=(".txt",), transform=transform)

        assert result == ["keep"]


class TestFindAlternativeDir:
    """Tests for find_alternative_dir function."""

    def test_finds_first_existing_alternative(self, tmp_path: Path) -> None:
        alt_dir = tmp_path / "alternative"
        alt_dir.mkdir()
        base_path = tmp_path / "primary"  # doesn't exist

        result = find_alternative_dir(base_path, ("other", "alternative"))

        assert result == alt_dir

    def test_returns_none_when_no_alternatives_exist(self, tmp_path: Path) -> None:
        base_path = tmp_path / "primary"

        result = find_alternative_dir(base_path, ("alt1", "alt2"))

        assert result is None

    def test_returns_none_for_empty_alternatives(self, tmp_path: Path) -> None:
        base_path = tmp_path / "primary"

        result = find_alternative_dir(base_path, ())

        assert result is None


class TestFileLoaderMixin:
    """Tests for FileLoaderMixin class."""

    class TestPlugin(FileLoaderMixin):
        """Test plugin using the mixin."""

        pass

    def test_resolve_source_dir_returns_existing_dir(self, tmp_path: Path) -> None:
        plugin = self.TestPlugin()
        existing = tmp_path / "source"
        existing.mkdir()

        result = plugin._resolve_source_dir(str(existing))

        assert result == existing

    def test_resolve_source_dir_tries_alternatives(self, tmp_path: Path) -> None:
        plugin = self.TestPlugin()
        alt = tmp_path / "q&a"
        alt.mkdir()
        primary = tmp_path / "qa"  # doesn't exist

        result = plugin._resolve_source_dir(str(primary), alternatives=("q&a",))

        assert result == alt

    def test_resolve_source_dir_returns_none_when_missing(self, tmp_path: Path) -> None:
        plugin = self.TestPlugin()
        missing = tmp_path / "nonexistent"

        result = plugin._resolve_source_dir(str(missing))

        assert result is None

    def test_load_text_files_delegates_correctly(self, tmp_path: Path) -> None:
        plugin = self.TestPlugin()
        (tmp_path / "test.txt").write_text("content")

        result = plugin._load_text_files(tmp_path, extensions=(".txt",))

        assert len(result) == 1
        assert result[0].content == "content"

    def test_iter_files_delegates_correctly(self, tmp_path: Path) -> None:
        plugin = self.TestPlugin()
        (tmp_path / "test.txt").write_text("content")
        (tmp_path / "other.md").write_text("markdown")

        result = list(plugin._iter_files(tmp_path, extensions=(".txt",)))

        assert len(result) == 1
        assert result[0].name == "test.txt"
