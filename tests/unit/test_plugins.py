"""Tests for the plugin manifest and context system.

Tests cover:
- PluginManifest validation (valid/invalid YAML)
- PluginContext capability enforcement
- scan() with manifest-based plugins
- Version compatibility checking
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextrouter.core.plugins import (
    PluginCapability,
    PluginContext,
    PluginManifest,
    load_manifest,
    load_plugin,
    reset_plugins,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset():
    """Reset plugin registry between tests."""
    reset_plugins()
    yield
    reset_plugins()


@pytest.fixture()
def tmp_plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary plugin directory."""
    plugin_dir = tmp_path / "my-plugin"
    plugin_dir.mkdir()
    return plugin_dir


def _write_manifest(plugin_dir: Path, content: str) -> Path:
    """Write a plugin.yaml file."""
    manifest_path = plugin_dir / "plugin.yaml"
    manifest_path.write_text(content)
    return manifest_path


def _write_entry_point(plugin_dir: Path, code: str, name: str = "plugin.py") -> Path:
    """Write a plugin entry point."""
    entry = plugin_dir / name
    entry.write_text(code)
    return entry


# ============================================================================
# PluginManifest Tests
# ============================================================================


class TestPluginManifest:
    """Test PluginManifest validation."""

    def test_valid_manifest(self):
        m = PluginManifest(
            name="my-plugin",
            version="1.0.0",
            description="Test plugin",
            capabilities=[PluginCapability.TOOLS, PluginCapability.GRAPHS],
        )
        assert m.name == "my-plugin"
        assert m.version == "1.0.0"
        assert PluginCapability.TOOLS in m.capabilities
        assert m.entry_point == "plugin.py"
        assert m.enabled is True

    def test_minimal_manifest(self):
        m = PluginManifest(name="x", version="0.0.1")
        assert m.capabilities == []
        assert m.entry_point == "plugin.py"

    def test_invalid_name_uppercase(self):
        with pytest.raises(ValueError):
            PluginManifest(name="MyPlugin", version="1.0.0")

    def test_invalid_name_spaces(self):
        with pytest.raises(ValueError):
            PluginManifest(name="my plugin", version="1.0.0")

    def test_invalid_name_empty(self):
        with pytest.raises(ValueError):
            PluginManifest(name="", version="1.0.0")

    def test_invalid_version(self):
        with pytest.raises(ValueError):
            PluginManifest(name="x", version="bad")

    def test_entry_point_path_traversal(self):
        with pytest.raises(ValueError, match="path separators"):
            PluginManifest(name="x", version="1.0.0", entry_point="../evil.py")

    def test_entry_point_not_py(self):
        with pytest.raises(ValueError, match=".py file"):
            PluginManifest(name="x", version="1.0.0", entry_point="plugin.sh")

    def test_with_requires(self):
        m = PluginManifest(
            name="x",
            version="1.0.0",
            requires={"contextrouter": ">=0.9.0"},
        )
        assert m.requires["contextrouter"] == ">=0.9.0"

    def test_disabled(self):
        m = PluginManifest(name="x", version="1.0.0", enabled=False)
        assert m.enabled is False


# ============================================================================
# PluginContext Tests
# ============================================================================


class TestPluginContext:
    """Test PluginContext capability enforcement."""

    def _ctx(self, caps: list[PluginCapability]) -> PluginContext:
        manifest = PluginManifest(name="test-plugin", version="1.0.0", capabilities=caps)
        return PluginContext(manifest, plugin_dir=Path("/tmp/test"))

    def test_properties(self):
        ctx = self._ctx([PluginCapability.TOOLS])
        assert ctx.name == "test-plugin"
        assert ctx.version == "1.0.0"
        assert PluginCapability.TOOLS in ctx.capabilities

    def test_capabilities_are_copy(self):
        """Ensure capabilities returns a copy, not mutable internal set."""
        ctx = self._ctx([PluginCapability.TOOLS])
        caps = ctx.capabilities
        caps.add(PluginCapability.GRAPHS)
        assert PluginCapability.GRAPHS not in ctx.capabilities

    def test_register_tool_without_capability(self):
        ctx = self._ctx([])  # No capabilities
        with pytest.raises(PermissionError, match="lacks 'tools' capability"):
            ctx.register_tool(MagicMock(name="fake-tool"))

    def test_register_graph_without_capability(self):
        ctx = self._ctx([PluginCapability.TOOLS])  # Only tools
        with pytest.raises(PermissionError, match="lacks 'graphs' capability"):
            ctx.register_graph("test", lambda: None)

    def test_register_connector_without_capability(self):
        ctx = self._ctx([])
        with pytest.raises(PermissionError, match="lacks 'connectors' capability"):
            ctx.register_connector("test", MagicMock)

    def test_register_provider_without_capability(self):
        ctx = self._ctx([])
        with pytest.raises(PermissionError, match="lacks 'providers' capability"):
            ctx.register_provider("test", MagicMock)

    def test_register_transformer_without_capability(self):
        ctx = self._ctx([])
        with pytest.raises(PermissionError, match="lacks 'transformers' capability"):
            ctx.register_transformer("test", MagicMock)

    def test_summary_empty(self):
        ctx = self._ctx([PluginCapability.TOOLS])
        s = ctx.summary()
        assert s["name"] == "test-plugin"
        assert s["tools"] == []
        assert s["graphs"] == []


# ============================================================================
# load_manifest Tests
# ============================================================================


class TestLoadManifest:
    """Test manifest loading from filesystem."""

    def test_load_valid_manifest(self, tmp_plugin_dir: Path):
        _write_manifest(
            tmp_plugin_dir,
            """\
name: test-plugin
version: 1.0.0
description: A test plugin
capabilities:
  - tools
  - graphs
entry_point: plugin.py
""",
        )
        m = load_manifest(tmp_plugin_dir)
        assert m is not None
        assert m.name == "test-plugin"
        assert m.version == "1.0.0"
        assert PluginCapability.TOOLS in m.capabilities
        assert PluginCapability.GRAPHS in m.capabilities

    def test_load_yml_extension(self, tmp_plugin_dir: Path):
        """Support plugin.yml as well as plugin.yaml."""
        (tmp_plugin_dir / "plugin.yml").write_text("name: yml-plugin\nversion: 0.1.0\n")
        m = load_manifest(tmp_plugin_dir)
        assert m is not None
        assert m.name == "yml-plugin"

    def test_no_manifest_returns_none(self, tmp_plugin_dir: Path):
        m = load_manifest(tmp_plugin_dir)
        assert m is None

    def test_invalid_manifest_raises(self, tmp_plugin_dir: Path):
        _write_manifest(tmp_plugin_dir, "just a string, not a mapping")
        with pytest.raises(ValueError, match="expected mapping"):
            load_manifest(tmp_plugin_dir)

    def test_missing_required_field(self, tmp_plugin_dir: Path):
        _write_manifest(tmp_plugin_dir, "description: no name or version\n")
        with pytest.raises(Exception):
            load_manifest(tmp_plugin_dir)


# ============================================================================
# load_plugin Tests
# ============================================================================


class TestLoadPlugin:
    """Test full plugin loading lifecycle."""

    def test_load_plugin_with_on_load(self, tmp_plugin_dir: Path):
        _write_manifest(
            tmp_plugin_dir,
            "name: hello-plugin\nversion: 1.0.0\ncapabilities: []\n",
        )
        _write_entry_point(
            tmp_plugin_dir,
            """\
LOADED = False

def on_load(ctx):
    global LOADED
    LOADED = True
""",
        )

        ctx = load_plugin(tmp_plugin_dir)
        assert ctx is not None
        assert ctx.name == "hello-plugin"

    def test_load_disabled_plugin(self, tmp_plugin_dir: Path):
        _write_manifest(
            tmp_plugin_dir,
            "name: disabled-plugin\nversion: 1.0.0\nenabled: false\n",
        )
        _write_entry_point(tmp_plugin_dir, "")

        ctx = load_plugin(tmp_plugin_dir)
        assert ctx is None

    def test_load_no_manifest(self, tmp_plugin_dir: Path):
        ctx = load_plugin(tmp_plugin_dir)
        assert ctx is None

    def test_load_missing_entry_point(self, tmp_plugin_dir: Path):
        _write_manifest(
            tmp_plugin_dir,
            "name: bad-plugin\nversion: 1.0.0\nentry_point: missing.py\n",
        )
        with pytest.raises(FileNotFoundError, match="entry point not found"):
            load_plugin(tmp_plugin_dir)

    def test_duplicate_plugin_skipped(self, tmp_plugin_dir: Path):
        _write_manifest(
            tmp_plugin_dir,
            "name: dup-plugin\nversion: 1.0.0\ncapabilities: []\n",
        )
        _write_entry_point(tmp_plugin_dir, "def on_load(ctx): pass\n")

        ctx1 = load_plugin(tmp_plugin_dir)
        ctx2 = load_plugin(tmp_plugin_dir)
        assert ctx1 is ctx2  # Same instance returned


# ============================================================================
# scan() Tests
# ============================================================================


class TestScan:
    """Test scan() with manifest plugins."""

    def test_scan_manifest_plugin(self, tmp_path: Path):
        from contextrouter.core.registry import scan

        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        _write_manifest(
            plugin_dir,
            "name: scanned-plugin\nversion: 1.0.0\ncapabilities: []\n",
        )
        _write_entry_point(plugin_dir, "def on_load(ctx): pass\n")

        loaded = scan(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].name == "scanned-plugin"

    def test_scan_nonexistent_dir(self, tmp_path: Path):
        from contextrouter.core.registry import scan

        loaded = scan(tmp_path / "nonexistent")
        assert loaded == []

    def test_scan_skips_non_plugin_dirs(self, tmp_path: Path):
        from contextrouter.core.registry import scan

        # Directory without plugin.yaml — should be ignored
        (tmp_path / "random-dir").mkdir()
        (tmp_path / "random-dir" / "stuff.txt").write_text("not a plugin")

        loaded = scan(tmp_path)
        assert loaded == []

    def test_scan_multiple_plugins(self, tmp_path: Path):
        from contextrouter.core.registry import scan

        for name in ["alpha-plugin", "beta-plugin"]:
            d = tmp_path / name
            d.mkdir()
            _write_manifest(d, f"name: {name}\nversion: 1.0.0\ncapabilities: []\n")
            _write_entry_point(d, "def on_load(ctx): pass\n")

        loaded = scan(tmp_path)
        assert len(loaded) == 2
        names = {p.name for p in loaded}
        assert names == {"alpha-plugin", "beta-plugin"}
