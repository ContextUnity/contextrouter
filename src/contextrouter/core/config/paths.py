"""Configuration paths and file handling."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigPaths:
    """Standardized paths for configuration files and directories.

    This class provides a consistent interface for finding and validating
    configuration-related paths across the application.
    """

    # Root directory (package or project root)
    root: Path

    # Configuration files
    toml_config: Path = None  # type: ignore
    env_file: Path = None  # type: ignore

    # Data directories
    data_dir: Path = None  # type: ignore
    assets_dir: Path = None  # type: ignore
    logs_dir: Path = None  # type: ignore

    def __post_init__(self) -> None:
        # Set computed paths if not provided
        object.__setattr__(self, "toml_config", self.root / "settings.toml")
        object.__setattr__(self, "env_file", self.root / ".env")
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "assets_dir", self.root / "assets")
        object.__setattr__(self, "logs_dir", self.root / "logs")

    @classmethod
    def from_root(cls, root: Path | str) -> "ConfigPaths":
        """Create ConfigPaths from a root directory."""
        return cls(root=Path(root).resolve())

    def ensure_directories(self) -> None:
        """Ensure all path directories exist."""
        for path in [self.data_dir, self.assets_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def find_config_files(self) -> list[Path]:
        """Find all available configuration files in priority order."""
        candidates = [
            self.toml_config,
            self.env_file,
        ]
        return [p for p in candidates if p.exists()]
