"""Secure graph serialization utilities.

This module provides secure alternatives to pickle for serializing NetworkX graphs
with integrity verification to prevent tampering.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def save_graph_secure(graph: Any, file_path: Path, hash_file_path: Path | None = None) -> None:
    """Save a graph securely with integrity verification.

    Args:
        graph: The graph object to save
        file_path: Path to save the graph file
        hash_file_path: Optional path to save the integrity hash (defaults to file_path + '.sha256')
    """
    if hash_file_path is None:
        hash_file_path = file_path.with_suffix(file_path.suffix + ".sha256")

    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)
    hash_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the graph using joblib (safer than pickle)
    joblib.dump(graph, file_path, compress=True, protocol=4)

    # Compute and save integrity hash
    file_hash = _compute_file_hash(file_path)
    hash_file_path.write_text(file_hash)

    logger.debug(
        "Graph saved securely to %s with integrity hash at %s",
        file_path,
        hash_file_path,
    )


def load_graph_secure(file_path: Path, hash_file_path: Path | None = None) -> Any:
    """Load a graph securely with integrity verification.

    Args:
        file_path: Path to the graph file
        hash_file_path: Optional path to the integrity hash file (defaults to file_path + '.sha256')

    Returns:
        The loaded graph object

    Raises:
        ValueError: If integrity check fails or file doesn't exist
        FileNotFoundError: If graph file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Graph file not found: {file_path}")

    if hash_file_path is None:
        hash_file_path = file_path.with_suffix(file_path.suffix + ".sha256")

    # Verify integrity if hash file exists
    if hash_file_path.exists():
        expected_hash = hash_file_path.read_text().strip()
        actual_hash = _compute_file_hash(file_path)

        if expected_hash != actual_hash:
            raise ValueError(
                f"Integrity check failed for {file_path}. "
                "File may have been tampered with. "
                "Expected hash: {expected_hash}, Actual hash: {actual_hash}"
            )
    else:
        logger.warning(
            "No integrity hash file found for %s. "
            "Loading without verification - this reduces security.",
            file_path,
        )

    # Load the graph using joblib
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load graph from {file_path}: {e}") from e
