"""Base configuration utilities and imports."""

from __future__ import annotations

import os

# Try to import tomllib for Python 3.11+ TOML support
try:
    import tomllib  # type: ignore[import-untyped]
except ImportError:
    # Fallback for older Python versions
    tomllib = None  # type: ignore[assignment]


# ---- Environment access (core-only) -----------------------------------------
#
# Modules must not read `os.environ` directly. If something truly needs to read
# environment variables, route through this module so the policy is enforceable.


def get_env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None:
        return default
    s = val.strip()
    return s if s else default


def get_bool_env(name: str, default: bool | None = None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def set_env_default(name: str, value: str) -> None:
    """Set an environment variable default value (core-only).

    This is the only sanctioned place for *writing* to `os.environ`.
    Prefer config/TOML/env inputs; use this only for SDK feature flags that must
    exist before third-party imports instantiate clients.
    """
    os.environ.setdefault(name, value)


# Security policy constants (moved from security/policies.py)
DEFAULT_READ_PERMISSION = "RAG_READ"
DEFAULT_WRITE_PERMISSION = "RAG_WRITE"
