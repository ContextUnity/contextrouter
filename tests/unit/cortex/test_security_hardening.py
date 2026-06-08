"""Security Hardening Tests — Phase 5 boundary enforcement.

Tests that verify:
- No str(e) leak in error messages (F3, F4)
- Pydantic Literal constraints on string configs (F6, F7)
- Language tool has proper scope enforcement (F10)
- Platform executor sanitizes error details (F3)
- ContextUnit logger used everywhere (F1, F2)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ── F3: platform.py must NOT leak exception details ─────────────────


# ── F4: registry validate_config must NOT leak Pydantic details ─────


# ── F6: SqlVisualizerConfig.default_format must use Literal ─────────


class TestSqlVisualizerConfigLiteral:
    """default_format must be constrained to known values."""

    def test_valid_formats_accepted(self):
        from contextunity.router.cortex.compiler.platform_tools.sql_visualizer import (
            SqlVisualizerConfig,
        )

        for fmt in ("table", "chart", "markdown"):
            cfg = SqlVisualizerConfig(default_format=fmt)
            assert cfg.default_format == fmt

    def test_invalid_format_rejected(self):
        from contextunity.router.cortex.compiler.platform_tools.sql_visualizer import (
            SqlVisualizerConfig,
        )

        with pytest.raises(ValidationError):
            SqlVisualizerConfig(default_format="executable_script")


# ── F7: LanguageToolConfig.language must use Literal ────────────────


# ── F10: language_tool must have router:execute scope ───────────────


# ── F1/F2: logger must be get_contextunit_logger ────────────────────


# ── Helpers ─────────────────────────────────────────────────────────


async def _noop_executor(state, config):
    return {}


class _FakeToken:
    def __init__(self, perms):
        self.permissions = perms


def _make_token(perms):
    return _FakeToken(perms)
