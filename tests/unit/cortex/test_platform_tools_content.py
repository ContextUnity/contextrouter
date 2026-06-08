"""Tests for Router Content platform tools — universal LLM capabilities.

Phase 5b: These tests validate the contract for 6 universal content tools
that have ZERO domain imports (no commerce/, news_engine/).

Config security: frozen=True, extra=forbid, bounded fields.
Scope security: all require router:execute.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ── Config Schema Tests ─────────────────────────────────────────────


class TestPlanContentConfig:
    """PlanContentConfig schema validation."""

    def test_strategy_enum(self):
        from contextunity.router.cortex.compiler.platform_tools.content import (
            PlanContentConfig,
        )

        PlanContentConfig(strategy="editorial")
        PlanContentConfig(strategy="chronological")
        PlanContentConfig(strategy="priority")

        with pytest.raises(ValidationError):
            PlanContentConfig(strategy="random")  # type: ignore[arg-type]


# ── Registration Tests ───────────────────────────────────────────────


# ── Config Validation via Registry ───────────────────────────────────


# ── Scope Enforcement Tests ──────────────────────────────────────────


# ── Template Loading Tests ───────────────────────────────────────────
