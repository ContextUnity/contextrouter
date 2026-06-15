"""Tests for Router → Brain service token tenant scoping."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from contextunity.core.exceptions import SecurityError

from contextunity.router.core.brain_token import get_brain_service_token


class TestGetBrainServiceToken:
    def test_empty_allowed_tenants_raises_security_error(self):
        with pytest.raises(SecurityError, match="explicit allowed_tenants"):
            get_brain_service_token(allowed_tenants=())

    def test_whitespace_only_tenants_raises_security_error(self):
        with pytest.raises(SecurityError, match="explicit allowed_tenants"):
            get_brain_service_token(allowed_tenants=("  ", ""))

    def test_explicit_tenants_returns_token(self):
        token = get_brain_service_token(allowed_tenants=("default",))
        assert token.allowed_tenants == ("default",)
        assert token.has_permission("brain:read")

    def test_call_sites_pass_allowed_tenants(self):
        router_src = Path(__file__).resolve().parents[2] / "src" / "contextunity" / "router"
        offenders: list[str] = []
        pattern = re.compile(r"get_brain_service_token\s*\((?!.*allowed_tenants)")
        for path in router_src.rglob("*.py"):
            if path.name == "brain_token.py":
                continue
            text = path.read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(router_src)}:{line_no}")
        assert offenders == [], "Missing allowed_tenants= at: " + ", ".join(offenders)
