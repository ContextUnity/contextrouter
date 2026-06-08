"""Unit tests for ``_intersect_tenant_with_project``.

These tests pin the tenant ∩ project allowed_tenants intersection contract
introduced to prevent multi-tenant token scope confusion when projects
have distinct ``allowed_tenants`` in the manifest.
"""

from __future__ import annotations

import pytest
from contextunity.core import ContextToken
from contextunity.core.exceptions import SecurityError

from contextunity.router.service.mixins.execution.helpers import (
    _intersect_tenant_with_project,
)


def _parent_token(*tenants: str) -> ContextToken:
    return ContextToken(
        token_id="t1",
        permissions=("tool:read",),
        allowed_tenants=tenants,
    )


def test_intersect_returns_tenant_id_when_member_of_project_tenants() -> None:
    project_config: dict[str, object] = {"allowed_tenants": ["nszu", "nszu-staging"]}
    token = _parent_token("nszu", "nszu-staging")
    assert _intersect_tenant_with_project(token, "nszu", project_config) == "nszu"


def test_intersect_returns_first_overlapping_tenant_when_initial_misses() -> None:
    project_config: dict[str, object] = {"allowed_tenants": ["nszu-staging"]}
    token = _parent_token("nszu", "nszu-staging")
    assert _intersect_tenant_with_project(token, "nszu", project_config) == "nszu-staging"


def test_intersect_raises_when_token_outside_project_scope() -> None:
    project_config: dict[str, object] = {"allowed_tenants": ["other-tenant"]}
    token = _parent_token("nszu", "nszu-staging")
    with pytest.raises(SecurityError, match="not covered by"):
        _intersect_tenant_with_project(token, "nszu", project_config)


def test_intersect_falls_back_to_project_id_when_no_tenants_declared() -> None:
    project_config: dict[str, object] = {"project_id": "my-project"}
    token = _parent_token("my-project")
    assert _intersect_tenant_with_project(token, "my-project", project_config) == "my-project"


def test_intersect_raises_on_none_token() -> None:
    project_config: dict[str, object] = {"project_id": "my-project"}
    with pytest.raises(SecurityError, match="valid token"):
        _intersect_tenant_with_project(None, "my-project", project_config)


def test_intersect_raises_on_token_without_tenants() -> None:
    project_config: dict[str, object] = {"project_id": "my-project"}
    token = ContextToken(
        token_id="t1",
        permissions=("tool:read",),
        allowed_tenants=(),
    )
    with pytest.raises(SecurityError, match="no allowed_tenants"):
        _intersect_tenant_with_project(token, "my-project", project_config)
