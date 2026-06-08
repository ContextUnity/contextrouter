"""Test fail-closed: SecureTool must refuse execution if attenuation fails.

Originally flagged that ``SecureTool._prepare_execution`` swallowed
attenuation errors with ``logger.warning``, which let tools run with the
un-attenuated parent token. This test pins the fail-closed contract: an
attenuation failure must raise SecurityError, never silently degrade.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from contextunity.core import ContextToken
from contextunity.core.exceptions import SecurityError
from contextunity.core.tokens import TokenBuilder
from langchain_core.tools import BaseTool

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.modules.tools.secure import SecureTool


class _Echo(BaseTool):
    name: str = "echo_fail"
    description: str = "echo"

    def _run(self, **kwargs: object) -> str:
        return "should-not-run"


def _parent_token() -> ContextToken:
    return ContextToken(
        token_id="t1",
        permissions=("tool:echo_fail:execute", "tool:echo_fail"),
        allowed_tenants=("a", "b"),
    )


def test_attenuation_failure_raises_security_error_not_warning() -> None:
    """When TokenBuilder.attenuate() raises, SecureTool.invoke() must raise SecurityError.

    The previous behavior was ``except Exception: logger.warning(...)`` which
    allowed the tool to run with the un-attenuated parent token, bypassing
    capability-stripping.
    """
    inner = _Echo()
    tool = SecureTool.wrap(
        inner,
        permission="tool:echo_fail",
        allowed_tenants=("a",),
    )
    parent = _parent_token()
    token_ref = set_current_access_token(parent)
    try:
        # TokenBuilder is imported inside _prepare_execution; patch the canonical
        # location so the local import picks up the mock.
        with patch.object(TokenBuilder, "attenuate", side_effect=ValueError("simulated failure")):
            with pytest.raises(SecurityError, match="attenuation failed"):
                tool.invoke({})
    finally:
        reset_current_access_token(token_ref)
