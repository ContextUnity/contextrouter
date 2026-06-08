from __future__ import annotations

import pytest
from contextunity.core import ContextToken
from contextunity.core.exceptions import ConfigurationError, SecurityError

from contextunity.router.service.mixins.execution.dispatcher import (
    _authorized_dispatcher_project_config,
    _response_messages_from_result,
)


def test_dispatcher_rejects_foreign_project_from_request_metadata() -> None:
    configs = {
        "victim": {
            "project_id": "victim",
            "allowed_tenants": ["victim-tenant"],
        }
    }
    token = ContextToken(token_id="attacker", allowed_tenants=("attacker-tenant",))

    with pytest.raises(SecurityError):
        _authorized_dispatcher_project_config(configs, token, "attacker-tenant", "victim")


def test_dispatcher_rejects_unknown_explicit_project() -> None:
    token = ContextToken(token_id="caller", allowed_tenants=("tenant-a",))

    with pytest.raises(ConfigurationError):
        _authorized_dispatcher_project_config({}, token, "tenant-a", "missing")


def test_dispatcher_response_omits_checkpoint_history() -> None:
    result = {
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
            {"role": "assistant", "content": "new answer"},
        ]
    }

    assert _response_messages_from_result(result, [{"role": "user", "content": "current"}]) == [
        {"role": "assistant", "content": "new answer"}
    ]
