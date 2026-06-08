"""FlowManager wiring tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from contextunity.core import ContextUnit

from contextunity.router.core.flow_manager import FlowConfig, FlowManager


@pytest.mark.asyncio
async def test_flow_manager_passes_logic_params_via_configure():
    """logic_params must reach transformer.configure(), not __init__ kwargs."""
    captured: dict[str, object] = {}

    class _StubTransformer:
        def __init__(self) -> None:
            self.configured_with: dict[str, object] | None = None

        def configure(self, params: dict[str, object] | None) -> None:
            self.configured_with = params

        async def transform(self, unit: ContextUnit) -> ContextUnit:
            return unit

    def _fake_create_transformer(name: str, **kwargs: object) -> _StubTransformer:
        transformer = _StubTransformer()
        if kwargs:
            transformer.configure(dict(kwargs))
            captured["params"] = dict(kwargs)
        return transformer

    class _StubConnector:
        async def connect(self):
            yield ContextUnit(payload={"content": "x"})

    flow = FlowConfig(
        source="web",
        logic=["keyphrases"],
        logic_params={"keyphrases": {"max_phrases": 3}},
        sink="response",
    )

    with (
        patch(
            "contextunity.router.core.flow_manager.ComponentFactory.create_connector",
            return_value=_StubConnector(),
        ),
        patch(
            "contextunity.router.core.flow_manager.ComponentFactory.create_transformer",
            side_effect=_fake_create_transformer,
        ),
    ):
        manager = FlowManager(
            connector_registry=MagicMock(),
            transformer_registry=MagicMock(),
            provider_registry=MagicMock(),
        )
        token = MagicMock()
        token.token_id = "tok-1"
        await manager.run(flow, token=token)

    assert captured["params"] == {"max_phrases": 3}
