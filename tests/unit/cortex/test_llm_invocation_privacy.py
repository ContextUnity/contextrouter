"""LLM-node prompt privacy tests."""

from __future__ import annotations

from typing import cast

import pytest
from contextunity.core.tokens import TokenBuilder

from contextunity.router.core.context import reset_current_access_token, set_current_access_token
from contextunity.router.cortex.compiler.node_executors import llm_invocation
from contextunity.router.modules.models.base import BaseLLM
from contextunity.router.modules.models.types import (
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    TextPart,
)


@pytest.mark.asyncio
async def test_generate_with_node_privacy_masks_provider_request(monkeypatch):
    token = TokenBuilder().mint_root(
        user_ctx={"user_id": "test-user"},
        permissions=["privacy:anonymize", "privacy:deanonymize"],
        ttl_s=3600,
        allowed_tenants=["default"],
    )
    token_ref = set_current_access_token(token)
    captured: dict[str, ModelRequest] = {}

    async def fake_model_telemetry(
        llm: BaseLLM,
        request: ModelRequest,
        config: object,
        **kwargs: object,
    ) -> ModelResponse:
        captured["request"] = request
        text_part = request.parts[0]
        assert isinstance(text_part, TextPart)
        return ModelResponse(
            text=text_part.text,
            raw_provider=ProviderInfo(
                provider="fake",
                model_name="fake-model",
                model_key="fake/model",
            ),
        )

    monkeypatch.setattr(llm_invocation, "model_telemetry", fake_model_telemetry)

    try:
        response = await llm_invocation.generate_with_node_privacy(
            cast(BaseLLM, object()),
            ModelRequest(parts=[TextPart(text="Лікар Іван Петренко, john@example.com")]),
            None,
            node_name="planner",
            state={"metadata": {"session_id": "test-session"}},
        )
    finally:
        reset_current_access_token(token_ref)

    sent_part = captured["request"].parts[0]
    assert isinstance(sent_part, TextPart)
    assert "Іван Петренко" not in sent_part.text
    assert "john@example.com" not in sent_part.text
    assert "Іван Петренко" in response.text
    assert "john@example.com" in response.text
