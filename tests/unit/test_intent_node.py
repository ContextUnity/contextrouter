from __future__ import annotations

import json

from contextunity.core.manifest.router import RetryPolicy
from langchain_core.messages import HumanMessage

from contextunity.router.cortex.compiler.platform_tools.intent import detect_intent
from contextunity.router.modules.models.types import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
)


class _StubLLM:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    @property
    def capabilities(self):
        return ModelCapabilities(supports_text=True, supports_image=False, supports_audio=False)

    async def generate(
        self, request: ModelRequest, *, retry_policy: RetryPolicy | None = None, **kwargs: object
    ) -> ModelResponse:
        _ = request, retry_policy, kwargs
        return ModelResponse(
            text=json.dumps(self._payload),
            raw_provider=ProviderInfo(provider="test", model_name="stub", model_key="test/stub"),
        )

    async def stream(self, request: ModelRequest, **kwargs: object):
        _ = request, kwargs
        raise NotImplementedError

    def get_token_count(self, text: str) -> int:
        return max(1, len(text) // 4)


def test_detect_intent_parses_llm_payload(monkeypatch) -> None:
    payload = {
        "intent": "rag",
        "ignore_history": False,
        "cleaned_query": "What is the mastermind principle?",
        "retrieval_queries": ["mastermind principle", "Think and Grow Rich mastermind"],
        "user_language": "en",
        "taxonomy_concepts": ["Mastermind Principle", "Success"],
    }

    monkeypatch.setattr(
        "contextunity.router.modules.models.registry.model_registry.create_llm",
        lambda *_a, **_kw: _StubLLM(payload),
    )

    state: dict = {
        "messages": [HumanMessage(content="What is the mastermind principle?")],
        "user_query": "",
    }
    import asyncio

    out = asyncio.run(detect_intent(state))
    dynamic = out["dynamic"]

    assert dynamic["intent"] == "rag"
    assert dynamic["intent_text"] == "What is the mastermind principle?"
    assert dynamic["should_retrieve"] is True
    assert dynamic["retrieval_queries"]
    assert "taxonomy_concepts" in dynamic


def test_detect_intent_handles_json_fenced_output(monkeypatch) -> None:
    fenced = "```json\n" + json.dumps({"intent": "identity"}) + "\n```"

    class _FencedLLM:
        @property
        def capabilities(self):
            return ModelCapabilities(supports_text=True, supports_image=False, supports_audio=False)

        async def generate(
            self,
            request: ModelRequest,
            *,
            retry_policy: RetryPolicy | None = None,
            **kwargs: object,
        ) -> ModelResponse:
            _ = request, retry_policy, kwargs
            return ModelResponse(
                text=fenced,
                raw_provider=ProviderInfo(
                    provider="test", model_name="stub", model_key="test/stub"
                ),
            )

        async def stream(self, request: ModelRequest, **kwargs: object):
            _ = request, kwargs
            raise NotImplementedError

        def get_token_count(self, text: str) -> int:
            return max(1, len(text) // 4)

    monkeypatch.setattr(
        "contextunity.router.modules.models.registry.model_registry.create_llm",
        lambda *_a, **_kw: _FencedLLM(),
    )

    state: dict = {"messages": [HumanMessage(content="Who are you?")]}
    import asyncio

    out = asyncio.run(detect_intent(state))
    assert out["dynamic"]["intent"] == "identity"


def test_detect_intent_routes_to_sql_analytics(monkeypatch) -> None:
    fenced = "```json\n" + json.dumps({"intent": "sql"}) + "\n```"

    class _SqlLLM:
        @property
        def capabilities(self):
            return ModelCapabilities(supports_text=True, supports_image=False, supports_audio=False)

        async def generate(
            self,
            request: ModelRequest,
            *,
            retry_policy: RetryPolicy | None = None,
            **kwargs: object,
        ) -> ModelResponse:
            _ = request, retry_policy, kwargs
            return ModelResponse(
                text=fenced,
                raw_provider=ProviderInfo(
                    provider="test", model_name="stub", model_key="test/stub"
                ),
            )

        async def stream(self, request: ModelRequest, **kwargs: object):
            _ = request, kwargs
            raise NotImplementedError

        def get_token_count(self, text: str) -> int:
            return max(1, len(text) // 4)

    monkeypatch.setattr(
        "contextunity.router.modules.models.registry.model_registry.create_llm",
        lambda *_a, **_kw: _SqlLLM(),
    )

    state: dict = {
        "messages": [HumanMessage(content="how many patients were admitted last week in SQL?")],
        "config": {
            "data_sources": [
                {"type": "vector", "binding": "router_retrieve"},
                {"type": "sql", "binding": "router_sql_plan"},
            ]
        },
    }
    import asyncio

    out = asyncio.run(detect_intent(state))
    assert out["dynamic"]["intent"] == "sql"
    assert out["dynamic"]["intent_route"] == "sql_analytics"
