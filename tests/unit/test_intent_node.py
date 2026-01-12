from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from contextrouter.cortex.steps.rag_retrieval.intent import detect_intent
from contextrouter.modules.models.types import (
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

    async def generate(self, request: ModelRequest, *, token=None) -> ModelResponse:
        _ = request, token
        return ModelResponse(
            text=json.dumps(self._payload),
            raw_provider=ProviderInfo(provider="test", model_name="stub", model_key="test/stub"),
        )

    async def stream(self, request: ModelRequest, *, token=None):
        _ = request, token
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
        "contextrouter.modules.models.registry.model_registry.create_llm",
        lambda *_a, **_kw: _StubLLM(payload),
    )

    from contextrouter.cortex.state import AgentState

    state: AgentState = {
        "messages": [HumanMessage(content="What is the mastermind principle?")],
        "user_query": "",
    }
    import asyncio

    out = asyncio.run(detect_intent(state))

    assert out["intent"] == "rag_and_web"
    assert out["intent_text"] == "What is the mastermind principle?"
    assert out["should_retrieve"] is True
    assert out["retrieval_queries"]
    assert "taxonomy_concepts" in out


def test_detect_intent_handles_json_fenced_output(monkeypatch) -> None:
    fenced = "```json\n" + json.dumps({"intent": "identity"}) + "\n```"

    class _FencedLLM:
        @property
        def capabilities(self):
            return ModelCapabilities(supports_text=True, supports_image=False, supports_audio=False)

        async def generate(self, request: ModelRequest, *, token=None) -> ModelResponse:
            _ = request, token
            return ModelResponse(
                text=fenced,
                raw_provider=ProviderInfo(
                    provider="test", model_name="stub", model_key="test/stub"
                ),
            )

        async def stream(self, request: ModelRequest, *, token=None):
            _ = request, token
            raise NotImplementedError

        def get_token_count(self, text: str) -> int:
            return max(1, len(text) // 4)

    monkeypatch.setattr(
        "contextrouter.modules.models.registry.model_registry.create_llm",
        lambda *_a, **_kw: _FencedLLM(),
    )

    from contextrouter.cortex.state import AgentState

    state: AgentState = {"messages": [HumanMessage(content="Who are you?")]}
    import asyncio

    out = asyncio.run(detect_intent(state))
    assert out["intent"] == "identity"
