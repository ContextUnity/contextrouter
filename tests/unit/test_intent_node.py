from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from contextrouter.cortex.steps.rag_retrieval.intent import detect_intent


class _StubLLM:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def invoke(self, _messages):  # matches ChatGoogleGenerativeAI.invoke signature usage
        return AIMessage(content=json.dumps(self._payload))


class _LLMWrapper:
    def __init__(self, chat_model):
        self._chat = chat_model

    def as_chat_model(self):
        return self._chat


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
        lambda *_a, **_kw: _LLMWrapper(_StubLLM(payload)),
    )

    from contextrouter.cortex.state import AgentState

    state: AgentState = {
        "messages": [HumanMessage(content="What is the mastermind principle?")],
        "user_query": "",
    }
    out = detect_intent(state)

    assert out["intent"] == "rag_and_web"
    assert out["intent_text"] == "What is the mastermind principle?"
    assert out["should_retrieve"] is True
    assert out["retrieval_queries"]
    assert "taxonomy_concepts" in out


def test_detect_intent_handles_json_fenced_output(monkeypatch) -> None:
    fenced = "```json\n" + json.dumps({"intent": "identity"}) + "\n```"

    class _FencedLLM:
        def invoke(self, _messages):
            return AIMessage(content=fenced)

    monkeypatch.setattr(
        "contextrouter.modules.models.registry.model_registry.create_llm",
        lambda *_a, **_kw: _LLMWrapper(_FencedLLM()),
    )

    from contextrouter.cortex.state import AgentState

    state: AgentState = {"messages": [HumanMessage(content="Who are you?")]}
    out = detect_intent(state)
    assert out["intent"] == "identity"
