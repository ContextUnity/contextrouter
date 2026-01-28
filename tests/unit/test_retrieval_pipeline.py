from __future__ import annotations

import asyncio

from contextrouter.modules.retrieval.rag import RagPipeline
from contextrouter.modules.retrieval.rag.models import Citation, RetrievedDoc


def test_retrieval_pipeline_returns_empty_on_empty_query(monkeypatch) -> None:
    from contextrouter.cortex.state import AgentState

    state: AgentState = {"user_query": ""}
    res = asyncio.run(RagPipeline().execute(state))
    assert res.retrieved_docs == []
    assert res.citations == []


def test_retrieval_pipeline_calls_vertex_and_builds_citations(monkeypatch) -> None:
    from contextrouter.modules.retrieval.pipeline import BaseRetrievalPipeline

    calls: list[tuple[str, int]] = []

    class MockResult:
        def __init__(self):
            doc = RetrievedDoc(source_type="book", content="book:hello", title="t")

            # Create a mock ContextUnit-like object with content attribute
            class MockUnit:
                def __init__(self):
                    self.content = doc

            self.units = [MockUnit()]

    async def mock_execute(
        self, query, *, limit: int = 5, filters=None, token=None, providers=None
    ):
        calls.append((str(query), limit))
        from contextrouter.modules.retrieval.pipeline import PipelineResult

        return PipelineResult(units=MockResult().units)

    class MockReranker:
        async def rerank(self, query, documents, top_n=None):
            return documents[:top_n] if top_n else documents

    # Mock BaseRetrievalPipeline.execute to return our test result
    monkeypatch.setattr(BaseRetrievalPipeline, "execute", mock_execute)
    monkeypatch.setattr(RagPipeline, "_should_run_web", lambda _s, _state: False)
    monkeypatch.setattr(RagPipeline, "_get_graph_facts", lambda _s, _state: ["f1"])
    monkeypatch.setattr(
        "contextrouter.modules.retrieval.rag.pipeline.get_reranker",
        lambda **kwargs: MockReranker(),
    )
    monkeypatch.setattr(
        "contextrouter.modules.retrieval.rag.pipeline.build_citations",
        lambda docs, **_kw: (
            [Citation(source_type=docs[0].source_type, title="t", content="c")] if docs else []
        ),
    )

    from contextrouter.core.tokens import ContextToken
    from contextrouter.cortex.state import AgentState

    state: AgentState = {
        "user_query": "hello",
        "retrieval_queries": ["hello"],
        "access_token": ContextToken(token_id="test-token", permissions=("RAG_READ",)),
    }
    res = asyncio.run(RagPipeline().execute(state))

    assert res.graph_facts == ["f1"]
    assert res.retrieved_docs
    assert res.citations
    assert calls
