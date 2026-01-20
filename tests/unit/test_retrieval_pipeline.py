from __future__ import annotations

import asyncio

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import IRead
from contextrouter.cortex.models import Citation, RetrievedDoc
from contextrouter.modules.retrieval.rag import RagPipeline


def test_retrieval_pipeline_returns_empty_on_empty_query(monkeypatch) -> None:
    from contextrouter.cortex.state import AgentState

    state: AgentState = {"user_query": ""}
    res = asyncio.run(RagPipeline().execute(state))
    assert res.retrieved_docs == []
    assert res.citations == []


def test_retrieval_pipeline_calls_vertex_and_builds_citations(monkeypatch) -> None:
    from contextrouter.core.registry import ComponentFactory

    calls: list[tuple[str, int, str]] = []

    class _Provider(IRead):
        async def read(self, query: str, *, limit: int = 5, filters=None, token=None):
            st = (filters or {}).get("source_type", "book")
            calls.append((query, limit, st))
            return [
                BisquitEnvelope(
                    content=RetrievedDoc(source_type=st, content=f"{st}:{query}", title="t")
                )
            ]

    async def _identity_rerank(**kw):
        return kw["documents"]

    class MockReranker:
        async def rerank(self, query, documents, top_n=None):
            return documents[:top_n] if top_n else documents

    # Mock ComponentFactory.create_provider to return our test provider
    monkeypatch.setattr(ComponentFactory, "create_provider", lambda name, **kwargs: _Provider())
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

    from contextrouter.core.tokens import BiscuitToken
    from contextrouter.cortex.state import AgentState

    state: AgentState = {
        "user_query": "hello",
        "retrieval_queries": ["hello"],
        "access_token": BiscuitToken(token_id="test-token", permissions=("RAG_READ",)),
    }
    res = asyncio.run(RagPipeline().execute(state))

    assert res.graph_facts == ["f1"]
    assert res.retrieved_docs
    assert res.citations
    assert calls
