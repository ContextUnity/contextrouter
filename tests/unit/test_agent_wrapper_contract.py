from __future__ import annotations

import inspect

import pytest


@pytest.mark.anyio
async def test_agent_wrappers_return_dict_not_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent-mode wrappers must always return a dict partial-update (never a coroutine).

    This guards the common pitfall: forgetting to `await` an async step inside an agent wrapper.
    """

    # Patch the underlying step callables that wrappers delegate to.
    from contextrouter.cortex.nodes.rag_retrieval import (
        extract as extract_mod,
    )
    from contextrouter.cortex.nodes.rag_retrieval import (
        generate as generate_mod,
    )
    from contextrouter.cortex.nodes.rag_retrieval import (
        intent as intent_mod,
    )
    from contextrouter.cortex.nodes.rag_retrieval import (
        retrieve as retrieve_mod,
    )
    from contextrouter.cortex.nodes.rag_retrieval import (
        suggest as suggest_mod,
    )

    def _sync_update(_: object) -> dict[str, object]:
        return {"ok": True}

    async def _async_update(_: object) -> dict[str, object]:
        return {"ok": True}

    monkeypatch.setattr(extract_mod, "_extract_user_query", _sync_update, raising=True)
    monkeypatch.setattr(intent_mod, "_detect_intent", _async_update, raising=True)
    monkeypatch.setattr(retrieve_mod, "_retrieve_documents", _async_update, raising=True)
    monkeypatch.setattr(generate_mod, "_generate_response", _async_update, raising=True)
    monkeypatch.setattr(suggest_mod, "_generate_search_suggestions", _async_update, raising=True)

    from contextrouter.core import agent_registry

    agent_names = [
        "extract_query",
        "detect_intent",
        "retrieve",
        "suggest",
        "generate",
        "routing",
    ]
    for name in agent_names:
        cls = agent_registry.get(name)
        agent = cls(None)
        res = await agent.process({})  # type: ignore[arg-type]
        assert isinstance(res, dict), f"{name} returned {type(res)} instead of dict"
        assert not inspect.isawaitable(res), f"{name} returned an awaitable (missing await?)"
