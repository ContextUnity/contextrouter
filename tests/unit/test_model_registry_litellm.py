from __future__ import annotations

import pytest

from contextrouter.core.config import Config
from contextrouter.modules.models.base import BaseLLM
from contextrouter.modules.models.registry import model_registry


def _cfg() -> Config:
    # Minimal config required by VertexLLM constructor (it validates these early).
    return Config.model_validate(
        {
            "vertex": {"project_id": "test-project", "location": "us-central1"},
            "models": {"default_llm": "vertex/gemini-2.5-flash"},
        }
    )


def test_model_registry_lazy_vertex_llm() -> None:
    cfg = _cfg()
    try:
        llm = model_registry.create_llm("vertex/gemini-2.5-flash-lite", config=cfg, streaming=False)
    except ImportError as e:
        # Skip if vertex dependencies are not available or incompatible
        pytest.skip(f"vertex dependencies not available or incompatible: {e}")
    assert isinstance(llm, BaseLLM)
    assert type(llm).__name__ == "VertexLLM"


def test_model_registry_litellm_key_routes_to_provider_without_eager_optional_import() -> None:
    cfg = _cfg()
    llm = model_registry.create_llm("litellm/openai/gpt-4o-mini", config=cfg, streaming=False)
    assert isinstance(llm, BaseLLM)
    assert type(llm).__name__ == "LiteLLMLLM"

    # Optional dependency is imported only when the chat model is requested.
    try:
        _ = llm.as_chat_model()
    except ImportError:
        # Expected in base install where litellm extras are not installed.
        return
    except Exception as e:
        pytest.fail(f"Unexpected error from as_chat_model(): {e}")
