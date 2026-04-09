import pytest
from contextcore.tokens import TokenBuilder

from contextrouter.cortex.graphs.secure_node import make_secure_node
from contextrouter.cortex.runtime_context import (
    get_accumulated_provenance,
    init_provenance_accumulator,
    reset_current_access_token,
    reset_provenance_accumulator,
    set_current_access_token,
)


@pytest.fixture
def base_token():
    return TokenBuilder().mint_root(
        user_ctx={},
        permissions=[
            "router:execute",
            "shield:secrets:read",
            "shield:secrets:read:default/api_keys/planner/model_secret_ref",
        ],
        ttl_s=3600,
        allowed_tenants=["default"],
    )


def dummy_node(state: dict) -> dict:
    return {"status": "ok"}


async def dummy_async_node(state: dict) -> dict:
    return {"status": "async_ok"}


def test_secure_node_logs_provenance(base_token):
    """Test that execution logs 'node:name' and 'shield:secrets:*' to provenance."""
    secure_node = make_secure_node(
        "planner", dummy_node, requires_llm=True, model_secret_ref="openai-key"
    )

    token_ref = set_current_access_token(base_token)
    accum_ref = init_provenance_accumulator()

    try:
        result = secure_node({"state": "init"})
        assert result == {"status": "ok"}

        history = get_accumulated_provenance()
        assert len(history) == 2
        assert history[0] == "node:planner"
        assert history[1] == "shield:secrets:read:default/api_keys/planner/model_secret_ref"
    finally:
        reset_provenance_accumulator(accum_ref)
        reset_current_access_token(token_ref)


@pytest.mark.asyncio
async def test_async_secure_node_logs_provenance(base_token):
    """Test that async execution logs 'node:name' and 'shield:secrets:*' to provenance."""
    secure_node = make_secure_node(
        "planner", dummy_async_node, requires_llm=True, model_secret_ref="openai-key"
    )

    token_ref = set_current_access_token(base_token)
    accum_ref = init_provenance_accumulator()

    try:
        result = await secure_node({"state": "init"})
        assert result == {"status": "async_ok"}

        history = get_accumulated_provenance()
        assert len(history) == 2
        assert history[0] == "node:planner"
        assert history[1] == "shield:secrets:read:default/api_keys/planner/model_secret_ref"
    finally:
        reset_provenance_accumulator(accum_ref)
        reset_current_access_token(token_ref)


def test_secure_node_no_llm_provenance(base_token):
    """Test that if requires_llm=False, shield secrets are NOT logged to provenance."""
    secure_node = make_secure_node(
        "worker",
        dummy_node,
        requires_llm=False,
    )

    token_ref = set_current_access_token(base_token)
    accum_ref = init_provenance_accumulator()

    try:
        secure_node({"state": "init"})

        history = get_accumulated_provenance()
        assert len(history) == 1
        assert history[0] == "node:worker"
    finally:
        reset_provenance_accumulator(accum_ref)
        reset_current_access_token(token_ref)
