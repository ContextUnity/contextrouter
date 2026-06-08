"""Executor functions for Brain platform tools."""

from typing import Protocol, TypeGuard

from contextunity.core.exceptions import PlatformServiceError, SecurityError
from contextunity.core.types import ContextUnitPayload, JsonDict, is_json_dict

from contextunity.router.cortex.compiler.state_routing import read_state_input
from contextunity.router.cortex.types import GraphState

from ..helpers.base import resolve_tenant_and_token
from ..helpers.contracts import PlatformResult
from ..helpers.state import get_last_message_text
from .client import BrainKnowledgeGraphClient, get_brain_client
from .schemas import (
    BrainBlackboardReadConfig,
    BrainBlackboardWriteConfig,
    BrainKGQueryConfig,
    BrainMemoryReadConfig,
    BrainMemoryWriteConfig,
    BrainSearchConfig,
    BrainUpsertConfig,
)

_get_brain_client = get_brain_client


class BrainBlackboardWriteFn(Protocol):
    """Brain SDK blackboard write hook (optional client extension)."""

    async def __call__(
        self,
        *,
        tenant_id: str,
        scope_path: str,
        content: ContextUnitPayload,
        ttl_seconds: int | None = None,
        created_by: str | None = None,
    ) -> object: ...


class BrainBlackboardReadFn(Protocol):
    """Brain SDK blackboard read hook (optional client extension)."""

    async def __call__(
        self,
        *,
        ids: list[str],
        tenant_id: str,
    ) -> object: ...


def _is_blackboard_write_fn(method: object) -> TypeGuard[BrainBlackboardWriteFn]:
    return callable(method)


def _is_blackboard_read_fn(method: object) -> TypeGuard[BrainBlackboardReadFn]:
    return callable(method)


def _require_blackboard_write(method: object) -> BrainBlackboardWriteFn:
    """Narrow optional blackboard write callables at the Brain client boundary."""
    if not _is_blackboard_write_fn(method):
        raise PlatformServiceError(
            message="BrainClient does not yet support write_blackboard",
            service_name="brain",
        )
    return method


def _require_blackboard_read(method: object) -> BrainBlackboardReadFn:
    """Narrow optional blackboard read callables at the Brain client boundary."""
    if not _is_blackboard_read_fn(method):
        raise PlatformServiceError(
            message="BrainClient does not yet support read_blackboard",
            service_name="brain",
        )
    return method


async def brain_search_executor(state: GraphState, config: BrainSearchConfig) -> PlatformResult:
    """Execute brain vector search."""
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_search")

    client = _get_brain_client(tenant_id, access_token)

    query_val = read_state_input(state, "query") or read_state_input(state, "search_query")
    if isinstance(query_val, str) and query_val:
        query = query_val
    else:
        query = get_last_message_text(state)

    results = await client.search(
        tenant_id=tenant_id,
        query_text=query,
        limit=config.top_k,
        source_types=[config.collection] if config.collection else [],
    )

    return {"results": results, "query": query, "top_k": config.top_k}


async def brain_memory_read_executor(
    state: GraphState, config: BrainMemoryReadConfig
) -> PlatformResult:
    """Read episodic memory and user facts."""
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_memory_read")
    user_id = access_token.user_id
    if not user_id:
        raise SecurityError("brain_memory_read requires a user-bound token")
    if config.user_id is not None and config.user_id != user_id:
        raise SecurityError("brain_memory_read user_id override must match the access token user")

    client = _get_brain_client(tenant_id, access_token)

    episodes = await client.get_recent_episodes(
        tenant_id=tenant_id,
        user_id=user_id,
        limit=config.last_n,
    )

    facts = await client.get_user_facts(
        tenant_id=tenant_id,
        user_id=user_id,
    )

    return {
        "episodes": episodes,
        "facts": facts,
        "user_id": user_id,
    }


async def brain_memory_write_executor(
    state: GraphState, config: BrainMemoryWriteConfig
) -> PlatformResult:
    """Write episodic memory."""
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_memory_write")
    user_id = access_token.user_id
    if not user_id:
        raise SecurityError("brain_memory_write requires a user-bound token")
    if config.user_id is not None and config.user_id != user_id:
        raise SecurityError("brain_memory_write user_id override must match the access token user")
    session_id = state.get("session_id", "")

    final = state.get("final_output")
    content = str(final) if final else get_last_message_text(state)

    client = _get_brain_client(tenant_id, access_token)

    episode_id = await client.add_episode(
        tenant_id=tenant_id,
        user_id=user_id,
        content=content,
        session_id=session_id,
    )

    return {
        "success": True,
        "episode_id": episode_id,
        "user_id": user_id,
        "memory_scope": config.memory_scope,
    }


async def brain_blackboard_write_executor(
    state: GraphState, config: BrainBlackboardWriteConfig
) -> PlatformResult:
    """Write data to the blackboard (pass-by-reference).

    NOTE: Requires BrainClient.write_blackboard() — planned SDK extension.
    """
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_blackboard_write")

    content: ContextUnitPayload = {}
    for key in ("classification", "analysis"):
        val = read_state_input(state, key)
        if val is not None:
            if is_json_dict(val):
                content = dict(val)
            else:
                content = {"data": val}
            break

    if not content:
        final = state.get("final_output")
        if is_json_dict(final):
            content = {str(k): v for k, v in final.items()}

    if not content:
        content = {
            k: v
            for k, v in dict(state).items()
            if not k.startswith("__") and k not in ("messages", "tenant_id")
        }

    client_obj: object = _get_brain_client(tenant_id, access_token)
    write_blackboard = _require_blackboard_write(getattr(client_obj, "write_blackboard", None))

    payload_obj = await write_blackboard(
        tenant_id=tenant_id,
        scope_path=config.scope_path,
        content=content,
        ttl_seconds=config.ttl_seconds,
        created_by=config.created_by,
    )

    if is_json_dict(payload_obj):
        return dict(payload_obj)
    return {}


async def brain_blackboard_read_executor(
    state: GraphState, config: BrainBlackboardReadConfig
) -> PlatformResult:
    """Read blackboard records by UUID.

    NOTE: Requires BrainClient.read_blackboard() — planned SDK extension.
    """
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_blackboard_read")

    client_obj: object = _get_brain_client(tenant_id, access_token)
    read_blackboard = _require_blackboard_read(getattr(client_obj, "read_blackboard", None))

    payload_obj = await read_blackboard(
        ids=config.ids,
        tenant_id=tenant_id,
    )
    if is_json_dict(payload_obj):
        return dict(payload_obj)
    return {}


async def brain_kg_query_executor(state: GraphState, config: BrainKGQueryConfig) -> PlatformResult:
    """Query the knowledge graph.

    NOTE: Requires BrainClient.query_kg() — planned SDK extension.
    """
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_kg_query")

    entity_val = read_state_input(state, "entity")
    entity = entity_val if isinstance(entity_val, str) else (config.entity or "")

    client = _get_brain_client(tenant_id, access_token)
    if not isinstance(client, BrainKnowledgeGraphClient):
        raise PlatformServiceError(
            message="BrainClient does not yet support query_kg",
            service_name="brain",
        )

    results = await client.query_kg(
        tenant_id=tenant_id,
        entity=entity,
        direction=config.direction,
        depth=config.depth,
    )

    return {"results": results, "entity": entity}


async def brain_upsert_executor(state: GraphState, config: BrainUpsertConfig) -> PlatformResult:
    """Upsert a document into Brain."""
    tenant_id, access_token = resolve_tenant_and_token(state, binding="brain_upsert")

    final = state.get("final_output")
    content = str(final) if final else ""

    client = _get_brain_client(tenant_id, access_token)

    upsert_metadata: JsonDict | None = None
    if config.metadata_schema is not None and is_json_dict(config.metadata_schema):
        upsert_metadata = dict(config.metadata_schema)

    result = await client.upsert(
        tenant_id=tenant_id,
        content=content,
        source_type=config.collection,
        metadata=upsert_metadata,
    )

    return {"success": True, "result": result}


# Backward-compatible aliases for unit tests and legacy imports.
_brain_search_executor = brain_search_executor
_brain_memory_read_executor = brain_memory_read_executor
_brain_memory_write_executor = brain_memory_write_executor
_brain_kg_query_executor = brain_kg_query_executor
_brain_upsert_executor = brain_upsert_executor


__all__ = [
    "brain_blackboard_read_executor",
    "brain_blackboard_write_executor",
    "brain_kg_query_executor",
    "brain_memory_read_executor",
    "brain_memory_write_executor",
    "brain_search_executor",
    "brain_upsert_executor",
]
