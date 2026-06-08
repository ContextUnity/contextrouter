"""Typed boundaries for Vertex AI Ranking API."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, TypeGuard


class RankServiceAsyncClient(Protocol):
    def ranking_config_path(self, *, project: str, location: str, ranking_config: str) -> str: ...

    async def rank(self, *, request: object) -> RankResponse: ...


class RankedRecord(Protocol):
    id: str
    score: float


class RankResponse(Protocol):
    records: list[RankedRecord]


class RankingRecord(Protocol):
    def __init__(self, *, id: str, title: str, content: str) -> None: ...


class RankRequest(Protocol):
    def __init__(
        self,
        *,
        ranking_config: str,
        model: str,
        top_n: int,
        query: str,
        records: list[RankingRecord],
    ) -> None: ...


def _discovery_engine_module() -> object:
    try:
        return import_module("google.cloud.discoveryengine_v1")
    except ImportError as exc:
        from contextunity.core.exceptions import ConfigurationError

        raise ConfigurationError(
            "google-cloud-discoveryengine is not installed",
        ) from exc


def _is_rank_service_client(value: object) -> TypeGuard[RankServiceAsyncClient]:
    rank = getattr(value, "rank", None)
    ranking_config_path = getattr(value, "ranking_config_path", None)
    return callable(rank) and callable(ranking_config_path)


def _is_ranking_record_type(value: object) -> TypeGuard[type[RankingRecord]]:
    return callable(value)


def _is_rank_request_type(value: object) -> TypeGuard[type[RankRequest]]:
    return callable(value)


def load_rank_client() -> RankServiceAsyncClient:
    """Load and construct a Ranking API client."""
    from contextunity.core.exceptions import ConfigurationError

    module = _discovery_engine_module()
    client_cls = getattr(module, "RankServiceAsyncClient", None)
    if not callable(client_cls):
        raise ConfigurationError("RankServiceAsyncClient is unavailable")
    client = client_cls()
    if not _is_rank_service_client(client):
        raise ConfigurationError("RankServiceAsyncClient is unavailable")
    return client


def load_ranking_record_factory() -> type[RankingRecord]:
    """Return the ``RankingRecord`` message constructor."""
    from contextunity.core.exceptions import ConfigurationError

    module = _discovery_engine_module()
    record_cls = getattr(module, "RankingRecord", None)
    if not _is_ranking_record_type(record_cls):
        raise ConfigurationError("RankingRecord is unavailable")
    return record_cls


def load_rank_request_factory() -> type[RankRequest]:
    """Return the ``RankRequest`` message constructor."""
    from contextunity.core.exceptions import ConfigurationError

    module = _discovery_engine_module()
    request_cls = getattr(module, "RankRequest", None)
    if not _is_rank_request_type(request_cls):
        raise ConfigurationError("RankRequest is unavailable")
    return request_cls


__all__ = [
    "RankRequest",
    "RankResponse",
    "RankServiceAsyncClient",
    "RankedRecord",
    "RankingRecord",
    "load_rank_client",
    "load_rank_request_factory",
    "load_ranking_record_factory",
]
