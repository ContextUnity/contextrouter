"""Minimal stubs for optional ``google-cloud-discoveryengine`` runtime dependency."""

from typing import Protocol

class RankedRecord(Protocol):
    id: str
    score: float

class RankResponse(Protocol):
    @property
    def records(self) -> list[RankedRecord]: ...

class RankServiceAsyncClient:
    def __init__(self) -> None: ...
    def ranking_config_path(self, *, project: str, location: str, ranking_config: str) -> str: ...
    async def rank(self, *, request: RankRequest) -> RankResponse: ...

class RankingRecord:
    def __init__(self, *, id: str, title: str, content: str) -> None: ...

class RankRequest:
    def __init__(
        self,
        *,
        ranking_config: str,
        model: str,
        top_n: int,
        query: str,
        records: list[RankingRecord],
    ) -> None: ...
