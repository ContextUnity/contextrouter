"""Config schemas for Brain platform tools."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.types import JsonMetadata


class BrainSearchConfig(BaseModel, frozen=True):
    """Config for brain_search tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    collection: str = "default"
    rerank: bool = False
    rerank_model: str | None = None
    filter_metadata: JsonMetadata | None = None


class BrainMemoryReadConfig(BaseModel, frozen=True):
    """Config for brain_memory_read tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    last_n: int = Field(default=5, ge=1, le=50)
    memory_scope: str = "default"
    user_id: str | None = None


class BrainMemoryWriteConfig(BaseModel, frozen=True):
    """Config for brain_memory_write tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    memory_scope: str = "default"
    user_id: str | None = None
    ttl: int | None = None


class BrainBlackboardWriteConfig(BaseModel, frozen=True):
    """Config for brain_blackboard_write tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    scope_path: str
    ttl_seconds: int | None = None
    created_by: str | None = None


class BrainBlackboardReadConfig(BaseModel, frozen=True):
    """Config for brain_blackboard_read tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    ids: list[str]


class BrainKGQueryConfig(BaseModel, frozen=True):
    """Config for brain_kg_query tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    entity: str | None = None
    direction: Literal["both", "inbound", "outbound"] = "both"
    depth: int = Field(default=1, ge=1, le=5)


class BrainUpsertConfig(BaseModel, frozen=True):
    """Config for brain_upsert tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    collection: str = "default"
    metadata_schema: JsonMetadata | None = None
