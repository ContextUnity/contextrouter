"""Brain Platform Tools — executors for compiled graph nodes.

Registers brain_search, brain_memory_read/write, brain_blackboard_read/write,
brain_kg_query, brain_upsert into PlatformToolRegistry.

Each tool wraps BrainClient SDK calls with proper tenant isolation.
"""

from ..helpers.registration import PlatformRegistry, ToolRegistrationSpec, register_tool_specs
from .executors import (
    brain_blackboard_read_executor,
    brain_blackboard_write_executor,
    brain_kg_query_executor,
    brain_memory_read_executor,
    brain_memory_write_executor,
    brain_search_executor,
    brain_upsert_executor,
)
from .schemas import (
    BrainBlackboardReadConfig,
    BrainBlackboardWriteConfig,
    BrainKGQueryConfig,
    BrainMemoryReadConfig,
    BrainMemoryWriteConfig,
    BrainSearchConfig,
    BrainUpsertConfig,
)


def register_brain_tools(registry: PlatformRegistry) -> None:
    """Register all Brain tools into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="brain_search",
                executor=brain_search_executor,
                config_schema=BrainSearchConfig,
                required_scopes=["brain:read"],
            ),
            ToolRegistrationSpec(
                binding="brain_memory_read",
                executor=brain_memory_read_executor,
                config_schema=BrainMemoryReadConfig,
                required_scopes=["memory:read"],
            ),
            ToolRegistrationSpec(
                binding="brain_memory_write",
                executor=brain_memory_write_executor,
                config_schema=BrainMemoryWriteConfig,
                required_scopes=["memory:write"],
            ),
            ToolRegistrationSpec(
                binding="brain_blackboard_write",
                executor=brain_blackboard_write_executor,
                config_schema=BrainBlackboardWriteConfig,
                required_scopes=["brain:write"],
            ),
            ToolRegistrationSpec(
                binding="brain_blackboard_read",
                executor=brain_blackboard_read_executor,
                config_schema=BrainBlackboardReadConfig,
                required_scopes=["brain:read"],
            ),
            ToolRegistrationSpec(
                binding="brain_kg_query",
                executor=brain_kg_query_executor,
                config_schema=BrainKGQueryConfig,
                required_scopes=["brain:read"],
            ),
            ToolRegistrationSpec(
                binding="brain_upsert",
                executor=brain_upsert_executor,
                config_schema=BrainUpsertConfig,
                required_scopes=["brain:write"],
            ),
        ],
    )


__all__ = [
    "register_brain_tools",
    "BrainSearchConfig",
    "BrainMemoryReadConfig",
    "BrainMemoryWriteConfig",
    "BrainBlackboardWriteConfig",
    "BrainBlackboardReadConfig",
    "BrainKGQueryConfig",
    "BrainUpsertConfig",
]
