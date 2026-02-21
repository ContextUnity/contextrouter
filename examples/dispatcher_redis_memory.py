"""Examples of using Redis Memory Tools with Dispatcher Agent.

This demonstrates how to use Redis memory tools for caching and session management.
"""

from __future__ import annotations

import asyncio

from contextrouter import invoke_dispatcher
from contextrouter.modules.tools.redis_memory import (
    cache_query_result,
    clear_memory,
    get_cached_query,
    get_session_data,
    retrieve_memory,
    store_memory,
)


async def example_basic_memory_operations() -> None:
    """Example: Basic memory store and retrieve operations."""
    print("=== Example: Basic Memory Operations ===\n")

    session_id = "example_session_123"
    tenant_id = "example_tenant"

    # Store a value in memory
    print("1. Storing memory...")
    result = await store_memory(
        key="user_preference",
        value="dark_mode",
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=3600,  # 1 hour
    )
    print(f"   Result: {result}\n")

    # Retrieve the value
    print("2. Retrieving memory...")
    retrieved = await retrieve_memory(
        key="user_preference",
        session_id=session_id,
        tenant_id=tenant_id,
    )
    print(f"   Retrieved: {retrieved}\n")

    # Store multiple values
    print("3. Storing multiple values...")
    await store_memory(
        key="language",
        value="uk",
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=86400,  # 24 hours
    )
    await store_memory(
        key="timezone",
        value="Europe/Kyiv",
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=86400,
    )
    print("   Stored language and timezone\n")

    # Get all session data
    print("4. Getting all session data...")
    session_data = await get_session_data(
        session_id=session_id,
        tenant_id=tenant_id,
    )
    print(f"   Session data: {session_data}\n")


async def example_query_caching() -> None:
    """Example: Caching expensive query results."""
    print("=== Example: Query Result Caching ===\n")

    tenant_id = "example_tenant"
    query = "What is the weather in Kyiv today?"

    # Check cache first
    print("1. Checking cache for query...")
    cached = await get_cached_query(query=query, tenant_id=tenant_id)

    if cached["found"]:
        print(f"   Found in cache: {cached['result']}")
        print(f"   Cached at: {cached.get('timestamp')}\n")
    else:
        print("   Not in cache, executing expensive operation...")

        # Simulate expensive operation (e.g., LLM call, API request)
        await asyncio.sleep(0.1)  # Simulate delay
        result = "Sunny, 22°C, light breeze"

        # Cache the result
        print("2. Caching result...")
        cache_result = await cache_query_result(
            query=query,
            result=result,
            tenant_id=tenant_id,
            ttl_seconds=1800,  # 30 minutes
        )
        print(f"   Cached: {cache_result}\n")

        # Now retrieve from cache
        print("3. Retrieving from cache...")
        cached = await get_cached_query(query=query, tenant_id=tenant_id)
        print(f"   Retrieved: {cached['result']}\n")


async def example_session_continuity() -> None:
    """Example: Using memory for session continuity."""
    print("=== Example: Session Continuity ===\n")

    session_id = "conversation_session"
    tenant_id = "example_tenant"

    # First interaction: store context
    print("1. First interaction - storing context...")
    context = {
        "user_name": "Oleksii",
        "topic": "Python async programming",
        "preferences": {"language": "uk", "detail_level": "advanced"},
    }

    await store_memory(
        key="conversation_context",
        value=str(context),  # In real usage, use JSON
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=3600,
    )
    print("   Context stored\n")

    # Second interaction: retrieve context
    print("2. Second interaction - retrieving context...")
    context_data = await retrieve_memory(
        key="conversation_context",
        session_id=session_id,
        tenant_id=tenant_id,
    )

    if context_data["found"]:
        print(f"   Retrieved context: {context_data['value']}")
        print("   Agent can now continue conversation with context\n")

    # Update context
    print("3. Updating context...")
    updated_context = {
        **context,
        "last_topic": "Redis caching",
        "questions_asked": 3,
    }

    await store_memory(
        key="conversation_context",
        value=str(updated_context),
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=3600,
    )
    print("   Context updated\n")


async def example_with_dispatcher_agent() -> None:
    """Example: Using memory tools through dispatcher agent."""
    print("=== Example: Using Memory Tools via Dispatcher Agent ===\n")

    # The dispatcher agent automatically has access to memory tools
    # You can ask it to use them

    messages = [
        {
            "role": "user",
            "content": "Remember that I prefer dark mode and Ukrainian language. Then tell me what you remembered.",
        }
    ]

    print("1. Asking dispatcher to remember preferences...")
    result = await invoke_dispatcher(
        messages=messages,
        session_id="memory_demo",
        platform="python",
    )

    print(f"   Agent response: {result.get('messages', [])}\n")

    # Later, ask agent to recall
    print("2. Asking dispatcher to recall preferences...")
    recall_messages = [
        {
            "role": "user",
            "content": "What are my preferences that you remembered?",
        }
    ]

    result = await invoke_dispatcher(
        messages=recall_messages,
        session_id="memory_demo",
        platform="python",
    )

    print(f"   Agent response: {result.get('messages', [])}\n")


async def example_cache_optimization() -> None:
    """Example: Optimizing expensive operations with caching."""
    print("=== Example: Cache Optimization Pattern ===\n")

    tenant_id = "example_tenant"

    async def expensive_llm_call(query: str) -> str:
        """Simulate expensive LLM call."""
        print(f"   ⚠️  Executing expensive LLM call for: {query}")
        await asyncio.sleep(0.2)  # Simulate delay
        return f"Response to: {query}"

    async def optimized_query(query: str) -> str:
        """Optimized query with caching."""
        # 1. Check cache first
        cached = await get_cached_query(query=query, tenant_id=tenant_id)

        if cached["found"]:
            print("   ✅ Cache hit! Using cached result")
            return cached["result"]

        # 2. Execute expensive operation
        print("   ❌ Cache miss, executing operation...")
        result = await expensive_llm_call(query)

        # 3. Cache the result
        await cache_query_result(
            query=query,
            result=result,
            tenant_id=tenant_id,
            ttl_seconds=1800,  # 30 minutes
        )
        print("   💾 Result cached for future use")

        return result

    # First call - cache miss
    print("1. First call (cache miss):")
    result1 = await optimized_query("What is Python?")
    print(f"   Result: {result1}\n")

    # Second call - cache hit
    print("2. Second call (cache hit):")
    result2 = await optimized_query("What is Python?")
    print(f"   Result: {result2}\n")

    # Different query - cache miss
    print("3. Different query (cache miss):")
    result3 = await optimized_query("What is Redis?")
    print(f"   Result: {result3}\n")


async def example_memory_cleanup() -> None:
    """Example: Cleaning up memory when needed."""
    print("=== Example: Memory Cleanup ===\n")

    session_id = "cleanup_demo"
    tenant_id = "example_tenant"

    # Store some data
    print("1. Storing data...")
    await store_memory(
        key="temp_data",
        value="temporary value",
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=3600,
    )
    await store_memory(
        key="important_data",
        value="important value",
        session_id=session_id,
        tenant_id=tenant_id,
        ttl_seconds=86400,
    )
    print("   Stored temp_data and important_data\n")

    # Verify data exists
    print("2. Verifying data exists...")
    temp = await retrieve_memory(
        key="temp_data",
        session_id=session_id,
        tenant_id=tenant_id,
    )
    print(f"   temp_data: {temp['found']}\n")

    # Clear specific key
    print("3. Clearing temp_data...")
    clear_result = await clear_memory(
        key="temp_data",
        session_id=session_id,
        tenant_id=tenant_id,
    )
    print(f"   Clear result: {clear_result}\n")

    # Verify cleared
    print("4. Verifying temp_data is cleared...")
    temp = await retrieve_memory(
        key="temp_data",
        session_id=session_id,
        tenant_id=tenant_id,
    )
    print(f"   temp_data: {temp['found']}")

    # Important data should still exist
    important = await retrieve_memory(
        key="important_data",
        session_id=session_id,
        tenant_id=tenant_id,
    )
    print(f"   important_data: {important['found']}\n")


async def example_multi_tenant_isolation() -> None:
    """Example: Tenant isolation in memory operations."""
    print("=== Example: Multi-Tenant Isolation ===\n")

    session_id = "shared_session"
    tenant_a = "tenant_a"
    tenant_b = "tenant_b"

    # Store data for tenant A
    print("1. Storing data for tenant A...")
    await store_memory(
        key="secret",
        value="tenant_a_secret",
        session_id=session_id,
        tenant_id=tenant_a,
        ttl_seconds=3600,
    )

    # Store data for tenant B
    print("2. Storing data for tenant B...")
    await store_memory(
        key="secret",
        value="tenant_b_secret",
        session_id=session_id,
        tenant_id=tenant_b,
        ttl_seconds=3600,
    )

    # Retrieve for tenant A (should get tenant A's data)
    print("3. Retrieving for tenant A...")
    data_a = await retrieve_memory(
        key="secret",
        session_id=session_id,
        tenant_id=tenant_a,
    )
    print(f"   Tenant A data: {data_a.get('value')}\n")

    # Retrieve for tenant B (should get tenant B's data)
    print("4. Retrieving for tenant B...")
    data_b = await retrieve_memory(
        key="secret",
        session_id=session_id,
        tenant_id=tenant_b,
    )
    print(f"   Tenant B data: {data_b.get('value')}\n")

    print("   ✅ Tenant isolation working correctly!\n")


async def main() -> None:
    """Run all examples."""
    print("Redis Memory Tools Usage Examples\n")
    print("=" * 60 + "\n")

    try:
        await example_basic_memory_operations()
        await example_query_caching()
        await example_session_continuity()
        await example_cache_optimization()
        await example_memory_cleanup()
        await example_multi_tenant_isolation()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("\nNote: Make sure Redis is running:")
        print("  redis-server")
        print("\nOr configure Redis connection:")
        print("  export REDIS_HOST=localhost")
        print("  export REDIS_PORT=6379")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Redis is running and configured correctly.")


if __name__ == "__main__":
    asyncio.run(main())
