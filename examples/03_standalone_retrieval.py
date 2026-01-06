"""Example of using the RetrievalPipeline standalone.

ContextRouter allows using individual modules without running the full LangGraph agent.
"""

import asyncio

from contextrouter.core.config import get_core_config
from contextrouter.modules.retrieval import RetrievalPipeline


async def main():
    # Ensure config is loaded (usually handled by the host app)
    config = get_core_config()
    print(f"Using default LLM: {config.models.default_llm}")

    # Initialize the retrieval pipeline
    pipeline = RetrievalPipeline()

    # Mock state for the pipeline
    state = {
        "user_query": "artificial intelligence trends 2024",
        "retrieval_queries": ["AI trends 2024", "generative AI future"],
        "enable_web_search": True,
    }

    print(f"Executing retrieval for: '{state['user_query']}'...")

    try:
        # Run the pipeline orchestration (retrieval -> rerank -> citations)
        result = await pipeline.execute(state)

        print(f"\nFound {len(result.retrieved_docs)} documents.")
        for idx, doc in enumerate(result.retrieved_docs[:3]):
            print(
                f"[{idx + 1}] {getattr(doc, 'title', 'No Title')} (Type: {getattr(doc, 'source_type', 'unknown')})"
            )

        print(f"\nGenerated {len(result.citations)} citations.")

    except Exception as e:
        print(f"Retrieval failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
