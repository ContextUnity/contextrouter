"""Example: Using Named Entity Recognition (NER) in ContextRouter.

This example demonstrates how to:
1. Extract named entities from documents using NER transformer
2. Use different backends (LLM, spaCy, transformers)
3. Access extracted entities in metadata
"""

from __future__ import annotations

import asyncio

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.modules.transformers.ner import NERTransformer


async def example_llm_ner():
    """Example: Extract entities using LLM backend."""
    transformer = NERTransformer()
    transformer.configure(
        {
            "mode": "llm",
            "entity_types": ["PERSON", "ORG", "LOC", "DATE"],  # Optional filter
        }
    )

    # Create a sample document
    text = """
    Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.
    The company is known for innovative products like the iPhone and MacBook.
    Tim Cook became CEO in 2011 after Jobs' resignation.
    """

    envelope = BisquitEnvelope(
        content={"content": text},
        metadata={"source": "example"},
    )

    # Transform
    enriched = await transformer.transform(envelope)

    # Access entities
    entities = enriched.metadata.get("ner_entities", [])
    entities_by_type = enriched.metadata.get("ner_entities_by_type", {})

    print("=== LLM NER Results ===")
    print(f"Total entities: {len(entities)}")
    print("\nEntities by type:")
    for entity_type, ents in entities_by_type.items():
        print(f"  {entity_type}: {len(ents)}")
        for ent in ents[:3]:  # Show first 3
            print(f"    - {ent['text']}")

    return enriched


async def example_spacy_ner():
    """Example: Extract entities using spaCy backend (requires spaCy installation)."""
    try:
        transformer = NERTransformer()
        transformer.configure(
            {
                "mode": "spacy",
                "min_confidence": 0.5,
            }
        )

        text = """
        Microsoft Corporation is headquartered in Redmond, Washington.
        Satya Nadella has been CEO since 2014.
        The company was founded by Bill Gates and Paul Allen in 1975.
        """

        envelope = BisquitEnvelope(
            content={"content": text},
            metadata={"source": "example"},
        )

        enriched = await transformer.transform(envelope)
        entities = enriched.metadata.get("ner_entities", [])

        print("\n=== spaCy NER Results ===")
        print(f"Total entities: {len(entities)}")
        for ent in entities:
            print(f"  {ent['entity_type']}: {ent['text']}")

        return enriched
    except Exception as e:
        print(f"spaCy NER failed (may need installation): {e}")
        return None


async def example_integration_with_retrieval():
    """Example: Using NER in retrieval pipeline to enhance search queries."""
    # This shows how NER could be integrated into the retrieval step
    # to extract entities from user queries and use them for better search

    transformer = NERTransformer()
    transformer.configure({"mode": "llm"})

    user_query = "Tell me about Apple's products and Tim Cook's leadership"

    envelope = BisquitEnvelope(
        content={"content": user_query},
        metadata={"source": "user_query"},
    )

    enriched = await transformer.transform(envelope)
    entities = enriched.metadata.get("ner_entities", [])

    # Extract entity names for search enhancement
    entity_names = [ent["text"] for ent in entities if ent["entity_type"] in ["ORG", "PERSON"]]

    print("\n=== Query Enhancement ===")
    print(f"Original query: {user_query}")
    print(f"Extracted entities: {entity_names}")
    print(f"Enhanced query could be: {user_query} {' '.join(entity_names)}")

    return enriched


async def main():
    """Run all examples."""
    print("Named Entity Recognition (NER) Examples\n")

    # LLM-based extraction (always works)
    await example_llm_ner()

    # spaCy-based extraction (requires installation)
    await example_spacy_ner()

    # Integration example
    await example_integration_with_retrieval()


if __name__ == "__main__":
    asyncio.run(main())
