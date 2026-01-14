# Named Entity Recognition (NER) in ContextRouter

## What is Named Entity Recognition?

**Named Entity Recognition (NER)** is a natural language processing technique that automatically identifies and classifies named entities in text. Unlike simple entity extraction (which already exists in ContextRouter for knowledge graphs), NER adds **entity types**, allowing you to distinguish:

- **PERSON** — people, characters
- **ORG** — organizations, companies
- **LOC/GPE** — locations, geopolitical entities
- **DATE** — dates, times
- **MONEY** — monetary amounts
- **PRODUCT** — products, brands
- and other types

## Differences from Current Entity Extraction

ContextRouter already has entity extraction in `GraphBuilder`, but it:

1. **Focuses on relationships** — extracts entities together with relations for knowledge graphs
2. **Doesn't classify types** — doesn't distinguish PERSON vs ORG vs LOC
3. **Used for indexing** — works during ingestion

NER transformer:

1. **Adds entity types** — each entity has a type (PERSON, ORG, etc.)
2. **Preserves positions** — knows where in the text the entity is located
3. **Can be used at runtime** — works with any text, not just during ingestion
4. **Supports multiple backends** — LLM, spaCy, transformers

## How to Use NER in ContextRouter

### 1. Basic Usage

```python
from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.modules.transformers.ner import NERTransformer

# Create transformer
transformer = NERTransformer()
transformer.configure({
    "mode": "llm",  # or "spacy", "transformers"
})

# Process document
envelope = BisquitEnvelope(
    content={"content": "Apple Inc. was founded by Steve Jobs in Cupertino."},
    metadata={},
)

enriched = await transformer.transform(envelope)

# Access entities
entities = enriched.metadata["ner_entities"]
# [
#   {"text": "Apple Inc.", "entity_type": "ORG", "start": 0, "end": 10, "confidence": 1.0},
#   {"text": "Steve Jobs", "entity_type": "PERSON", "start": 30, "end": 40, "confidence": 1.0},
#   {"text": "Cupertino", "entity_type": "GPE", "start": 44, "end": 53, "confidence": 1.0},
# ]
```

### 2. Filtering by Entity Types

```python
transformer.configure({
    "mode": "llm",
    "entity_types": ["PERSON", "ORG"],  # Only these types
})
```

### 3. Using Different Backends

#### LLM (default, highest quality)
```python
transformer.configure({"mode": "llm"})
```

#### spaCy (fast, offline, requires model installation)
```python
# Install: python -m spacy download en_core_web_sm
# or: python -m spacy download uk_core_news_sm
transformer.configure({"mode": "spacy"})
```

#### Transformers (balance of quality and speed)
```python
transformer.configure({
    "mode": "transformers",
    "min_confidence": 0.7,  # Minimum confidence
})
```

### 4. Integration into Ingestion Pipeline

NER transformer can be added to the ingestion pipeline via registry:

```python
from contextrouter.core.registry import ComponentFactory

# Create and configure
ner_transformer = ComponentFactory.create_transformer(
    "ner",
    mode="llm",
    entity_types=["PERSON", "ORG", "LOC"],
)

# Use in pipeline
enriched = await ner_transformer.transform(envelope)
```

### 5. Using in Retrieval to Improve Search

NER can be used to extract entities from user queries and improve search:

```python
# In detect_intent or extract_query step
async def enhance_query_with_ner(state: AgentState) -> dict[str, object]:
    user_query = state.get("user_query", "")

    transformer = NERTransformer()
    transformer.configure({"mode": "llm"})

    envelope = BisquitEnvelope(
        content={"content": user_query},
        metadata={},
    )

    enriched = await transformer.transform(envelope)
    entities = enriched.metadata.get("ner_entities", [])

    # Add entity names to retrieval queries
    entity_names = [e["text"] for e in entities if e["entity_type"] in ["PERSON", "ORG"]]

    retrieval_queries = state.get("retrieval_queries", [])
    if entity_names:
        retrieval_queries.extend(entity_names)

    return {"retrieval_queries": retrieval_queries[:3]}
```

## Data Structure

After processing with NER transformer, metadata contains:

```python
{
    "ner_entities": [
        {
            "text": "Apple Inc.",
            "entity_type": "ORG",
            "start": 0,
            "end": 10,
            "confidence": 1.0
        },
        # ...
    ],
    "ner_entities_by_type": {
        "ORG": [...],
        "PERSON": [...],
        "GPE": [...],
    },
    "ner_entity_count": 5,
    "ner_mode": "llm"
}
```

## Standard Entity Types

- **PERSON** — people
- **ORG** — organizations
- **GPE** — geopolitical entities (countries, cities)
- **LOC** — locations (non-GPE)
- **DATE** — dates
- **MONEY** — monetary amounts
- **PERCENT** — percentages
- **QUANTITY** — measurements
- **EVENT** — events
- **PRODUCT** — products
- **LAW** — laws
- **LANGUAGE** — languages
- **WORK_OF_ART** — works of art
- **FAC** — facilities
- **NORP** — nationalities, religious/political groups

## Usage Examples

See `examples/ner_usage.py` for complete examples.

## Installing Dependencies

### For LLM Mode (default)
No additional installation needed — uses existing model registry.

### For spaCy Mode
```bash
uv add spacy
python -m spacy download en_core_web_sm  # English
python -m spacy download uk_core_news_sm  # Ukrainian
```

### For Transformers Mode
```bash
uv add transformers torch
```

## Integration with Existing Components

NER transformer integrates with:

1. **Ingestion pipeline** — adds entities to document metadata
2. **Retrieval pipeline** — can improve search queries
3. **Graph builder** — can use entity types for better relationship extraction
4. **Citation system** — can highlight entities in citations

## Future Improvements

- Support for custom entity types
- Caching of NER results
- Batch processing for improved performance
- Integration with knowledge graph for automatic entity linking
