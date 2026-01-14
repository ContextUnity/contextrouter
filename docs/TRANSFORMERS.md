# Additional Transformers for ContextRouter

This document describes useful transformers similar to NER that can enhance ContextRouter's capabilities.

## Recommended Transformers

### 1. Keyphrase Extraction Transformer

**Purpose**: Extract key phrases and terms from documents for better searchability and indexing.

**Use Cases**:
- Improve search query matching
- Generate document tags automatically
- Enhance retrieval relevance
- Create topic summaries

**Output**:
```python
{
    "keyphrases": [
        {"text": "machine learning", "score": 0.95, "rank": 1},
        {"text": "neural networks", "score": 0.87, "rank": 2},
    ],
    "keyphrase_count": 10,
    "keyphrases_by_length": {"short": [...], "medium": [...], "long": [...]}
}
```

**Backends**:
- LLM-based extraction
- YAKE (Yet Another Keyword Extractor)
- KeyBERT (BERT-based)
- Rake-nltk

**Priority**: ⭐⭐⭐⭐⭐ (Very High - directly improves search)

---

### 2. Sentiment Analysis Transformer

**Purpose**: Analyze sentiment (positive, negative, neutral) and emotional tone of text.

**Use Cases**:
- Filter documents by sentiment
- Understand user feedback tone
- Analyze product reviews
- Monitor brand sentiment

**Output**:
```python
{
    "sentiment": {
        "label": "positive",
        "score": 0.85,
        "confidence": 0.92
    },
    "emotions": {
        "joy": 0.7,
        "anger": 0.1,
        "sadness": 0.05
    },
    "sentiment_by_section": [...]  # For long documents
}
```

**Backends**:
- LLM-based analysis
- Transformers (RoBERTa, BERT sentiment models)
- TextBlob/VADER (fast, rule-based)

**Priority**: ⭐⭐⭐⭐ (High - useful for content filtering)

---

### 3. Language Detection Transformer

**Purpose**: Detect the language(s) of text content.

**Use Cases**:
- Route documents to language-specific pipelines
- Filter multilingual content
- Improve language-specific processing
- Validate content language

**Output**:
```python
{
    "language": "uk",
    "confidence": 0.98,
    "detected_languages": [
        {"code": "uk", "score": 0.98},
        {"code": "en", "score": 0.02}
    ],
    "is_multilingual": False
}
```

**Backends**:
- langdetect (fast, accurate)
- polyglot
- LLM-based detection
- fastText language identification

**Priority**: ⭐⭐⭐⭐⭐ (Very High - essential for multilingual support)

---

### 4. Topic Modeling Transformer

**Purpose**: Identify main topics and themes in documents.

**Use Cases**:
- Automatic categorization
- Topic-based document clustering
- Content organization
- Topic-aware retrieval

**Output**:
```python
{
    "topics": [
        {
            "topic_id": 0,
            "label": "Artificial Intelligence",
            "score": 0.75,
            "keywords": ["AI", "machine learning", "neural networks"]
        },
        {
            "topic_id": 1,
            "label": "Software Development",
            "score": 0.45,
            "keywords": ["programming", "code", "development"]
        }
    ],
    "primary_topic": 0,
    "topic_distribution": [0.75, 0.45, ...]
}
```

**Backends**:
- LDA (Latent Dirichlet Allocation)
- BERTopic (modern, high-quality)
- LLM-based topic extraction
- NMF (Non-negative Matrix Factorization)

**Priority**: ⭐⭐⭐⭐ (High - useful for organization)

---

### 5. Text Classification Transformer

**Purpose**: Classify documents into predefined categories.

**Use Cases**:
- Route documents to appropriate handlers
- Filter content by category
- Organize knowledge base
- Content moderation

**Output**:
```python
{
    "category": "technical_documentation",
    "confidence": 0.92,
    "all_categories": [
        {"label": "technical_documentation", "score": 0.92},
        {"label": "tutorial", "score": 0.15},
        {"label": "reference", "score": 0.08}
    ],
    "is_confident": True
}
```

**Backends**:
- LLM-based classification
- Fine-tuned BERT/RoBERTa classifiers
- Zero-shot classification models
- Rule-based classification

**Priority**: ⭐⭐⭐⭐ (High - useful for routing)

---

### 6. Coreference Resolution Transformer

**Purpose**: Resolve pronouns and references to their actual entities.

**Use Cases**:
- Improve entity linking in knowledge graphs
- Better understanding of document context
- Enhance question answering
- Improve summarization

**Output**:
```python
{
    "coreferences": [
        {
            "mention": "he",
            "resolved_to": "Steve Jobs",
            "confidence": 0.95,
            "position": {"start": 120, "end": 122}
        },
        {
            "mention": "the company",
            "resolved_to": "Apple Inc.",
            "confidence": 0.88,
            "position": {"start": 200, "end": 211}
        }
    ],
    "resolved_text": "Steve Jobs founded Apple Inc. He was CEO..."
}
```

**Backends**:
- spaCy coreference resolution
- NeuralCoref
- LLM-based resolution
- AllenNLP coref models

**Priority**: ⭐⭐⭐ (Medium - enhances other features)

---

### 7. Relation Extraction Transformer

**Purpose**: Extract relationships between entities (complements NER and GraphBuilder).

**Use Cases**:
- Build richer knowledge graphs
- Understand entity relationships
- Improve retrieval with relationship context
- Answer relationship questions

**Output**:
```python
{
    "relations": [
        {
            "subject": "Steve Jobs",
            "predicate": "founded",
            "object": "Apple Inc.",
            "confidence": 0.95,
            "type": "FOUNDED_BY"
        },
        {
            "subject": "Apple Inc.",
            "predicate": "located_in",
            "object": "Cupertino",
            "confidence": 0.88,
            "type": "LOCATED_IN"
        }
    ],
    "relation_types": ["FOUNDED_BY", "LOCATED_IN", "CEO_OF"],
    "relation_count": 5
}
```

**Backends**:
- LLM-based extraction
- OpenIE (Open Information Extraction)
- REBEL (Relation Extraction)
- Custom relation models

**Priority**: ⭐⭐⭐⭐ (High - complements existing graph builder)

---

### 8. Document Embedding Transformer

**Purpose**: Generate embeddings for documents and store them in metadata.

**Use Cases**:
- Semantic search enhancement
- Similarity calculations
- Clustering documents
- Fast retrieval

**Output**:
```python
{
    "embedding": [0.123, -0.456, ...],  # Vector of floats
    "embedding_model": "text-embedding-3-large",
    "embedding_dim": 3072,
    "embedding_hash": "sha256:..."
}
```

**Backends**:
- Existing embedding models in ContextRouter
- Sentence transformers
- OpenAI embeddings
- Vertex AI embeddings

**Priority**: ⭐⭐⭐⭐⭐ (Very High - essential for semantic search)

---

## Implementation Priority

Based on ContextRouter's architecture and use cases:

1. **Keyphrase Extraction** - Directly improves search quality
2. **Language Detection** - Essential for multilingual support
3. **Document Embedding** - Core for semantic search
4. **Sentiment Analysis** - Useful for content filtering
5. **Topic Modeling** - Helps with organization
6. **Text Classification** - Useful for routing
7. **Relation Extraction** - Complements graph builder
8. **Coreference Resolution** - Enhances other features

## Integration Pattern

All transformers follow the same pattern as NER:

```python
@register_transformer("keyphrase_extractor")
class KeyphraseTransformer(Transformer):
    name = "keyphrase_extractor"

    def configure(self, params: dict[str, Any] | None) -> None:
        # Configuration logic
        pass

    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        # Extract text
        # Process with backend
        # Store in metadata
        # Return enriched envelope
        pass
```

## Combining Transformers

Transformers can be chained in the ingestion pipeline:

```python
# Example pipeline
envelope = await keyphrase_transformer.transform(envelope)
envelope = await ner_transformer.transform(envelope)
envelope = await language_detector.transform(envelope)
envelope = await sentiment_analyzer.transform(envelope)
```

## Performance Considerations

- **LLM-based**: High quality, slower, costs tokens
- **Local models**: Fast, offline, may need GPU
- **Rule-based**: Very fast, limited accuracy
- **Hybrid**: Best balance (try local first, fallback to LLM)

## Next Steps

1. Implement Keyphrase Extraction transformer (highest priority)
2. Implement Language Detection transformer (essential)
3. Enhance Document Embedding transformer (if not already complete)
4. Add Sentiment Analysis for content filtering
5. Consider Topic Modeling for better organization
