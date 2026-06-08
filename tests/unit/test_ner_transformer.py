"""Tests for NER transformer — pure logic, config, and entity extraction.

Tests _normalize_entity_type, NERConfig, NERTransformer.configure,
NERTransformer.transform (metadata enrichment), and entity_types filtering.
No spaCy/transformers/LLM dependency — uses monkeypatched extraction.
"""

from __future__ import annotations

import pytest
from contextunity.core import ContextUnit

from contextunity.router.modules.transformers.ner import (
    STANDARD_ENTITY_TYPES,
    NEREntity,
    NERTransformer,
    _normalize_entity_type,
)

# ── Tests: _normalize_entity_type ────────────────────────────────────────


class TestNormalizeEntityType:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("PER", "PERSON"),
            ("per", "PERSON"),
            ("ORG", "ORG"),
            ("org", "ORG"),
            ("LOC", "LOC"),
            ("GPE", "GPE"),
            ("MISC", "MISC"),
            ("DATE", "DATE"),
            ("MONEY", "MONEY"),
            ("PRODUCT", "PRODUCT"),
            ("NORP", "NORP"),
            ("WORK_OF_ART", "WORK_OF_ART"),
        ],
    )
    def test_known_aliases(self, raw, expected):
        assert _normalize_entity_type(raw) == expected

    def test_unknown_type_uppercased(self):
        assert _normalize_entity_type("custom_type") == "CUSTOM_TYPE"


class TestNERTransformerConfigure:
    def test_configure_sets_mode(self):
        t = NERTransformer()
        t.configure({"mode": "spacy"})
        assert t.mode == "spacy"

    def test_entity_types_normalized(self):
        t = NERTransformer()
        t.configure({"entity_types": ["per", "org"]})
        assert t.entity_types == {"PERSON", "ORG"}


# ── Tests: NERTransformer.transform (metadata enrichment) ───────────────


class TestNERTransformerTransform:
    @pytest.fixture()
    def transformer(self):
        t = NERTransformer()
        t.configure({"mode": "llm"})
        return t

    def _fake_entities(self) -> list[NEREntity]:
        return [
            {
                "text": "Kyiv",
                "entity_type": "GPE",
                "start": 0,
                "end": 4,
                "confidence": 1.0,
                "source": "llm",
            },
            {
                "text": "ContextUnity",
                "entity_type": "ORG",
                "start": 10,
                "end": 22,
                "confidence": 0.95,
                "source": "llm",
            },
        ]

    @pytest.mark.asyncio
    async def test_enriches_metadata_with_entities(self, transformer, monkeypatch):
        """Extracted entities are written to metadata."""
        entities = self._fake_entities()
        monkeypatch.setattr(transformer, "_extract_with_llm", lambda text: entities)
        # Need to make the monkeypatched function a coroutine

        async def fake_extract(text):
            return entities

        monkeypatch.setattr(transformer, "_extract_with_llm", fake_extract)

        unit = ContextUnit(payload={"content": "Kyiv based ContextUnity platform"})
        result = await transformer.transform(unit)

        metadata = result.payload["metadata"]
        assert metadata["ner_entity_count"] == 2
        assert metadata["ner_mode"] == "llm"
        assert len(metadata["ner_entities"]) == 2
        assert "GPE" in metadata["ner_entities_by_type"]
        assert "ORG" in metadata["ner_entities_by_type"]

    @pytest.mark.asyncio
    async def test_skips_short_content(self, transformer):
        """Content < 10 chars is skipped."""
        unit = ContextUnit(payload={"content": "Hi"})
        result = await transformer.transform(unit)
        metadata = result.payload.get("metadata", {})
        assert "ner_entities" not in metadata

    @pytest.mark.asyncio
    async def test_skips_empty_content(self, transformer):
        """Empty content is skipped."""
        unit = ContextUnit(payload={"content": ""})
        result = await transformer.transform(unit)
        metadata = result.payload.get("metadata", {})
        assert "ner_entities" not in metadata

    @pytest.mark.asyncio
    async def test_handles_dict_content(self, transformer, monkeypatch):
        """Content as dict with nested 'content' key is supported."""
        entities = self._fake_entities()[:1]

        async def fake_extract(text):
            return entities

        monkeypatch.setattr(transformer, "_extract_with_llm", fake_extract)

        unit = ContextUnit(payload={"content": {"content": "Kyiv is the capital of Ukraine"}})
        result = await transformer.transform(unit)
        assert result.payload["metadata"]["ner_entity_count"] == 1

    @pytest.mark.asyncio
    async def test_no_entities_returns_unchanged(self, transformer, monkeypatch):
        """When no entities are extracted, metadata is not modified."""

        async def fake_extract(text):
            return []

        monkeypatch.setattr(transformer, "_extract_with_llm", fake_extract)

        unit = ContextUnit(payload={"content": "This is a plain text with nothing"})
        result = await transformer.transform(unit)
        metadata = result.payload.get("metadata", {})
        assert "ner_entities" not in metadata

    @pytest.mark.asyncio
    async def test_struct_data_propagation(self, transformer, monkeypatch):
        """Entities are also written to struct_data if present."""
        entities = self._fake_entities()

        async def fake_extract(text):
            return entities

        monkeypatch.setattr(transformer, "_extract_with_llm", fake_extract)

        unit = ContextUnit(
            payload={
                "content": "Kyiv based ContextUnity platform",
                "metadata": {"struct_data": {"existing_key": "value"}},
            }
        )
        result = await transformer.transform(unit)
        struct_data = result.payload["metadata"]["struct_data"]
        assert "ner_entities" in struct_data
        assert struct_data["existing_key"] == "value"


# ── Tests: STANDARD_ENTITY_TYPES constant ────────────────────────────────


class TestStandardEntityTypes:
    def test_contains_core_types(self):
        assert "PERSON" in STANDARD_ENTITY_TYPES
        assert "ORG" in STANDARD_ENTITY_TYPES
        assert "GPE" in STANDARD_ENTITY_TYPES
        assert "LOC" in STANDARD_ENTITY_TYPES
        assert "DATE" in STANDARD_ENTITY_TYPES
