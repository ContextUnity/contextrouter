import json

from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig, TaxonomySection


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_taxonomy_sampling_doc_coverage_is_deterministic(tmp_path):
    # Import module directly to avoid side-effect imports from
    # contextrouter.modules.ingestion.rag.processors.__init__ (keeps test hermetic).
    import contextrouter.modules.ingestion.rag.processors.taxonomy_builder as tb

    clean_text_dir = tmp_path / "clean_text"
    book_path = clean_text_dir / "book.jsonl"

    # One doc with 3 records to force start/middle/end.
    rows = [
        {
            "content": "START_MARKER alpha beta",
            "source_type": "book",
            "metadata": {"book_title": "Doc A"},
        },
        {
            "content": "MIDDLE_MARKER gamma delta",
            "source_type": "book",
            "metadata": {"book_title": "Doc A"},
        },
        {
            "content": "END_MARKER epsilon zeta",
            "source_type": "book",
            "metadata": {"book_title": "Doc A"},
        },
    ]
    _write_jsonl(book_path, rows)

    cfg = RagIngestionConfig(taxonomy=TaxonomySection(include_types=["book"]))

    s1 = tb._collect_clean_text_samples_from_dir(
        clean_text_dir=clean_text_dir, config=cfg, max_samples=50
    )
    s2 = tb._collect_clean_text_samples_from_dir(
        clean_text_dir=clean_text_dir, config=cfg, max_samples=50
    )

    assert [x.text for x in s1] == [x.text for x in s2]
    assert [x.doc_key for x in s1] == [x.doc_key for x in s2]

    all_text = "\n".join([x.text for x in s1])
    assert "START_MARKER" in all_text
    assert "MIDDLE_MARKER" in all_text
    assert "END_MARKER" in all_text


def test_parse_concepts_tsv_strips_prefix():
    """Test that _parse_concepts_tsv strips concepts[N] prefix from terms."""
    import contextrouter.modules.ingestion.rag.processors.taxonomy_builder as tb

    # Simulated LLM output with the bad format
    raw = """concepts[0]term:Self-belief\tmindset\t\tBelief in oneself
concepts[1]term:Growth mindset\tmindset\t\tMindset focused on growth
Persistence\taction\t\tKeeping going despite obstacles"""

    concepts = tb._parse_concepts_tsv(raw)

    # Should have parsed 3 concepts
    assert len(concepts) == 3

    terms = [c["term"] for c in concepts]
    # Verify the prefix was stripped
    assert "Self-belief" in terms
    assert "Growth mindset" in terms
    assert "Persistence" in terms
    # Verify the prefix is NOT in any term
    assert not any("concepts[" in t for t in terms)


def test_parse_concepts_tsv_rejects_garbage():
    """Test that _parse_concepts_tsv rejects garbage terms."""
    import contextrouter.modules.ingestion.rag.processors.taxonomy_builder as tb

    raw = """buyiton_amazon\tpromo\t\tBuy it
FREE\tpromo\t\tFree stuff
Mindset principles\tmindset\t\tGood term"""

    concepts = tb._parse_concepts_tsv(raw)

    terms = [c["term"] for c in concepts]
    # Garbage should be filtered
    assert "buyiton_amazon" not in terms
    assert "FREE" not in terms
    # Good term should be kept
    assert "Mindset principles" in terms
