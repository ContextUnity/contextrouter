"""Tests for Phase 6.8 template aliasing."""

from contextunity.router.cortex.compiler.builder import build_from_template


def test_alias_rag_retrieval():
    """rag_retrieval aliases to retrieval_augmented with default vector source."""
    graph = build_from_template("rag_retrieval")
    # Actually build_from_template returns a compiled graph, but maybe we can just make sure it compiles
    # and has the retrieval_augmented nodes.
    assert graph is not None
    assert "retrieve" in graph.nodes


def test_alias_sql_analytics():
    """sql_analytics aliases to retrieval_augmented with default sql source."""
    graph = build_from_template("sql_analytics")
    assert graph is not None
    assert "plan" in graph.nodes
