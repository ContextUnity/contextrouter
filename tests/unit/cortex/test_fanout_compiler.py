"""Tests for fan-out compiler (Phase 6.3)."""

from contextunity.router.cortex.compiler.state import create_condition


def test_regular_condition_route():
    """If condition returns a normal string, route as normal."""
    cond_map = {"skip_retrieve": "generate"}
    condition_fn = create_condition("intent_route", cond_map)

    state = {"final_output": {"intent_route": "skip_retrieve"}}
    result = condition_fn(state)
    assert result == "skip_retrieve"


def test_fan_out_condition():
    """Generates Send() objects for each selected source when intent_route='selected_sources_fanout'."""
    cond_map = {"vector": "retrieve", "sql": "plan"}
    condition_fn = create_condition("intent_route", cond_map)

    state = {
        "final_output": {"intent_route": "selected_sources_fanout"},
        "selected_sources": ["wiki_vector", "sales_sql"],
        "config": {
            "data_sources": [
                {"binding": "wiki_vector", "type": "vector"},
                {"binding": "sales_sql", "type": "sql"},
            ]
        },
    }

    result = condition_fn(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].node == "retrieve"
    assert result[0].arg == {"__source_binding__": "wiki_vector"}
    assert result[1].node == "plan"
    assert result[1].arg == {"__source_binding__": "sales_sql"}


def test_fan_out_empty_sources_fallback():
    """If 'selected_sources_fanout' is requested but no sources selected, fallback to no_results."""
    cond_map = {"vector": "retrieve", "sql": "plan"}
    condition_fn = create_condition("intent_route", cond_map)

    state = {
        "final_output": {"intent_route": "selected_sources_fanout"},
        "selected_sources": [],
        "config": {"data_sources": []},
    }

    result = condition_fn(state)
    assert result == "no_results"
