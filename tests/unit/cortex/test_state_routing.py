"""Tests for state routing — dynamic state with input/output key routing.

TDD: RED phase — tests written before implementation.
"""


class TestReadStateInput:
    """Test reading from state via configurable input keys."""

    def test_read_from_dynamic_key(self):
        """Reads from state['dynamic'][key] when key exists there."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input,
        )

        state = {"dynamic": {"user_query": "What is RAG?"}}
        result = read_state_input(state, "user_query")
        assert result == "What is RAG?"

    def test_fallback_to_top_level_key(self):
        """Falls back to state[key] when dynamic doesn't have it."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input,
        )

        state = {"messages": [{"role": "user", "content": "hello"}]}
        result = read_state_input(state, "messages")
        assert result == [{"role": "user", "content": "hello"}]

    def test_dynamic_takes_priority_over_top_level(self):
        """dynamic[key] wins over state[key] when both exist."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input,
        )

        state = {
            "dynamic": {"data": "from_dynamic"},
            "data": "from_top_level",
        }
        result = read_state_input(state, "data")
        assert result == "from_dynamic"

    def test_missing_key_returns_none(self):
        """Returns None when key doesn't exist anywhere."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input,
        )

        state = {"dynamic": {}}
        result = read_state_input(state, "nonexistent")
        assert result is None

    def test_read_with_default_value(self):
        """Returns default when key is missing."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input,
        )

        state = {"dynamic": {}}
        result = read_state_input(state, "missing", default=[])
        assert result == []


class TestReadStateInputMapping:
    """Test multi-key input mapping."""

    def test_mapping_resolves_multiple_keys(self):
        """state_input_mapping resolves multiple keys from state."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input_mapping,
        )

        state = {
            "dynamic": {
                "user_query": "What is RAG?",
                "retrieved_docs": ["doc1", "doc2"],
            },
        }
        mapping = {"query": "user_query", "context": "retrieved_docs"}

        result = read_state_input_mapping(state, mapping)

        assert result == {
            "query": "What is RAG?",
            "context": ["doc1", "doc2"],
        }

    def test_mapping_with_missing_key_returns_none_for_it(self):
        """Missing keys in mapping get None value."""
        from contextunity.router.cortex.compiler.state_routing import (
            read_state_input_mapping,
        )

        state = {"dynamic": {"query": "hello"}}
        mapping = {"q": "query", "ctx": "missing_key"}

        result = read_state_input_mapping(state, mapping)
        assert result["q"] == "hello"
        assert result["ctx"] is None


class TestWriteStateOutput:
    """Test writing to state with dynamic routing."""

    def test_write_to_dynamic_key(self):
        """Writes result to dynamic[output_key]."""
        from contextunity.router.cortex.compiler.state_routing import (
            write_state_output,
        )

        result = write_state_output("analysis", {"score": 0.95})
        assert result == {"dynamic": {"analysis": {"score": 0.95}}}

    def test_write_append_mode(self):
        """state_append=True produces a list wrapper for LangGraph reducer."""
        from contextunity.router.cortex.compiler.state_routing import (
            write_state_output,
        )

        result = write_state_output("results", {"item": 1}, append=True)
        # For append mode, we wrap in a format that works with state update
        assert "dynamic" in result
        assert isinstance(result["dynamic"]["results"], list)
        assert result["dynamic"]["results"] == [{"item": 1}]

    def test_write_to_legacy_key(self):
        """Writing to a known legacy key (e.g. 'final_output') goes top-level."""
        from contextunity.router.cortex.compiler.state_routing import (
            write_state_output,
        )

        result = write_state_output(
            "final_output", {"data": "done"}, legacy_keys={"final_output", "messages"}
        )
        assert result == {"final_output": {"data": "done"}}
